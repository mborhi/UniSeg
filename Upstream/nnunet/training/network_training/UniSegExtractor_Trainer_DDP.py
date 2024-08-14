# -*- coding:utf-8 _*-
# @author: ywye
# @contact: ywye@mail.nwpu.edu.cn
# @version: 0.1.0
# @file: UniSeg with nnunet version
# @time: 2022/11/29
import numpy as np
import itertools
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.cuda.amp import autocast
from nnunet.training.network_training.nnUNetTrainerV2_DDP import nnUNetTrainerV2_DDP
import torch
import torch.distributed as dist
from torch.optim import lr_scheduler
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn, distributed
import SimpleITK as sitk
import shutil
from nnunet.network_architecture.Prompt_generic_UNet_DP import UniSeg_model, UniSegExtractor_DP, DynamicDistributionModel_DP
from nnunet.training.dataloading.dataset_loading import DataLoader3D_UniSeg
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
import traceback
from time import sleep
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from tqdm import tqdm
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation_uniseg
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.loss_functions.dist_match_loss import DynamicDistMatchingLoss
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss, SoftDiceLoss, get_tp_fp_fn_tn
from torch.nn.parallel import DistributedDataParallel as DDP
# from nnunet.utilities.sep import get_task_set, extract_task_set
from nnunet.utilities.tensor_utilities import sum_tensor
import copy 
import wandb

class UniSegExtractor_Trainer_DDP(nnUNetTrainerV2_DDP):
    def __init__(self, plans_file, fold, local_rank, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, distribute_batch_size=False, fp16=False, 
                 feature_space_dim=32, loss_type="kl", update_iter=1, queue_size=5000, max_num_epochs=1000, 
                 batch_size=2, num_gpus=1, single_task=False):
        super().__init__(plans_file, fold, local_rank, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, distribute_batch_size, fp16)
        self.max_num_epochs = max_num_epochs
        self.task = {"live":0, "kidn":1, "hepa":2, "panc":3, "colo":4, "lung":5, "sple":6, "sub-":7, "pros":8, "BraT":9}
        self.task_class = {0: 3, 1: 3, 2: 3, 3: 3, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 4}
        self.task_id_class_lst_mapping = {
            0: [0, 1, 2], 
            1: [0, 3, 4], 
            2: [0, 5, 6], 
            3: [0, 7, 8], 
            4: [0, 9], 
            5: [0, 10], 
            6: [0, 11], 
            7: [0, 12], 
            8: [0, 13], 
            9: [0, 14, 15, 16], 
        }
        if single_task:
            self.task = { "pros":0, }
            self.task_class = {0: 2}
            self.task_id_class_lst_mapping = {0: [0, 1]}
        self.class_lst_task_id_mapping = {}
        self.class_lst_to_std_mapping = {}
        for task_id, cls_lst in self.task_id_class_lst_mapping.items():
            for i, cls in enumerate(cls_lst):
                self.class_lst_task_id_mapping[cls] = task_id
                self.class_lst_to_std_mapping[cls] = i
                
        print("task_class", self.task_class)
        self.visual_epoch = -1
        self.total_task_num = 10 if not single_task else 1 # NOTE
        self.batch_size = batch_size
        self.num_batches_per_epoch = 1000 // (num_gpus * self.batch_size) #int((50 // num_gpus) * self.total_task_num)
        self.num_val_batches_per_epoch = self.num_batches_per_epoch // 5
        print("num batches per epoch:", self.num_batches_per_epoch)
        print("num batches per val epoch:", self.num_val_batches_per_epoch)
        print("total task num", self.total_task_num)
        print(os.getcwd())
        # if os.path.exists(os.path.join(self.output_folder, "code")):
        #     shutil.rmtree(os.path.join(self.output_folder, "code"))
        # dirname, _ = os.path.split(os.path.abspath(__file__))
        # shutil.copytree(os.path.join(dirname.split("nnunet")[0], "nnunet"), os.path.join(self.output_folder, "code"))
        print("copy code successfully!")
        self.task_index = [0 for _ in range(self.total_task_num)]
        ### Distribution Matching
        self.update_target_iter = update_iter
        self.feature_space_dim = feature_space_dim
        self.queue_size = queue_size
        self.num_components = len(self.class_lst_to_std_mapping.keys())
        self.return_est_dists = False
        self.loss_type = loss_type

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """


        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d
        # TODO: change last layer nonlin to identity
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        # NOTE
        self.base_num_features = self.feature_space_dim
        # self.uniseg_network = UniSeg_model(self.patch_size, self.total_task_num, [1, 2, 4], self.base_num_features, self.num_classes,
        #                             len(self.net_num_pool_op_kernel_sizes),
        #                             self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
        #                             dropout_op_kwargs,
        #                             net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
        #                             self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        uniseg_args = (self.patch_size, self.total_task_num, [1, 2, 4], self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        uniseg_kwargs = {}

        # self.network.inference_apply_nonlin = softmax_helper
        # num_components = len(self.class_lst_to_std_mapping.keys())
        self.network = UniSegExtractor_DP(self.base_num_features, 
                                          self.num_components,  
                                          self.class_lst_to_std_mapping, 
                                          self.task_id_class_lst_mapping,
                                          *uniseg_args, 
                                          with_wandb=self.with_wandb,
                                        #   **uniseg_kwargs
                                        )
        # pi = [55, 56, 57, 58, 110, 111, 112, 113, 114]
        pi = [53, 54, 55, 56, 57, 58,]
        for i, (p_name, p) in enumerate(self.network.named_parameters()):
            if i in pi:
                p.requires_grad = False
                print(f"{i}: {p_name}")
                self.print_to_log_file(f"{i}: {p_name}")
            # elif i == 53 or i == 54:
            #     print(f"{i}: {p_name}")

        
        # print(1 /0)
        self.network.inference_apply_nonlin = lambda x: x
        if torch.cuda.is_available():
            self.network.cuda()

        self.tp_dim = int(self.network.intermediate_prompt.numel() / self.network.num_class)
        # Isn't actually DP, onyl used for further DP modules
        self.dynamic_dist_network = DynamicDistributionModel_DP(self.feature_space_dim, self.tp_dim, self.num_components, momentum=0.999, queue_size=self.queue_size)
        if torch.cuda.is_available():
            self.dynamic_dist_network.cuda()

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        # self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                                 amsgrad=True)
        self.optimizer = torch.optim.Adam(itertools.chain(self.network.parameters(), self.dynamic_dist_network.parameters()), self.initial_lr, weight_decay=self.weight_decay,
                                        amsgrad=True)
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2,
                                                        patience=self.lr_scheduler_patience,
                                                        verbose=True, threshold=self.lr_scheduler_eps,
                                                        threshold_mode="abs")

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            # self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            # self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            self.dc_loss = SoftDiceLoss(batch_dice=True, smooth=1e-5, do_bg=True, apply_nonlin=None, do_one_hot=False)
            
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    if self.local_rank == 0:
                        print("unpacking dataset")
                        unpack_dataset(self.folder_with_preprocessed_data)
                        print("done")
                    distributed.barrier()
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation_uniseg(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                    task_num=self.total_task_num, iter_each_task_epoch=int(self.num_batches_per_epoch // self.total_task_num)
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            ######  Initialize target distributions #####
            torch.cuda.manual_seed_all(42)
            max_var = 0.001
            delta=1-(1e-03)
            # NOTE just use linspace here
            min_dist = 100. #self.distance_bounds(k=num_components, delta=delta, max_var=max_var*100)
            # Multivariate
            unscaled_means = torch.stack([min_dist * torch.randint(0, self.num_components, size=(self.feature_space_dim,)) for _ in range(self.num_components)])
            max_norm = torch.max(torch.norm(unscaled_means, dim=1, p=2))
            self.mus = unscaled_means / max_norm
            # self.mus = torch.linspace(0, 1, num_components, requires_grad=False)[:, None]
            min_dist = torch.cdist(self.mus, self.mus, p=2).fill_diagonal_(torch.inf).min().item()
            
            # self.vars = torch.full_like(self.mus, fill_value=max_var, requires_grad=False)
            self.sigs = torch.stack([
                max_var * torch.eye(self.feature_space_dim, dtype=self.mus.dtype) for _ in range(self.num_components)
            ])
            # self.sigs = (unscaled_vars - 1e-05) / (1e-03 - 1e-05)
            self.input_vars = max_var * torch.ones((1, self.num_components), dtype=self.mus.dtype)

            self.print_to_log_file(f"initial means: {self.mus}")
            self.print_to_log_file(f"initial vars: {self.sigs}")

            self.min_dist = min_dist
            if torch.cuda.is_available():
                self.mus = self.mus.cuda()
                self.sigs = self.sigs.cuda()
                

            ##### END ####
            if wandb.run is not None:
                self.with_wandb = True 
            else:
                self.with_wandb = False
            self.initialize_network()
            self.loss = DynamicDistMatchingLoss(self.min_dist, loss_type=self.loss_type, with_wandb=self.with_wandb) # NOTE
            self.initialize_optimizer_and_scheduler()
            self.network = DDP(self.network, device_ids=[self.local_rank])

            # assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')


        self.was_initialized = True

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        keys = data_dict['keys']
        task_id = np.zeros(keys.shape)
        for itr in range(keys.shape[0]):
            task_id[itr] = self.task[keys[itr][:4]]

        if do_backprop:
            self.task_index[int(task_id[0])] += 1

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        task_id = maybe_to_torch(task_id).long()


        if torch.cuda.is_available():
            task_id = to_cuda(task_id, gpu_id=None)
            data = to_cuda(data, gpu_id=None)
            target = to_cuda(target, gpu_id=None)

        update_target_dist = False

        self.optimizer.zero_grad()
        if (self.epoch + 1) % self.update_target_iter == 0:
                # self.network.set_update_target_dist(True)
                update_target_dist = True
        # update_target_dist = True
        self.print_to_log_file("epoch, update", self.epoch, update_target_dist)
        if self.fp16:
            with autocast():
                # output = self.network(data, task_id)
                tc_inds = copy.deepcopy(self.task_id_class_lst_mapping[int(task_id[0])])
                target_classes = target[0].unique().detach().cpu().numpy()
                output = self.network(data, task_id=task_id, gt_seg=target[0], target_classes=target_classes, update_target_dist=update_target_dist, for_loss=True)

                # extract tasks per gpu (should be big speed up)
                output, extractions = output
                # output, feature_extractions = output
                
                # NOTE sanity check output
                # output = torch.zeros_like(target[0], dtype=torch.float, device=data.device).repeat(1, self.feature_space_dim, 1, 1, 1)
                # for i, tc_ind in enumerate(tc_inds):
                #     mask = (target[0] == i)
                #     for j in range(self.feature_space_dim):  # Loop over the channels
                #         values = mus[tc_ind][j].repeat(mask.sum())
                #         output[:, j, :, :, :][mask.squeeze()] = values
                # univariate sanity check
                # output = torch.zeros_like(target[0], device=data.device)
                # mus, sigs = self.network.get_mean_var()
                # for i, tc_ind in enumerate(tc_inds):
                #     output[target[0] == i] = mus[tc_ind]

                # tc_inds = [self.task_id_class_lst_mapping[int(task_id[0])][int(c)] for c in target_classes]
                tc_inds = []
                i, j = 0, 0
                while i < len(extractions) and j < len(target_classes):
                    # re-arrange dimensions for loss
                    extractions[i] = extractions[i].permute(1, 0)
                    if extractions[i].size(-1) < 2:
                        _ = extractions.pop(i)
                    else:
                        c = target_classes[j]
                        tc_inds.append(self.task_id_class_lst_mapping[int(task_id[0])][int(c)])
                        i += 1
                    j += 1

                # assert len(output) > 1
                # for out in range(len(output)):
                #     output[out] = output[out][:, :self.task_class[int(task_id[0])]]

                # output, mus, sigs = output
                # if self.network.get_update_target_dist():
                if update_target_dist:
                    output, flat_tp_feats = output
                    input_sigs = torch.tensor([[0.001] * self.num_components])
                    self.mus, _ = self.dynamic_dist_network(flat_tp_feats, self.mus, input_sigs, tc_inds)
                    # self.mus, self.sigs = self.dynamic_dist_network(flat_tp_feats, self.mus, self.sigs, tc_inds)

                del data
                self.print_to_log_file(f"task_id: {task_id}")

                if self.loss_type == "kl":
                    l =  self.loss(extractions, self.mus, self.sigs, tc_inds, return_est_dists=self.return_est_dists, with_sep_loss=False)
                else:
                    l =  self.loss(output, self.mus, self.sigs, target[0], tc_inds, pred_dists=extractions, return_est_dists=self.return_est_dists, with_sep_loss=False)
                
                
                # Test the dice score:
                if self.return_est_dists: # False
                    l, est_dists = l
                # if self.return_est_dists:
                #     # l, est_dists = l
                #     with torch.no_grad():
                #         est_seg = self.network.module.segment_from_dists(output, est_dists, self.class_lst_to_std_mapping)
                #         # NOTE change permute to be (0, num_classes_in_gt_seg, 1, ..., num_classes_in_gt_seg-1 )
                #         perm_dims = tuple([0, len(output.shape)-1] + list(range(1, len(output.shape)-1)))
                #         # est_seg = torch.nn.functional.one_hot(est_seg, self.task_class[int(task_id[0])]).permute(0, 4, 1, 2, 3)
                #         # self.print_to_log_file(f"num classes in gt; {num_classes_in_gt}, {est_seg.unique()}")
                #         est_seg = torch.nn.functional.one_hot(est_seg, self.num_classes)
                #         est_seg = torch.permute(est_seg, perm_dims)
                #         # dc_score = self.dc_loss(est_seg.unsqueeze(1), target[0])
                #         dc_score = self.dc_loss(est_seg, target[0])
                #         # self.print_to_log_file(f"Dice Score: {-dc_score.item()}")
                #         if do_backprop:
                #             if self.with_wandb: wandb.log({"train_dc": -dc_score.item()})
                #             self.print_to_log_file(f"train_dc {-dc_score.item()}")
                #         else:
                #             if self.with_wandb: wandb.log({"val_dc": -dc_score.item()})
                #             self.print_to_log_file(f"val_dc {-dc_score.item()}")
                
                # Get sep loss
                if update_target_dist:
                    sep_loss = self.loss.get_sep_loss(self.mus)
                    if self.with_wandb:
                        wandb.log({"sep_loss": sep_loss})
                    self.print_to_log_file(f"sep_loss: {sep_loss}")
                    
                    l = l + sep_loss

                if self.with_wandb:
                    wandb.log({"loss": l.item()})
                self.print_to_log_file(f"loss: {l.item()}")
                # TODO make method of network; features = network output
                # Updated Queues
                # pick random element from extraction 
                self.dynamic_dist_network.update_queues(extractions, tc_inds)

            if do_backprop:
                # self.amp_grad_scaler.scale(l).backward(retain_graph=True) # NOTE
                self.amp_grad_scaler.scale(l).backward(retain_graph=False) # NOTE
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            pass

        if run_online_evaluation:
            # train GMMs
            if not self.network.module.gmm_fitted:
                # self.network.module.init_gmms(self.mus, self.sigs)
                self.network.module.train_gmms(self.dynamic_dist_network.feature_space_qs, self.mus, self.sigs)
            # self.network.construct_torch_gmm()
            # segment output
            output_segmentation = self.network.module.segment(output)
            std_output_segmentation = self.network.module.standardize_segmentation(output_segmentation, copy.copy(self.class_lst_to_std_mapping))
            # est_seg = torch.nn.functional.one_hot(std_output_segmentation, num_classes_in_gt).permute(0, 4, 1, 2, 3)
            perm_dims = tuple([0, len(output.shape)-1] + list(range(1, len(output.shape)-1)))
            # est_seg = torch.nn.functional.one_hot(std_output_segmentation, num_classes_in_gt)
            est_seg = torch.nn.functional.one_hot(std_output_segmentation, self.num_classes)
            est_seg = torch.permute(est_seg, perm_dims)
            # wrap in lists
            self.run_online_evaluation([est_seg], target)
            # tp, fp, fn, _ = get_tp_fp_fn_tn(std_output_segmentation, target[0])
            # tp_hard, fp_hard, fn_hard = self.get_hard_tp_fp_fn(est_seg, target[0])
            # self.run_online_evaluation(tp_hard, fp_hard, fn_hard)

        del target

        if (self.epoch + 1) % self.update_target_iter == 0:
                # self.network.set_update_target_dist(False)
                update_target_dist = False # redundant 

        return l.detach().cpu().numpy()


    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D_UniSeg(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r', task=self.task)
            # import pdb
            # pdb.set_trace()
            dl_val = DataLoader3D_UniSeg(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r', task=self.task)
        else:
            pass
        return dl_tr, dl_val

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """
        if isinstance(self.network, DDP):
            net = self.network.module
        else:
            net = self.network
        ds = net.do_ds
        net.do_ds = False

        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        export_pool = Pool(8)
        results = []

        for k in tqdm(self.dataset_val.keys()):
            properties = load_pickle(self.dataset[k]['properties_file'])
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                    (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                data = np.load(self.dataset[k]['data_file'])['data']


                task_id = np.zeros((1,))
                task_id[0] = self.task[fname[:4]]
                print("fname", fname, "covert id:", task_id[0])

                print(k, data.shape)
                data[-1][data[-1] == -1] = 0

                # NOTE
                # self.network.conv_op = self.network.feature_extractor.conv_op

                softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(data[:-1], task_id,
                                                                                     do_mirroring=do_mirroring,
                                                                                     mirror_axes=mirror_axes,
                                                                                     use_sliding_window=use_sliding_window,
                                                                                     step_size=step_size,
                                                                                     use_gaussian=use_gaussian,
                                                                                     all_in_gpu=all_in_gpu,
                                                                                     mixed_precision=self.fp16)[1]

                softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                softmax_pred = softmax_pred[:self.task_class[int(task_id)]]



                if save_softmax:
                    softmax_fname = join(output_folder, fname + ".npz")
                else:
                    softmax_fname = None

                """There is a problem with python process communication that prevents us from communicating objects
                larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
                enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                filename or np.ndarray and will handle this automatically"""
                if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                    np.save(join(output_folder, fname + ".npy"), softmax_pred)
                    softmax_pred = join(output_folder, fname + ".npy")


                # import pdb
                # pdb.set_trace()
                results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                         ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                           properties, interpolation_order, self.regions_class_order,
                                                           None, None,
                                                           softmax_fname, None, force_separate_z,
                                                           interpolation_order_z),
                                                          )
                                                         )
                               )

            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # # evaluate raw predictions
        # self.print_to_log_file("evaluation of raw predictions")
        # task = self.dataset_directory.split("/")[-1]
        # job_name = self.experiment_name
        # _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
        #                      json_output_file=join(output_folder, "summary.json"),
        #                      json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
        #                      json_author="Fabian",
        #                      json_task=task, num_threads=default_num_threads)
        #
        # if run_postprocessing_on_folds:
        #     # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
        #     # except the largest connected component for each class. To see if this improves results, we do this for all
        #     # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
        #     # have this applied during inference as well
        #     self.print_to_log_file("determining postprocessing")
        #     determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
        #                              final_subf_name=validation_folder_name + "_postprocessed", debug=debug)
        #     # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
        #     # They are always in that folder, even if no postprocessing as applied!

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError:
                    print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                    traceback.print_exc()
                    attempts += 1
                    sleep(1)
            if not success:
                raise OSError(f"Something went wrong while copying nifti files to {gt_nifti_folder}. See above for the trace.")

        self.network.train(current_mode)

        self.network.do_ds = ds

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, task_id, do_mirroring: bool = True,
                                                         mirror_axes= None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True):
        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        valid = list((SegmentationNetwork, nn.DataParallel, DDP))
        assert isinstance(self.network, tuple(valid))
        if isinstance(self.network, DDP):
            net = self.network.module
        else:
            net = self.network
        ds = net.do_ds
        net.do_ds = False
        ret = net.predict_3D(data, task_id, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                             use_sliding_window=use_sliding_window, step_size=step_size,
                             patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                             use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                             pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                             mixed_precision=mixed_precision)
        net.do_ds = ds
        return ret

    def get_hard_tp_fp_fn(self, output, target):
        with torch.no_grad():
            num_classes = output.shape[1]
            # output_softmax = softmax_helper(output)
            output_softmax = output
            output_seg = output_softmax.argmax(1)
            target = target[:, 0]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

            tp_hard = tp_hard.sum(0, keepdim=False)[None]
            fp_hard = fp_hard.sum(0, keepdim=False)[None]
            fn_hard = fn_hard.sum(0, keepdim=False)[None]

            return tp_hard, fp_hard, fn_hard


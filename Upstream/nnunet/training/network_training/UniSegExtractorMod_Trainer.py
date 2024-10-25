# -*- coding:utf-8 _*-
# @author: ywye
# @contact: ywye@mail.nwpu.edu.cn
# @version: 0.1.0
# @file: UniSeg with nnunet version
# @time: 2022/11/29
import numpy as np
import random
import itertools
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.cuda.amp import autocast
# from nnunet.training.network_training.nnUNetTrainerV2_DDP import nnUNetTrainerV2_DDP
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
import torch
import torch.distributed as dist
from torch.optim import lr_scheduler
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn, distributed
import SimpleITK as sitk
import shutil
from nnunet.network_architecture.Prompt_generic_UNet_DP import UniSeg_model, UniSegExtractor_DP, DynamicDistributionModel_DP, TAPFeatureExtractor_DP
from nnunet.network_architecture.TAP import TAP
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
from nnunet.training.loss_functions.deep_supervision import MixedDSLoss
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss, SoftDiceLoss, get_tp_fp_fn_tn
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunet.utilities.sep import component_wise_kl_div, measure_change, is_positive_definite, pen_domain, get_non_uniform_dynamic_sep_loss
from nnunet.utilities.gmm import get_choleskys
from nnunet.utilities.preloaded_targets import get_targets
from nnunet.utilities.wandb_util import wandb_log_outputs, test_img_log, test_img_log_color
from nnunet.utilities.tensor_utilities import sum_tensor
import copy 
import wandb

class UniSegExtractorMod_Trainer(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, 
                 feature_space_dim=32, gmm_comps=5, loss_type="gnlll", update_iter=1, queue_size=5000, max_num_epochs=1000, 
                 batch_size=2, num_gpus=1, single_task=False, ood_detection_mode=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = max_num_epochs
        # self.task = {"live":0, "kidn":1, "hepa":2, "panc":3, "colo":4, "lung":5, "sple":6, "sub-":7, "pros":8, "BraT":9}
        # self.task_class = {0: 3, 1: 3, 2: 3, 3: 3, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 4}
        # self.task_id_class_lst_mapping = {
        #     0: [0, 1, 2], 
        #     1: [0, 1, 2], 
        #     2: [0, 1, 2],
        #     3: [0, 1, 2],
        #     4: [0, 1], 
        #     5: [0, 1],
        #     6: [0, 1],
        #     7: [0, 1],
        #     8: [0, 1],
        #     9: [0, 1, 2, 3], 
        # }
        # self.task = {"live":0, "kidn":1, "hepa":2, "panc":3, "colo":4, "lung":5, "sple":6, "sub-":7, "pros":8}
        # self.task_class = {0: 3, 1: 3, 2: 3, 3: 3, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2}
        # self.task_id_class_lst_mapping = {
        #     0: [0, 1, 2], 
        #     1: [0, 1, 2], 
        #     2: [0, 1, 2], 
        #     3: [0, 1, 2], 
        #     4: [0, 1], 
        #     5: [0, 1], 
        #     6: [0, 1], 
        #     7: [0, 1], 
        #     8: [0, 1], 
        # }
        # self.task = {"pros":0, "lung":1, "sple":2, "live":3}
        # self.task_class = {0: 2, 1: 2, 2: 2, 3: 3}
        # self.task_id_class_lst_mapping = {
        #     0: [0, 1], 
        #     1: [0, 2], 
        #     2: [0, 3], 
        #     3: [0, 4, 5], 
        # }
        self.task = {"pros":0, "lung":1, "sple":2, "live":3}
        self.task_class = {0: 2, 1: 2, 2: 2, 3: 3}
        self.task_id_class_lst_mapping = {
            0: [0, 1], 
            1: [0, 1], 
            2: [0, 1], 
            3: [0, 1, 2], 
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
        self.total_task_num = len(self.task.keys()) if not single_task else 1 # NOTE
        self.batch_size = batch_size
        self.num_batches_per_epoch = (50 * self.total_task_num) #// (num_gpus * (self.batch_size // 2)) #int((50 // num_gpus) * self.total_task_num)
        # self.num_batches_per_epoch = (3 * self.total_task_num) #// (num_gpus * self.batch_size) #int((50 // num_gpus) * self.total_task_num)
        # self.num_val_batches_per_epoch = self.num_batches_per_epoch // self.total_task_num#// 5
        print("num batches per epoch:", self.num_batches_per_epoch)
        # print("num batches per val epoch:", self.num_val_batches_per_epoch)
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
        self.gmm_comps = gmm_comps
        self.queue_size = queue_size
        self.num_components = len(self.class_lst_to_std_mapping.keys()) # NOTE num classes
        self.num_classes = self.num_components
        self.update_target_dist = False
        self.return_est_dists = True
        self.loss_type = loss_type
        self.ood_detection_mode = ood_detection_mode

        self.max_gmm_comps = gmm_comps

        self.recalc_dist = False

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
        self.base_num_features = 32 #int(self.feature_space_dim // 2)
        # self.uniseg_network = UniSeg_model(self.patch_size, self.total_task_num, [1, 2, 4], self.base_num_features, self.num_classes,
        #                             len(self.net_num_pool_op_kernel_sizes),
        #                             self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
        #                             dropout_op_kwargs,
        #                             net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
        #                             self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        # uniseg_args = (self.patch_size, self.total_task_num, [1, 2, 4], self.base_num_features, self.num_classes,
        #                             len(self.net_num_pool_op_kernel_sizes),
        #                             self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
        #                             dropout_op_kwargs,
        #                             net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
        #                             self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        uniseg_args = (self.patch_size, self.total_task_num, [1, 2, 4], self.base_num_features, self.num_components,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        uniseg_kwargs = {}

        # self.network.inference_apply_nonlin = softmax_helper
        # num_components = len(self.class_lst_to_std_mapping.keys())
        self.network = TAPFeatureExtractor_DP(self.feature_space_dim, 
                                            self.gmm_comps,
                                            self.num_components,  
                                            copy.deepcopy(self.class_lst_to_std_mapping), 
                                            copy.deepcopy(self.task_id_class_lst_mapping),
                                            *uniseg_args, 
                                            with_wandb=self.with_wandb,
                                            ood_detection_mode=self.ood_detection_mode,
                                        #   **uniseg_kwargs
                                        )
        # self.tap = TAP(self.feature_space_dim, self.num_components, 0.995, queue_size=5000, gmm_comps = self.gmm_comps)
        # self.tap = TAP(self.feature_space_dim, self.num_components, 0.995, queue_size=5000, gmm_comps = 10)
        # self.tap = TAP(self.feature_space_dim, self.num_components, self.total_task_num, 0.995, queue_size=5000, gmm_comps = self.max_gmm_comps)
        # self.task_taps = [
        #     TAP(self.feature_space_dim, self.num_components, self.total_task_num, 0.995, queue_size=5000, gmm_comps = self.max_gmm_comps)
        #     for _ in range(self.total_task_num)
        # ]
        
        self.network.inference_apply_nonlin = softmax_helper # lambda x: x
        if torch.cuda.is_available():
            self.network = self.network.cuda()
            # self.tap = self.tap.cuda()
            # for t in range(self.total_task_num):
            #     self.task_taps[t] = self.task_taps[t].cuda()


        # self.network.min_dist = self.min_dist

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                        amsgrad=True)
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2,
                                                        patience=self.lr_scheduler_patience,
                                                        verbose=True, threshold=self.lr_scheduler_eps,
                                                        threshold_mode="abs")
        # self.tap_optimizer = torch.optim.Adam(self.tap.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                                 amsgrad=True)
        # self.tap_optimizer = torch.optim.Adam(itertools.chain(*(tp.parameters() for tp in self.task_taps)), self.initial_lr, weight_decay=self.weight_decay,
        #                                 amsgrad=True)
        # self.tap_lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.tap_optimizer, mode='min', factor=0.2,
        #                                                 patience=self.lr_scheduler_patience,
        #                                                 verbose=True, threshold=self.lr_scheduler_eps,
        #                                                 threshold_mode="abs")
        self.tap_tasks_optimizer = [None for _ in range(self.total_task_num)]
        self.tap_tasks_lr_scheduler = [None for _ in range(self.total_task_num)]
        for t in range(self.total_task_num):
            # self.tap_tasks_optimizer[t] = torch.optim.Adam(self.task_taps[t].parameters(), self.initial_lr, weight_decay=self.weight_decay,
            #                                 amsgrad=True)
            self.tap_tasks_optimizer[t] = torch.optim.Adam(self.network.task_taps[t].parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                            amsgrad=True)
            self.tap_tasks_lr_scheduler[t] = lr_scheduler.ReduceLROnPlateau(self.tap_tasks_optimizer[t], mode='min', factor=0.2,
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
            # self.dc_loss = SoftDiceLoss(batch_dice=True, smooth=1e-5, do_bg=True, apply_nonlin=None, do_one_hot=False)
            
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    unpack_dataset(self.folder_with_preprocessed_data)
                    # if self.local_rank == 0:
                    #     print("unpacking dataset")
                    #     print("done")
                    # distributed.barrier()
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
            # torch.cuda.manual_seed_all(42)
            # max_var = 0.001
            # self.max_var = 0.001
            # delta=1-(1e-03)
            # # NOTE just use linspace here
            # # min_dist = 100. #self.distance_bounds(k=num_components, delta=delta, max_var=max_var*100)

            # # Multivariate
            # unscaled_means = torch.stack([min_dist * torch.randint(0, self.num_components, size=(self.feature_space_dim,)) for _ in range(self.num_components)])
            # max_norm = torch.max(torch.norm(unscaled_means, dim=1, p=2))
            # self.mus = unscaled_means / max_norm
            # # self.mus = torch.linspace(0, 1, num_components, requires_grad=False)[:, None]
            # min_dist = torch.cdist(self.mus, self.mus, p=2).fill_diagonal_(torch.inf).min().item()
            
            # # self.vars = torch.full_like(self.mus, fill_value=max_var, requires_grad=False)
            # self.sigs = torch.stack([
            #     self.max_var * torch.eye(self.feature_space_dim, dtype=self.mus.dtype) for _ in range(self.num_components)
            # ])
            # # self.sigs = (unscaled_vars - 1e-05) / (1e-03 - 1e-05)
            # self.input_vars = self.max_var * torch.ones((1, self.num_components), dtype=self.mus.dtype)

            # self.print_to_log_file(f"initial means: {self.mus}")
            # self.print_to_log_file(f"initial vars: {self.sigs}")

            # min_dist = 2 * np.sqrt(self.feature_space_dim * self.max_var)# . #self.distance_bounds(k=num_components, delta=delta, max_var=max_var*100)
            # min_dist = 1 * np.sqrt(self.feature_space_dim * self.max_var)# . #self.distance_bounds(k=num_components, delta=delta, max_var=max_var*100)
            # self.mus = self.poisson_disk_sample(self.num_components, self.feature_space_dim, min_dist)
            # computed_min_dist = torch.cdist(self.mus, self.mus, p=2).fill_diagonal_(torch.inf).min().item()
            # self.mus = self.mus.to(dtype=torch.float16)

            """
            vars = np.ones(3) / 10
            c = 0.1
            self.csep = c
            domain = [0, 1]
            vars = np.ones(3) / 10
            c = 0.1
            self.csep = c
            domain = [0, 1]
            self.domain = domain
            """

            # Current best
            # vars = [1/30, 1/30, 1/30]
            vars = [1/30, 1/30, 1/30, 1/30]
            csep = 1.5
            self.csep = csep
            domain = [-1,1]
            self.domain = domain

            # self.mus, self.sigs, self.weights, self.min_dists = self.init_task_distribution(vars, csep, domain=domain)
            # self.min_dist = max(self.min_dists)
            
            # NOTE Old, can remove
            # print(f"{vars}, {csep}, {self.feature_space_dim}")
            # min_dists = [csep * np.sqrt(self.feature_space_dim * v) for i, v in enumerate(vars)]
            # self.min_dists = min_dists
            # print(f"{min_dists}")
            # self.min_dist = max(min_dists)
            # max_trys = 15000
            # for t in range(max_trys):
            #     if t % 100 == 0: print(t)
            #     try:
            #         # centers = self.init_centers(self.num_components, self.feature_space_dim, min_dists, sort=False)
            #         centers = self.init_centers(self.num_components, self.feature_space_dim, min_dists, domain=domain, sort=False)
            #         break
            #     except ValueError:
            #         pass

            # self.mus, self.sigs = self.init_gmm_centers(centers, self.feature_space_dim, min_dists, csep)
            # # self.mus, self.sigs = self.init_from_fixed()
            # self.weights = self.init_uniform_mixture_weights()
            
            # init 
            # self.choleskys = get_choleskys(self.sigs)
            
            # self.mus = self.mus.to(dtype=torch.float16)
            # self.mus, self.min_dist, self.max_var = self.optimal_sep(self.num_components, self.feature_space_dim)
            # self.mus = self.mus.to(dtype=torch.float16)

            # self.sigs = torch.stack([
            #     self.max_var * torch.eye(self.feature_space_dim, dtype=torch.float32) for _ in range(self.num_components)
            # ])
            # self.input_vars = self.max_var * torch.ones((1, self.num_components), dtype=torch.float16)
            # self.sigs = torch.stack([
            #     vars[i] * torch.eye(self.feature_space_dim, dtype=torch.float32) for i in range(self.num_components)
            # ])
            # self.input_vars = torch.tensor(vars).to(dtype=torch.float16)

            print(self.total_task_num)
            tasks_vars = [vars for _ in range(self.total_task_num)]
            tasks_cseps = [csep for _ in range(self.total_task_num)]

            self.tasks_mus, self.tasks_sigs, self.tasks_weights, self.tasks_min_dists, self.tasks_min_dist = self.init_all_distributions(tasks_vars, tasks_cseps, domain=domain, **{"center_method": "optim"})
            # self.tasks_mus, self.tasks_sigs, self.tasks_weights, self.tasks_min_dists, self.tasks_min_dist = self.init_all_distributions(tasks_vars, tasks_cseps, domain=domain, **{"center_method": "partition"})
            # self.tasks_mus, self.tasks_sigs, self.tasks_weights, self.tasks_min_dists, self.tasks_min_dist = get_targets()
            self.min_dist = max([self.tasks_min_dist[t] for t in range(self.total_task_num)])
            
            ##### END ####
            self.with_wandb = wandb.run is not None
            self.initialize_network()

            #### Loss Function Inits ####
            # Deep supervision loss
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # self.ds_loss = DynamicDistMatchingLoss(self.min_dist, loss_type=self.loss_type, with_wandb=self.with_wandb) # NOTE
            # self.ds_loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
            
            # Dice loss
            # dc_loss = SoftDiceLoss(apply_nonlin=lambda x: x, **{'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}) # NOTE
            # dc_loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
            # dc_loss.dc.apply_nonlin = lambda x: x
            # self.dc_loss = dc_loss
            
            # Main - dist matching, cross entropy loss
            # self.main_loss = DynamicDistMatchingLoss(self.min_dist, loss_type=self.loss_type, with_wandb=self.with_wandb) # NOTE
            # self.main_loss = DynamicDistMatchingLoss(self.min_dist, loss_type=self.loss_type, do_bg=False, with_wandb=self.with_wandb) # NOTE
            
            # self.loss = MixedDSLoss(main_loss=self.main_loss, ds_loss=self.ds_loss, weight_factors=self.ds_loss_weights)
            # self.loss = MultipleOutputLoss2(DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {}), self.ds_loss_weights)
            # self.loss = MixedDSLoss(main_loss=self.main_loss, ds_loss=self.ds_loss, dice_loss=dc_loss, weight_factors=self.ds_loss_weights)
            self.sanity_loss = MultipleOutputLoss2(loss=DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {}), weight_factors=self.ds_loss_weights)
            
            self.initialize_optimizer_and_scheduler()
            # self.network = DDP(self.network, device_ids=[self.local_rank])
            # self.network.set_feature_space_distribution_parameters(self.mus, self.sigs)
            
            # self.network.construct_feature_space_gmm(self.mus, self.sigs)

            for t in range(self.total_task_num):
                self.network.construct_feature_space_gmm(self.tasks_mus[t], self.tasks_sigs[t], task=t)
            
            # self.dynamic_dist_network = DDP(self.dynamic_dist_network, device_ids=[self.local_rank])
            self.update_target_dist = False

            # self.main_loss.dist_weights = self.weights
            # self.main_loss.dist_weights = self.tasks_weights[0]

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

        #     # check if device of mus/sigs
        #     if not self.mus[0][0].is_cuda:
        #         for t in range(self.num_components):
        #             # self.print_to_log_file(f"prev size {t}: {[m.size() for m in self.mus[t]]}")
        #             self.mus[t] = self.mus[t].cuda()
        #             self.sigs[t] = self.sigs[t].cuda()
        #             self.weights[t] = self.weights[t].cuda()
            
        self.log_img_wandb = False
        print(f"recalc: {self.recalc_dist}")
        self.optimizer.zero_grad()
        # self.recalc_dist = False
        # if self.epoch > 0 and self.recalc_dist:# and (self.epoch + 1) % self.update_target_iter == 0:
        if self.epoch > 0 and self.recalc_dist:# and (self.epoch + 1) % self.update_target_iter == 0:
            self.recalc_targets()
            # if self.n_iter_not_improved >= 3:
            #     self.recalc_targets()

            #     self.n_iter_not_improved = 0

            #     self.recalced = True
            self.log_img_wandb = True

        update_target_dist = self.update_target_dist
        # self.print_to_log_file("epoch, update", self.epoch, update_target_dist)
        if self.fp16:
            with autocast():
                # output = self.network(data, task_id)
                # tc_inds = copy.deepcopy(self.task_id_class_lst_mapping[int(task_id[0])])
                # target_classes = target[0].unique().detach().cpu().numpy()
                self.print_to_log_file(f"\ntask_id: {task_id}")
                lst_target_classes = [target[i].unique().detach().cpu().numpy() for i in range(len(target))]
                # lst_target_classes = [np.delete(target[i].unique().detach().cpu().numpy(), 0) for i in range(len(target))]
                # output, gt_extractions, tc_inds = self.network(data, task_id=task_id, gt_seg=target, target_classes=lst_target_classes, update_target_dist=update_target_dist, for_loss=True)
                # output, gt_extractions, tc_inds, output_probs = self.network(data, task_id=task_id, gt_seg=target, target_classes=lst_target_classes, update_target_dist=update_target_dist, for_loss=True)
                output, gt_extractions, tc_inds, output_probs, output_probs_softmax = self.network(data, task_id=task_id, gt_seg=target, target_classes=lst_target_classes, update_target_dist=update_target_dist, for_loss=True)
                # for out in range(len(output)):
                #     output[out] = output[out][:, :self.task_class[int(task_id[0])]]
                # for out in range(len(output_probs_softmax)):
                #     output_probs_softmax[out] = output_probs_softmax[out][:, :self.task_class[int(task_id[0])]]
                for out in range(len(output_probs)):
                    output_probs[out] = output_probs[out][:, :self.task_class[int(task_id[0])]]
                
                # l = self.loss(output, target) lst_extractions[0], self.mus, self.sigs, lst_tc_inds[0 means, covs, indices, handle_nan=False, return_est_dists=False, with_sep_loss=True)
                # l = self.loss(output, target, features, self.mus, self.sigs, tc_inds, return_est_dists=False, handle_nan=False, with_sep_loss=False)
                # l = self.loss.forward_ce_dc(output, target, features, self.mus, self.sigs, tc_inds, return_est_dists=False, handle_nan=False, with_sep_loss=False)
                
                tidx = int(task_id[0])
                # self.loss.main_loss.dist_weights = self.tasks_weights[tidx]
                # l = self.loss.forward_dist_dc_ds(output, target, features, tc_inds, self.mus, self.sigs, return_est_dists=False, handle_nan=False, with_sep_loss=False)
                # l = self.loss.forward_dist_dc_ds(output, target, gt_extractions, tc_inds, self.tasks_mus[tidx], self.tasks_sigs[tidx], return_est_dists=False, handle_nan=False, with_sep_loss=False)
                # l = self.loss.forward_ce_dc_ds(output, target, output_probs)

                # wandb_log_outputs(data[0], features[0], output[0][0], None, target[0][0])
                if self.with_wandb and tidx in self.to_log:
                    # test_img_log(data[0], target[0], output_probs[0], tidx, self.task_id_class_lst_mapping[tidx])
                    # test_img_log_color(data[0], target[0], output_probs[0], tidx, self.task_id_class_lst_mapping[tidx])
                    test_img_log_color(data, target[0], output_probs[0], tidx, self.task_class[tidx])
                    # self.print_to_log_file(f"prior to rmv: {self.to_log}")
                    self.to_log = self.to_log[self.to_log != tidx]
                    # self.print_to_log_file(f"removed {tidx} to rmv: {self.to_log}")

                del data
                
                
                # Sanity check
                # l = self.sanity_loss(output_probs_softmax, target)
                l = self.sanity_loss(output_probs, target)
                
                # Test the dice score:
                # if self.return_est_dists and self.epoch > 0:
                #     # l, est_dists = l
                # with torch.no_grad():
                #     # dc_score = self.dc_loss(est_seg.unsqueeze(1), target[0])
                #     # dc_score = self.dc_loss(output[0], target[0])
                #     dc_score = self.sanity_loss.loss.dc(output_probs_softmax[0], target[0])
                #     # self.print_to_log_file(f"Dice Score: {-dc_score.item()}")
                #     if do_backprop:
                #         if self.with_wandb: wandb.log({"train_dc": -dc_score.item()})
                #         self.print_to_log_file(f"train_dc {-dc_score.item()}")
                #     else:
                #         if self.with_wandb: wandb.log({"val_dc": -dc_score.item()})
                #         self.print_to_log_file(f"val_dc {-dc_score.item()}")
                
                
                if self.with_wandb:
                    wandb.log({"loss": l.item()})
                self.print_to_log_file(f"loss: {l.item()}")


            if do_backprop:
                # self.amp_grad_scaler.scale(l).backward(retain_graph=False) # NOTE
                self.amp_grad_scaler.scale(l).backward(retain_graph=True) # NOTE
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
                # TAP
                if self.epoch > 0:
                    # self.tap_amp_grad_scaler.scale(l).backward(retain_graph=True)
                    self.tap_tasks_amp_grad_scaler[tidx].scale(l).backward(retain_graph=True)

            # else:
            #     # self.tap.update_val_queues(gt_extractions, tc_inds)
            #     # self.tap.update_val_queues(gt_extractions[0], tc_inds[0], update_size=25) # NOTE not used
            #     # self.task_taps[tidx].update_val_queues(gt_extractions[0], tc_inds[0], update_size=25)
            #     self.network.task_taps[tidx].update_val_queues(gt_extractions[0], tc_inds[0], update_size=20)
            

        else:
            pass

        if run_online_evaluation:
            # self.run_online_evaluation(output, target)
            # self.run_online_evaluation(output_probs_softmax, target)
            self.run_online_evaluation(output_probs, target)

        del target

        return l.detach().cpu().numpy()

    def recalc_targets(self):
        task_class_optimal_ns = []
        
        for task_idx in range(self.total_task_num):
            optimal_ns = self.recalc_task_targets(task_idx)
            task_class_optimal_ns.append(optimal_ns)

        # NOTE revise
        # self.loss.main_loss.dist_weights = self.weights

        # if self.n_iter_not_improved >= 0:
        if True:
        # if False:
            # self.allocate_targets(component_optimal_ns)
            for tidx in range(self.total_task_num):
                # self.allocate_targets([10 for _ in range(self.num_components)], tidx)
                self.allocate_targets([self.max_gmm_comps for _ in range(self.task_class[tidx])], tidx)

            self.n_iter_not_improved = 0

            self.recalced = True
        # else:
        #     # Prune
        #     for task_idx in range(self.total_task_num):
        #         component_optimal_ns = task_class_optimal_ns[task_idx]
        #         # for c in range(self.num_components):
        #         for c in range(self.task_class[task_idx]):
        #             optimal_n = self.gmm_comps #component_optimal_ns[c]
        #             self.tasks_mus[task_idx][c] = self.tasks_mus[task_idx][c][:optimal_n]
        #             self.tasks_sigs[task_idx][c] = self.tasks_sigs[task_idx][c][:optimal_n]
        #             self.tasks_weights[task_idx][c] = self.tasks_weights[task_idx][c][:optimal_n]
                    
            # for c in range(self.num_components):
            #     optimal_n = component_optimal_ns[c]
            #     self.mus[c] = self.mus[c][:optimal_n]
            #     self.sigs[c] = self.sigs[c][:optimal_n]
                
            # self.network.set_feature_space_distribution_parameters(self.mus, self.sigs)
            # self.network.weights = self.weights

        for tidx in range(self.total_task_num):
            self.network.set_feature_space_distribution_parameters(self.tasks_mus[tidx], self.tasks_sigs[tidx], self.tasks_weights[tidx], task=tidx)
            # self.network.weights = self.weights

        self.recalc_dist = False

        # self.network.construct_feature_space_gmm_implicit()
        for tidx in range(self.total_task_num):
            self.network.construct_task_feature_space_gmm_implicit(tidx)

    def recalc_task_targets(self, task_idx):
        class_optimal_ns = []
        # for class_idx in range(self.num_components):
        for class_idx in range(self.task_class[task_idx]):
            self.print_to_log_file(f"recalc for task {task_idx}, class {class_idx} / {self.task_class[task_idx]}")
            # task_class_queue = self.network.tap.feature_space_qs[class_idx] 
            # task_class_val_queue = self.tap.val_feature_space_qs[class_idx] 
            # task_class_val_queue = self.task_taps[task_idx].val_feature_space_qs[class_idx] 
            task_class_queue = self.network.task_taps[task_idx].feature_space_qs[class_idx] 
            # task_class_val_queue = self.network.task_taps[task_idx].val_feature_space_qs[class_idx] 
            # task_class_val_queue = self.network.task_taps[-1].val_feature_space_qs[class_idx] 
            # if len(task_class_queue) < 10:
            if len(task_class_queue) < 5:
                self.print_to_log_file(f"Not enough samples: {len(task_class_queue)}, skipping...")
                continue
            task_class_queue = torch.vstack(task_class_queue).detach().cpu().numpy()
            # task_class_queue = torch.vstack(task_class_queue + task_class_val_queue).detach().cpu().numpy()
            # task_class_queue = torch.vstack(task_class_val_queue).detach().cpu().numpy()
            # task_class_val_queue = torch.vstack(task_class_val_queue).detach().cpu().numpy() if len(task_class_val_queue) > 10 else task_class_queue
            # self.network.gaussian_mixtures[t].fit(task_queue)
            # momentum = 0.1 * (1 - (self.epoch / self.max_num_epochs))
            momentum = 0.1 #* (1 - (self.epoch / self.max_num_epochs))
            # momentum = 0.1 * (self.epoch / self.max_num_epochs)
            # mu_hat_t = torch.from_numpy(self.network.gaussian_mixtures[t].means_).cuda()
            # sig_hat_t = torch.from_numpy(self.network.gaussian_mixtures[t].covariances_).cuda()
            # curr_num_comps = self.mus[class_idx].shape[0]
            # curr_num_comps = self.tasks_mus[task_idx][class_idx].shape[0]
            # min_comps, max_comps = max(1, curr_num_comps-1), min(self.max_gmm_comps, curr_num_comps+1)
            # best_gmm, optimal_n, bics = self.network.find_optimal_components(val_queue, min_components = min_comps, max_components=max_comps)
            optimal_n = self.gmm_comps#10
            class_optimal_ns.append(optimal_n)
            if self.with_wandb:
                # wandb.log({f'{class_idx}_opt_n': optimal_n})
                wandb.log({f'{task_idx}_{class_idx}_opt_n': optimal_n})
            # best_gmm, _, _  = self.network.find_optimal_components(task_queue, min_components = optimal_n, max_components=optimal_n)
            # best_gmm, _, _  = self.network.fit_gmms(task_class_queue)
            # best_gmm, _, _  = self.network.fit_task_gmms(task_class_queue, task_idx)
            best_gmm, _, _  = self.network.fit_task_gmms(task_class_queue, task_idx, class_idx)
            mu_hat_c = torch.from_numpy(best_gmm.means_).cuda()
            if best_gmm.covariance_type == "full":
                sig_hat_c = torch.from_numpy(best_gmm.covariances_).cuda()
            else:
                sig_hat_c = torch.stack([torch.diag(d) for d in torch.from_numpy(best_gmm.covariances_).cuda()])
            weights_hat_c = torch.from_numpy(best_gmm.weights_).cuda()
            new_mus, new_sigs = [], []
            new_weights = []
                # also record mixture weights
                # self.weights[t] = torch.from_numpy(best_gmm.weights_).cuda()
                # set mean, cov
                # self.mus[t] = mu_hat_t #(1 - momentum) * self.mus[t] + (momentum * mu_hat_t)
                # self.sigs[t] = sig_hat_t.detach()#(1 - momentum) * self.sigs[t] + (momentum * sig_hat_t)
            # prev_comp_n = len(self.mus[class_idx])
            prev_comp_n = len(self.tasks_mus[class_idx])
                # optimal_n = prev_comp_n
                # component_optimal_ns.append(prev_comp_n)
                # self.print_to_log_file(f"prev size: {[m.size() for m in self.mus[t]]}")
                # for comp_idx in range(self.network.gaussian_mixtures[t].n_components):
                # for comp_idx in range(optimal_n):
            for comp_idx in range(self.max_gmm_comps):
                if comp_idx >= optimal_n:
                    new_mus.append(torch.zeros(self.feature_space_dim).cuda())
                    new_sigs.append(torch.eye(self.feature_space_dim).cuda()) # NOTE
                    new_weights.append(torch.ones(1).cuda() / self.max_gmm_comps)
                elif comp_idx >= prev_comp_n:
                    # new_mus.append((1 - momentum) * torch.mean(self.mus[class_idx], 0) + (momentum * mu_hat_c[comp_idx]))
                    new_mus.append((1 - momentum) * torch.mean(self.tasks_mus[task_idx][class_idx], 0) + (momentum * mu_hat_c[comp_idx]))
                        # new_sigs.append((1 - momentum) * torch.mean(self.sigs[t], 0) + (momentum * sig_hat_t[comp_idx]))
                    new_sigs.append(sig_hat_c[comp_idx])
                    # new_weights.append((1 - momentum) * torch.mean(self.weights[class_idx], 0) + (momentum * weights_hat_c[comp_idx]))
                    new_weights.append((1 - momentum) * torch.mean(self.tasks_weights[task_idx][class_idx], 0) + (momentum * weights_hat_c[comp_idx]))
                else:
                    # new_mus.append((1 - momentum) * self.mus[class_idx][comp_idx] + (momentum * mu_hat_c[comp_idx]))
                    new_mus.append((1 - momentum) * self.tasks_mus[task_idx][class_idx][comp_idx] + (momentum * mu_hat_c[comp_idx]))
                    new_sigs.append(sig_hat_c[comp_idx])
                        # new_sigs.append((1 - momentum) * self.sigs[t][comp_idx] + (momentum * sig_hat_t[comp_idx]))
                    # new_weights.append((1 - momentum) * self.weights[t][comp_idx] + (momentum * weights_hat_t[comp_idx]))
                    momentum_updated_weight_comp = (1 - momentum) * self.tasks_weights[task_idx][class_idx][comp_idx] + (momentum * weights_hat_c[comp_idx] )
                    new_weights.append(momentum_updated_weight_comp)
                    # new_weights.append(weights_hat_c[comp_idx])

                if self.with_wandb:
                    wandb.log({
                        f"em_{task_idx}_mu_{class_idx}_{comp_idx}": new_mus[-1], 
                        f"em_{task_idx}_cov_{class_idx}_{comp_idx}": new_sigs[-1], 
                        f"em_{task_idx}_weight_{class_idx}_{comp_idx}": new_weights[-1]
                    })
                    self.print_to_log_file(f"em_{task_idx}_weight_{class_idx}_{comp_idx}: {new_weights[-1]}")

            # self.mus[class_idx] = torch.vstack(new_mus).detach()
            # self.sigs[class_idx] = torch.stack(new_sigs).detach()
            # self.weights[class_idx] = torch.vstack(new_weights).detach().permute(1,0)
            self.tasks_mus[task_idx][class_idx] = torch.vstack(new_mus).detach().clone()
            self.tasks_sigs[task_idx][class_idx] = torch.stack(new_sigs).detach().clone()
            self.tasks_weights[task_idx][class_idx] = torch.stack(new_weights).detach().clone()#.permute(1,0)

        return class_optimal_ns

    def allocate_targets(self, component_optimal_ns, task_idx):
        # input_covs = []
        # for cov_comp_set in self.sigs:
        #     for cov in cov_comp_set:
        #         # input_covs.append(torch.max(torch.real(torch.linalg.eigvals(cov))))
        #         input_covs.append(torch.diagonal(cov)) # NOTE

        # input_covs = torch.stack(input_covs)

        input_covs = []
        # for cidx in range(self.num_components):
        for cidx in range(self.task_class[task_idx]):
            # comp_full_covs = self.sigs[cidx]
            comp_full_covs = self.tasks_sigs[task_idx][cidx]
            e_covs = []
            for eidx in range(self.max_gmm_comps):
                # if self.network.task_taps[task_idx][eidx].covariance_type == "full":
                if self.network.task_var_gmms[task_idx][cidx].covariance_type == "full":
                    cov_eigvals = torch.real(torch.linalg.eigvals(comp_full_covs[eidx])) # sanity, eigvals should be real already
                    # cov_eigvals = torch.diagonal(comp_full_covs[eidx])
                else:
                    cov_eigvals = torch.diagonal(comp_full_covs[eidx])
                e_covs.append(cov_eigvals)
            input_covs.append(e_covs)

        tc_inds = [0, 1, 2]
            # input_tc_inds = np.repeat(tc_inds, self.gmm_comps)
        # input_tc_inds = np.repeat(tc_inds, 10)
        # input_tc_inds = [list(range(self.max_gmm_comps)) for t in range(self.num_components)]
        input_tc_inds = [list(range(self.max_gmm_comps)) for t in range(self.task_class[task_idx])]
        # torch.autograd.set_detect_anomaly(True)
        # self.tap_optimizer.zero_grad()
        self.tap_tasks_optimizer[task_idx].zero_grad()
        # for e in range(20):
        for e in range(15):
        # for e in range(40):
            # new_mus, new_covs = self.tap(self.mus, input_covs, input_tc_inds)
            # new_mus, new_covs = self.tap.forward_dedicated(self.mus, input_covs, input_tc_inds)
            # new_mus, new_covs = self.tap.forward_dedicated(self.mus, input_covs, input_tc_inds, with_update=True)
            # new_mus, new_covs = self.task_taps[task_idx].forward_dedicated(self.tasks_mus[task_idx], input_covs, input_tc_inds, with_update=True)
            new_mus, new_covs = self.network.task_taps[task_idx].forward_dedicated(self.tasks_mus[task_idx], input_covs, input_tc_inds, with_update=True)
                # Prune the unneeded mus
            pruned_mus, pruned_covs = [], []
            em_mus, em_covs = [], []
            # for c in range(self.num_components):
            for c in range(self.task_class[task_idx]):
                optimal_n = component_optimal_ns[c]
                # keep_mus = new_mus[c*self.max_gmm_comps:c*self.max_gmm_comps + optimal_n]
                # keep_covs = new_covs[c*self.max_gmm_comps:c*self.max_gmm_comps + optimal_n]
                # pruned_mus.append(torch.vstack(keep_mus).reshape(optimal_n, self.feature_space_dim))
                # pruned_covs.append(torch.stack(keep_covs).reshape(optimal_n, self.feature_space_dim, self.feature_space_dim).double())

                keep_mus = new_mus[c*self.max_gmm_comps:c*self.max_gmm_comps + optimal_n]
                keep_covs = new_covs[c*self.max_gmm_comps:c*self.max_gmm_comps + optimal_n]
                keep_mus = torch.vstack(keep_mus).reshape(optimal_n, self.feature_space_dim)
                keep_covs = torch.stack([torch.diag(d) for d in keep_covs])

                pruned_mus.append(keep_mus)
                pruned_covs.append(keep_covs.double())

                # em_mus.append(self.mus[c][:optimal_n])
                # em_covs.append(self.sigs[c][:optimal_n])
                em_mus.append(self.tasks_mus[task_idx][c][:optimal_n])
                em_covs.append(self.tasks_sigs[task_idx][c][:optimal_n])

            # l = self.main_loss.get_dynamic_sep_loss(new_mus, new_covs, self.min_dists, tc_inds, csep=self.csep) \
            # l = self.main_loss.get_non_uniform_dynamic_sep_loss(pruned_mus, pruned_covs, self.min_dists, csep=self.csep) \
            #         + 1 * component_wise_kl_div(self.mus, self.sigs, pruned_mus, pruned_covs)
            dm_pen = pen_domain(pruned_mus, domain=self.domain)
            tap_kl = component_wise_kl_div(em_mus, em_covs, pruned_mus, pruned_covs) 
            if self.with_wandb:
                wandb.log({"domain_pen": dm_pen})
                wandb.log({"tap_kl": tap_kl})
            self.print_to_log_file(f"domain pen {dm_pen}")
            self.print_to_log_file(f"tap kl {tap_kl}")
            # l = self.main_loss.get_non_uniform_dynamic_sep_loss(pruned_mus, pruned_covs, self.min_dists, csep=self.csep) + \
            #         1.5 * component_wise_kl_div(em_mus, em_covs, pruned_mus, pruned_covs) + \
            #         0.5 * dm_pen
            l = get_non_uniform_dynamic_sep_loss(pruned_mus, pruned_covs, self.tasks_min_dists[task_idx], csep=self.csep) + \
                    0.0001 * tap_kl + \
                    0.5 * dm_pen
            
            self.print_to_log_file(f"tap loss: {l}")
            if self.with_wandb:
                wandb.log({"tap_loss": l})
            # if (e + 1) != 30:
            # self.tap_amp_grad_scaler.scale(l).backward(retain_graph=True)
            # self.tap_amp_grad_scaler.unscale_(self.tap_optimizer)
            # # torch.nn.utils.clip_grad_norm_(self.tap.parameters(), 12)
            # torch.nn.utils.clip_grad_norm_(self.task_taps[task_idx].parameters(), 12)
            # self.tap_amp_grad_scaler.step(self.tap_optimizer)
            # self.tap_amp_grad_scaler.update()
            # self.tap_optimizer.zero_grad()
            self.tap_tasks_amp_grad_scaler[task_idx].scale(l).backward(retain_graph=True)
            self.tap_tasks_amp_grad_scaler[task_idx].unscale_(self.tap_tasks_optimizer[task_idx])
            # torch.nn.utils.clip_grad_norm_(self.tap.parameters(), 12)
            # torch.nn.utils.clip_grad_norm_(self.task_taps[task_idx].parameters(), 12)
            torch.nn.utils.clip_grad_norm_(self.network.task_taps[task_idx].parameters(), 12)
            self.tap_tasks_amp_grad_scaler[task_idx].step(self.tap_tasks_optimizer[task_idx])
            self.tap_tasks_amp_grad_scaler[task_idx].update()
            self.tap_tasks_optimizer[task_idx].zero_grad()

        pruned_mus_, pruned_covs_ = pruned_mus, pruned_covs
        # pruned_mus_, pruned_covs_ = [], []
        # for t in range(self.num_components):
        #     pruned_mus_t = torch.vstack(pruned_mus[t])[:, 0, :]
        #     pruned_mus_.append(pruned_mus_t)
        #     pruned_covs_t = torch.stack(pruned_covs[t])
        #     pruned_covs_.append(pruned_covs_t)
            # Measure the adjustment:
            # mean_changes, cov_changes = measure_change(self.mus, self.sigs, new_mus, new_covs)
        # mean_changes, cov_changes = measure_change(self.mus, self.sigs, pruned_mus_, pruned_covs_)
        # mean_changes, cov_changes = measure_change(self.tasks_mus[task_idx], self.tasks_sigs[task_idx], pruned_mus_, pruned_covs_)
        # for i, (mean_change, cov_change) in enumerate(zip(mean_changes, cov_changes)):
        #     self.print_to_log_file(f"mean[{i}] change: {mean_change}")
        #     self.print_to_log_file(f"cov[{i}] change: {cov_change}")
        #     if self.with_wandb:
        #         wandb.log({f"mean_change[{i}]": mean_change})
        #         wandb.log({f"cov_change[{i}]": cov_change})
            
            # self.mus, self.sigs = new_mus.detach(), new_covs.detach()
        # tap_momentum = 0.999 * (self.epoch / 50)
        # tap_momentum = 1 - (self.epoch / self.max_num_epochs)
        # tap_momentum = max(0.5 - (self.epoch / self.max_num_epochs), 0.2)
        # tap_momentum = max(0.5 - (self.epoch / self.max_num_epochs), 0.3)
        # tap_momentum = max(0.4 - np.power(self.epoch / self.max_num_epochs, 2), 0.3)
        # tap_momentum = min(0.3 + np.power(self.epoch / self.max_num_epochs, 1/2), 0.45) # high glb mean dc
        tap_momentum = 0.40
        # tap_momentum = 1 - np.power(self.epoch / self.max_num_epochs, 2)
        # tap_momentum = 0.1
        updated_mus, updated_covs = [], []
        # for t in range(self.num_components):
        for t in range(self.task_class[task_idx]):
                # self.mus[t], self.sigs[t] = new_mus[t].detach(), new_covs[t].detach()
                # self.mus[t] = new_mus[t]
                # self.mus[t] = (1 - tap_momentum) * self.mus[t][:component_optimal_ns[t]] + (tap_momentum * new_mus[t])

            # updated_mu = (1 - tap_momentum) * self.mus[t][:component_optimal_ns[t]] + (tap_momentum * pruned_mus_[t])
            updated_mu = (1 - tap_momentum) * self.tasks_mus[task_idx][t][:component_optimal_ns[t]] + (tap_momentum * pruned_mus_[t])
            updated_mu = torch.clamp(updated_mu, min=self.domain[0], max=self.domain[1])
            self.print_to_log_file(f"fin updated mu: {updated_mu}")
            updated_mus.append(updated_mu)
            # updated_mus.append(updated_mu.clone())
                # self.mus[t] = (1 - tap_momentum) * self.mus[t] + (tap_momentum * pruned_mus[t])
                # self.sigs[t] = new_covs[t]
                # self.sigs[t] = (1 - tap_momentum) * self.sigs[t][:component_optimal_ns[t]] + (tap_momentum * new_covs[t])
            # updated_cov = (1 - tap_momentum) * self.sigs[t][:component_optimal_ns[t]] + (tap_momentum * pruned_covs_[t])
            # updated_cov = (1 - tap_momentum) * self.tasks_sigs[task_idx][t][:component_optimal_ns[t]] + (tap_momentum * pruned_covs_[t])
            # self.print_to_log_file(f"fin updated cov: {updated_cov}")
            # if not is_positive_definite(updated_cov):
            #     # updated_cov = self.sigs[t][:component_optimal_ns[t]]
            #     updated_cov = self.tasks_sigs[task_idx][t][:component_optimal_ns[t]]
            # updated_covs.append(updated_cov)
            # updated_covs.append(updated_cov.clone())
                # self.sigs[t] = (1 - tap_momentum) * self.sigs[t] + (tap_momentum * pruned_covs[t])

        # self.mus, self.sigs = updated_mus, updated_covs
        # self.tasks_mus[task_idx], self.tasks_sigs[task_idx] = updated_mus, updated_covs
        self.tasks_mus[task_idx] = updated_mus # NOTE
        # self.network.set_feature_space_distribution_parameters(updated_mus, updated_covs, self.tasks_weights[task_idx], task=task_idx)
        
        # self.network.weights = self.weights
        # self.choleskys = get_choleskys(self.sigs)

        # self.network.construct_feature_space_gmm_implicit()


        # for tidx in range(self.total_task_num):
        # self.network.construct_task_feature_space_gmm_implicit(task_idx)


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
        # self.inference_apply_nonlin = lambda x: x

        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        # self.refill_queue_and_train_gmm()
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        # net.train_gmms(self.dynamic_dist_network.feature_space_qs, self.mus, self.sigs)

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

                if self.network.ood_detection_mode:
                    softmax_pred, anomaly_probabilities = softmax_pred
                    anomaly_probabilities = anomaly_probabilities.transpose([0] + [i + 1 for i in self.transpose_backward])
                    anomaly_probabilities = anomaly_probabilities[:self.task_class[int(task_id)]]
                
                softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                softmax_pred = softmax_pred[:self.task_class[int(task_id)]]



                if save_softmax:
                    softmax_fname = join(output_folder, fname + ".npz")
                    if self.network.ood_detection_mode:
                        anomaly_fname = join(output_folder, fname + "_anomaly_prob" + ".npz")
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
                    if self.network.ood_detection_mode:
                        np.save(join(output_folder, fname + "_anomaly_prob" + ".npy"), anomaly_probabilities)
                        anomaly_probabilities = join(output_folder, fname + "_anomaly_prob" + ".npy")


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
                
                results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                         ((anomaly_probabilities, join(output_folder, fname + "_anomaly_prob" + ".nii.gz"),
                                                           properties, interpolation_order, self.regions_class_order,
                                                           None, None,
                                                           anomaly_fname, None, force_separate_z,
                                                           interpolation_order_z),
                                                          )
                                                         )
                               )

            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])
            
            pred_gt_tuples.append([join(output_folder, fname + "_anomaly" + ".nii.gz"),
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

    def refill_queue_and_train_gmm(self):
        dl_tr, dl_val = self.get_basic_generators()
        print("unpacking dataset")
        unpack_dataset(self.folder_with_preprocessed_data)
        print("done")
        tr_gen, _ = get_moreDA_augmentation_uniseg(
            dl_tr, dl_val,
            self.data_aug_params[
                'patch_size_for_spatialtransform'],
            self.data_aug_params,
            deep_supervision_scales=self.deep_supervision_scales,
            pin_memory=self.pin_memory,
            use_nondetMultiThreadedAugmenter=False,
            task_num=self.total_task_num, iter_each_task_epoch=int(self.num_batches_per_epoch // self.total_task_num)
                )
        
        self.update_iter = 999
        self.epoch = 0
        for b in range(self.num_batches_per_epoch // 2):
            self.run_iteration(tr_gen, False, False)

        if not self.network.gmm_fitted:
            # self.network.module.init_gmms(self.mus, self.sigs)
            self.network.train_gmms(self.dynamic_dist_network.feature_space_qs, self.mus, self.sigs)


    def poisson_disk_sample(self, N, d, m, k=100):
        """
        N: number of vectors,
        d: dimension of each vector, 
        m: minimum separation
        """
        def get_random_point_around(point, min_dist):
            return point + np.random.uniform(-min_dist, min_dist, size=d)

        # Start with an initial random point
        points = []
        first_point = np.random.uniform(0, 1, size=d)
        points.append(first_point)
        active_list = [first_point]

        while len(points) < N and active_list:
            current_point = active_list.pop(np.random.randint(len(active_list)))
            
            for _ in range(k):
                new_point = get_random_point_around(current_point, m)
                
                # Check if the new point is within bounds [0,1]^d and has the minimum distance
                if all(0 <= coord <= 1 for coord in new_point):
                    if all(np.linalg.norm(new_point - p) >= m for p in points):
                        points.append(new_point)
                        active_list.append(new_point)
                        if len(points) >= N:
                            break

        if len(points) < N:
            raise RuntimeError(f"Failed to generate {N} vectors after exhausting all possibilities.")

        return torch.from_numpy(np.array(points))

    def multi_comp_poisson_disk_sample(self, N, k, d, m, max_iter=100):

        def get_random_point_around(point, min_dist):
            return point + np.random.uniform(-min_dist, min_dist, size=d)

        points, centers = [], []
        first_center = np.random.uniform(-1+m, 1-m, size=d)
        first_points = [get_random_point_around(first_center, m/2) for _ in range(k)]
        # in some radius, choose 
        points.append(first_points)
        # centers.append(first_center)
        active_list = [first_center]

        # choose centers each `m` apart
        while len(centers) < N:
            
            new_center = np.random.uniform(-1+m, 1-m, size=d)

        while len(points) < N and active_list:
            current_center = active_list.pop(np.random.randint(len(active_list)))
            new_points = [get_random_point_around(current_center, m/2) for i in range(k)]
            # check 

            active_list.append()
            
            for _ in range(max_iter):
                for new_point in new_points:
                    # Check if the new point is within bounds [-1,1]^d and has the minimum distance
                    if all(-1 <= coord <= 1 for coord in new_point):
                        if all(np.linalg.norm(new_point - p) >= m for p in points):
                            points.append(new_point)
                            active_list.append(new_point)
                            if len(points) >= N:
                                break

        if len(points) < N:
            raise RuntimeError(f"Failed to generate {N} vectors after exhausting all possibilities.")

        return torch.from_numpy(np.array(points))


    def optimal_sep(self, N, d):
        torch.manual_seed(42)
        class OptimumSep(nn.Module):

            def __init__(self, t=0.7, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.t = t

            def forward(self, x):

                return x.matmul(x.T).div(self.t).exp().sum(dim=-1).log().mean()

        sep_cond = OptimumSep()
        x = torch.randn((N, d), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.SGD([x], lr=0.001)
        min_loss = 100
        optimal_target = None

        epochs = 5000
        for i in range(epochs):

            x_norm = torch.nn.functional.normalize(x, dim=1)
            loss = sep_cond(x_norm)
            if i % 100 == 0:
                # print(i, loss.item())
                pass
            if loss.item() < min_loss:
                min_loss = loss.item()
                optimal_target = x_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # min_dist = torch.cdist(optimal_target, optimal_target).fill_diagonal_(-torch.inf).max().item()
        # min_dist = torch.cdist(optimal_target, optimal_target).fill_diagonal_(torch.inf).min().item()

        # Find the maximum variance under which the distributions are still well-separated 
        # max_var = min_dist / np.sqrt(d)
        # max_var = np.square(min_dist / (2 * self.csep)) / d

        # min_dist = torch.cdist(optimal_target, optimal_target).fill_diagonal_(-torch.inf).max().item()
        min_dist = -torch.inf 
        for i in range(optimal_target.shape[0]):
            for j in range(i+1, optimal_target.shape[0]):
                dist = torch.dist(optimal_target[i], optimal_target[j])
                if dist > min_dist :
                    min_dist = dist 

        # Find the maximum variance under which the distributions are still well-separated 
        # max_var = min_dist / np.sqrt(d)
        # max_var = np.square(min_dist / csep) / d
        max_var = np.power((min_dist / self.csep).detach().cpu().numpy(), 2) / d
        # max_var = np.power(min_dist / self.csep, 2) / d
        # max_var = (min_dist - 1e-04) / (self.csep * np.sqrt(d))
        

        # return optimal_target.detach(), min_dist, max_var
        # return optimal_target.detach(), min_dist.detach(), np.square(max_var.detach())
        return optimal_target.detach(), min_dist.detach(), max_var#.detach()


    def partition_interval(self, intervals, center, m):
        for idx in range(len(intervals)):
            interval = intervals[idx]

            if center >= interval[0] and center <=interval[1]:
                # partition this interval
                updated_interval = [(interval[0], center-m), (center+m, interval[1])]
                # check for overlap
                if center-m <= interval[0] or center+m >= interval[1]:
                    if idx + 1 < len(intervals):
                        intervals = intervals[:idx] + intervals[idx+1:]
                    else:
                        intervals = intervals[:idx]

                # remove old interval and break
                if idx + 1 < len(intervals):
                    intervals = intervals[:idx] + updated_interval + intervals[idx+1:]
                else:
                    intervals = intervals[:idx] + updated_interval
                
                break
        
        return intervals

    def init_centers(self, N, dim, ms, sort=False, domain=[-1, 1]):
        torch.manual_seed(433)
        np.random.seed(433)
        centers = []
        # m = max(ms)
        m = 0#min(ms)/2
        all_position_intervals = [[(domain[0]+m, domain[1]-m)] for _ in range(dim)]
        idx = 0
        while all(len(pos_interval) > 0 for pos_interval in all_position_intervals) and len(centers) < N:
            m = ms[idx]
            # 1. choose a new center
            center = np.asarray([
                np.random.uniform(*random.choices(all_position_intervals[d], weights=[r[1]-r[0] for r in all_position_intervals[d]])[0])
                for d in range(dim)
            ])
            # print(center)
            centers.append(center)

            # 2. partition intervals
            for d in range(dim):
                new_interval = self.partition_interval(all_position_intervals[d], center[d], m)
                # print(f"{d} new interval: {new_interval}")
                all_position_intervals[d] = new_interval
            idx += 1
        if sort:
            centers = sorted(centers, key=lambda x: np.linalg.norm(x))
        centers = torch.from_numpy(np.vstack(centers))
        return centers

    def init_optim_gmm_centers(self, centers, dim, ms, c, return_weights=False):
        K = self.gmm_comps
        N = len(centers)
        new_csep = 2
        # s = new_csep / 10
        s = 1 / new_csep
        all_comp_centers = [[] for _ in range(N)]
        all_comp_centers_torch = [torch.empty(K, dim) for _ in range(N)]
        all_comp_covs_torch = [torch.empty(K, dim, dim) for _ in range(N)]
        for i, center in enumerate(centers):
            m = ms[i]
            # m = min_dist
            for k in range(K):
                # (m/(2c))^2/dim = v
                # comp_center = np.random.uniform(center-s, center+s)
                point = np.random.normal(0, 1, dim)
                point /= np.linalg.norm(point)
                # rad = np.power(np.random.uniform(0, 1), 1/dim)
                rad = np.power(np.random.uniform(0, 1), 1)
                point *= rad * s
                comp_center = point + center.detach().cpu().numpy()
                # comp_center = np.random.normal(center-s, center+s)
                # comp_center = center
                # comp_cov = (np.power(m/(2*c), 2) / dim) * torch.eye(dim) + (1e-05*torch.eye(dim))
                # comp_cov = (np.power((m * s / csep), 2) / dim) * torch.eye(dim) + (1e-05*torch.eye(dim))
                # comp_cov = max_var / (1 / s) * torch.eye(dim) + (1e-05*torch.eye(dim))
                all_comp_centers[i].append(comp_center)
                all_comp_centers_torch[i][k] = torch.from_numpy(comp_center)

        new_min_dists = []
        for i in range(len(all_comp_centers_torch)): # iterates over classes
            for j in range(i+1, len(all_comp_centers_torch)): # iterates over classes

                # for every component of the first class,
                smallest_class_dist = torch.inf 
                for m in range(len(all_comp_centers_torch[i])):
                    # for every component of the other class
                    for n in range(i+1, len(all_comp_centers_torch[j])):
                        # calculate the distance
                        dist = torch.dist(all_comp_centers_torch[j][n], all_comp_centers_torch[i][m])
                        # if the distance between the components (belonging to two different classes) is less than the smallest distance
                        if dist < smallest_class_dist :
                            # update
                            smallest_class_dist = dist 
                
                print(smallest_class_dist)
                new_min_dists.append(smallest_class_dist.item())

        new_min_dist = np.min(new_min_dists)
        # new_csep = 2
        new_csep = self.csep
        for i in range(len(all_comp_covs_torch)):
            for k in range(K):
                comp_cov = np.square((new_min_dist - 1e-04) / new_csep) * (1 / dim) * torch.eye(dim) + (1e-05*torch.eye(dim))
                all_comp_covs_torch[i][k] = comp_cov
            
        all_comp_centers_torch = torch.stack(all_comp_centers_torch)
        all_comp_covs_torch = torch.stack(all_comp_covs_torch)

        weights = [torch.ones(K, 1) / K for _ in range(N)]

        if return_weights:
            return all_comp_centers_torch, all_comp_covs_torch, weights
        return all_comp_centers_torch.detach(), all_comp_covs_torch.detach()
    

    def init_gmm_centers(self, centers, dim, ms, c, return_weights=False):
        K = self.gmm_comps
        N = len(centers)
        s = 1
        all_comp_centers = [[] for _ in range(N)]
        all_comp_centers_torch = [torch.empty(K, dim) for _ in range(N)]
        all_comp_covs_torch = [torch.empty(K, dim, dim) for _ in range(N)]
        weights = [torch.ones(K, 1) / K for _ in range(N)]
        for i, center in enumerate(centers):
            m = ms[i]
            for k in range(K):
                # comp_cov = (np.power(m/(2*c), 2) / dim) * torch.eye(dim) + (1e-05*torch.eye(dim))
                comp_cov = (np.power((m * s / c), 2) / dim) * torch.eye(dim) + (1e-05*torch.eye(dim))
                all_comp_covs_torch[i][k, :] = comp_cov
                comp_center = np.random.uniform(center-(m/2), center+(m/2))
                # comp_center = comp_center = np.random.uniform(center-(0.1 * s), center+(0.1 * s))
                all_comp_centers_torch[i][k] = torch.from_numpy(comp_center)

        # return torch.stack(all_comp_centers_torch), torch.stack(all_comp_covs_torch)
        if return_weights:
            return all_comp_centers_torch, all_comp_covs_torch, weights
        return all_comp_centers_torch, all_comp_covs_torch


        # dists = []
        # for i, center_set in enumerate(all_comp_centers):
        #     weights = torch.ones(K) / K
        #     dist = GaussianMixtureModel(
        #         Categorical(weights), 
        #         [
        #             MultivariateNormal(torch.from_numpy(center_set[k]), 0.001*torch.eye(dim))
        #             for k in range(K)
        #         ]
        #     )
        #     dists.append(dist)

        # weights = Categorical(torch.ones(N) / N)
        # comp_gmm = GaussianMixtureModel(weights, dists)


    def init_uniform_mixture_weights(self):
        weights = []
        for _ in range(self.num_components):
            weights.append(torch.ones(self.gmm_comps)/self.gmm_comps)

        return weights


    def init_task_distribution(self, vars, csep, num_classes, domain=[-1, 1], center_method="partition"):

        print(f"{vars}, {csep}, {self.feature_space_dim}")
        min_dists = [csep * np.sqrt(self.feature_space_dim * v) for i, v in enumerate(vars)]
        # self.min_dists = min_dists
        print(f"{min_dists}")

        if center_method == "partition":
        
            max_trys = 15000
            for t in range(max_trys):
                if t % 100 == 0: print(t)
                try:
                    # centers = self.init_centers(self.num_components, self.feature_space_dim, min_dists, sort=False)
                    centers = self.init_centers(num_classes, self.feature_space_dim, min_dists, domain=domain, sort=False)
                    break
                except ValueError:
                    pass
        
        elif center_method == "optim":

            centers, min_dist, max_var = self.optimal_sep(num_classes, self.feature_space_dim)
            min_dists = [min_dist for _ in range(len(min_dists))]

        # mus, sigs = self.init_gmm_centers(centers, self.feature_space_dim, min_dists, csep)
        mus, sigs  = self.init_optim_gmm_centers(centers, self.feature_space_dim, min_dists, csep)
        weights = self.init_uniform_mixture_weights()

        return mus, sigs, weights, min_dists

    def init_all_distributions(self, tasks_vars, tasks_cseps, domain=[-1, 1], **center_kwargs):
        
        tasks_mus, tasks_sigs, tasks_weights, tasks_min_dists, tasks_min_dist = [], [], [], [], []
        for task_id in range(self.total_task_num):
            task_num_classes = self.task_class[task_id]
            task_mus, task_sigs, task_weights, min_dists = self.init_task_distribution(tasks_vars[task_id], tasks_cseps[task_id], task_num_classes, domain=domain, **center_kwargs)

            self.print_to_log_file(f"task {task_id} initial means: {task_mus}")
            self.print_to_log_file(f"task {task_id} initial vars: {task_sigs}")
            self.print_to_log_file(f"task {task_id} initial weights: {task_weights}")
            self.print_to_log_file(f"task {task_id} min dists: {min_dists}")

            # move to cuda
            if torch.cuda.is_available():
                if isinstance(task_mus, list):
                    # for class_idx in range(self.num_components):
                    for class_idx in range(self.task_class[task_id]):
                        # self.print_to_log_file(f"prev size {t}: {[m.size() for m in self.mus[t]]}")
                        task_mus[class_idx] = task_mus[class_idx].cuda()
                        task_sigs[class_idx] = task_sigs[class_idx].cuda()
                        task_weights[class_idx] = task_weights[class_idx].cuda()
                        # self.choleskys[t] = self.choleskys[t].cuda()
                else:
                    task_mus = task_mus.cuda()
                    task_sigs = task_sigs.cuda()

            tasks_mus.append(task_mus)
            tasks_sigs.append(task_sigs)
            tasks_weights.append(task_weights)
            tasks_min_dists.append(min_dists)
            tasks_min_dist.append(max(min_dists))

        return tasks_mus, tasks_sigs, tasks_weights, tasks_min_dists, tasks_min_dist

    
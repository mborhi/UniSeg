#!/bin/bash

# Traniner="UniSeg_Trainer_DP"
# Traniner="UniSegExtractor_Trainer_DP"
# Traniner="UniSegExtractor_Trainer_Fast_DP"
Traniner="UniSegExtractorMod_Trainer_DDP"

GPU=0

task_exp="test_ddp_est_sep"
wandb_project_name="test_est_sep"
exp_name=$task_exp"_gnlll_qt_est_sep_rrcycle1iter_dc_ds_3dim_gmmseg_newtap_dyn_cls_bayes"

# CUDA_VISIBLE_DEVICES=$GPU nnUNet_n_proc_DA=32 nnUNet_train 3d_fullres $Traniner 97 0 -exp_name $task_exp
# CUDA_VISIBLE_DEVICES=$GPU nnUNet_n_proc_DA=32 nnUNet_train_DP 3d_fullres $Traniner 94 0 -exp_name $task_exp -gpus 2 -loss_type "kl" -feature_space_dim 16 -update_iter 2 -max_num_epochs 10 -queue_size 5000 -batch_size 1 # --with_wandb -wandb_project_name $wandb_project_name
#-val --valbest # --with_wandb -wandb_project_name $wandb_project_name 
CUDA_VISIBLE_DEVICES=$GPU nnUNet_n_proc_DA=32 python3 -m torch.distributed.launch --master_port=4123 --nproc_per_node=1 run/run_training_DDP.py "3d_fullres" $Traniner 92 0 -exp_name $exp_name --dbs -loss_type "gnlll" -feature_space_dim 3 -update_iter 1 -max_num_epochs 10 -queue_size 10000 -batch_size 1 -num_gpus 1 --with_wandb -wandb_project_name $wandb_project_name && \
cd /data/nnUNet_trained_models && \
python3 UniSeg_Metrics_quick_test.py --result_path $exp_name'/3d_fullres/Task092_QuickTest/'$Traniner'__DoDNetPlans/fold_0/validation_raw/'
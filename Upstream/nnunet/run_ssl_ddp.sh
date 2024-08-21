#!/bin/bash

# Traniner="UniSeg_Trainer_DP"
# Traniner="UniSegExtractor_Trainer_DP"
# Traniner="UniSegExtractor_Trainer_Fast_DP"
Traniner="UniSegExtractor_Trainer_DDP"

GPU=0,1

task_exp="test_ddp"
wandb_project_name="test"

# CUDA_VISIBLE_DEVICES=$GPU nnUNet_n_proc_DA=32 nnUNet_train 3d_fullres $Traniner 97 0 -exp_name $task_exp
# CUDA_VISIBLE_DEVICES=$GPU nnUNet_n_proc_DA=32 nnUNet_train_DP 3d_fullres $Traniner 94 0 -exp_name $task_exp -gpus 2 -loss_type "kl" -feature_space_dim 16 -update_iter 2 -max_num_epochs 10 -queue_size 5000 -batch_size 1 # --with_wandb -wandb_project_name $wandb_project_name
CUDA_VISIBLE_DEVICES=$GPU nnUNet_n_proc_DA=32 python3 -m torch.distributed.launch --master_port=4123 --nproc_per_node=2 run/run_training_DDP.py "3d_fullres" $Traniner 93 0 -exp_name $task_exp --dbs -loss_type "gnlll" -feature_space_dim 32 -update_iter 2 -max_num_epochs 3 -queue_size 5000 -batch_size 2 -num_gpus 2 # --with_wandb -wandb_project_name $wandb_project_name

python3 UniSeg_Metrics_test.py --result_path $task_exp'/3d_fullres/Task093_FineTune/'$Traniner'__DoDNetPlans/fold_0/validation_raw/'
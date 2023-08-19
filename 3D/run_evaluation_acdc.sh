#!/bin/sh

DATASET_PATH=DATASET_Acdc
#CHECKPOINT_PATH=unetr_pp/evaluation/unetr_pp_acdc_checkpoint
CHECKPOINT_PATH=/work/scratch/niggemeier/projects/project_UNETRpp/unetr_plus_plus/output_acdc_deform_LKA_depth_4
export PYTHONPATH=./
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export d_lka_former_preprocessed="$DATASET_PATH"/d_lka_former_raw/d_lka_former_raw_data/Task01_ACDC
export d_lka_former_raw_data_base="$DATASET_PATH"/d_lka_former_raw

python d_lka_former/run/run_training.py 3d_fullres d_lka_former_trainer_acdc 1 0 -val --trans_block TransformerBlock_3D_single_deform_LKA --depth 4

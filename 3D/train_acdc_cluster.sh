#!/bin/bash
cd /work/scratch/niggemeier/projects/project_UNETRpp/unetr_plus_plus/
chmod +x train_acdc_cluster.sh

DATASET_PATH=DATASET_Acdc

export PYTHONPATH=./
export RESULTS_FOLDER=output_acdc_deform_LKA_kernel_sizes_1
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task01_ACDC
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw


/work/scratch/niggemeier/miniconda3/envs/unetr_pp/bin/python /work/scratch/niggemeier/projects/project_UNETRpp/unetr_plus_plus/unetr_pp/run/run_training.py "$@"
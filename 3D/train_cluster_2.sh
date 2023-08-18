#!/bin/bash
cd /work/scratch/niggemeier/projects/project_UNETRpp/unetr_plus_plus/
chmod +x train_cluster_2.sh

DATASET_PATH=DATASET

export PYTHONPATH=./
export RESULTS_FOLDER=output_synapse_Deform_LKA_inter_ckpts_3_skip_seed_1
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

/work/scratch/niggemeier/miniconda3/envs/unetr_pp/bin/python /work/scratch/niggemeier/projects/project_UNETRpp/unetr_plus_plus/unetr_pp/run/run_training.py "$@"
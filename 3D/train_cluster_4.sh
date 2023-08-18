#!/bin/bash
cd /work/scratch/niggemeier/projects/project_UNETRpp/unetr_plus_plus/
chmod +x train_cluster_4.sh

DATASET_PATH=DATASET

export PYTHONPATH=./
export RESULTS_FOLDER=output_synapse_LKA_with_3D_Conv_2nd_run
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

/work/scratch/niggemeier/miniconda3/envs/unetr_pp/bin/python /work/scratch/niggemeier/projects/project_UNETRpp/unetr_plus_plus/unetr_pp/run/run_training.py "$@"
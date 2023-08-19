#!/bin/sh

DATASET_PATH=DATASET

export PYTHONPATH=./
export RESULTS_FOLDER=output_synapse_testing_depths
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

/work/scratch/niggemeier/miniconda3/envs/unetr_pp/bin/python unetr_pp/run/run_training.py 3d_fullres unetr_pp_trainer_synapse 2 0

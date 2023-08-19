#!/bin/bash
cd /work/scratch/niggemeier/projects/MCF/code/
chmod +x train_dlkaformer.sh
/work/scratch/niggemeier/miniconda3/envs/unetr_pp/bin/python /work/scratch/niggemeier/projects/MCF/code/test_LA_MCF.py "$@"
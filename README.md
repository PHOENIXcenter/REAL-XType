# REAL-XType

This is the python and tensorflow implementation of the subtyping algorithm 'REAL-XType' proposed in the paper 'REAL: The Principle and Approach for Clinical Multicenter Proteomic Tumor Analysis'.

Here are the instructions:

1. Create an anaconda environment with anaconda installed on you desktop/server, note that we recommand to test on a windows system for the best reproducibility.

conda env create -f code/real_env.yml


2. To reproduce our benchmarking results and the application experiment which transfers the original subtypes from Jiang et al.'s cohort to SH, GZ, FZ and Gao et al.'s cohorts, you need to activate the conda environment and run following batch commands

conda activate real_env

code/bash cmd.bat
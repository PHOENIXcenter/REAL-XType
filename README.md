# REAL-XType

This is the python and tensorflow implementation of the subtyping algorithm 'REAL-XType' proposed in the paper 'REAL: The Principle and Approach for Clinical Multicenter Proteomic Tumor Analysis'.

Here are the instructions:

1. Create an anaconda environment with anaconda installed on you desktop/server.

conda create -n xtype_env python=3.7 tensorflow-gpu=2.2 numpy scikit-learn scipy pandas progressbar2 statsmodels

conda activate xtype_env

conda install -c conda-forge -c bioconda -c numba matplotlib lifelines matplotlib-venn harmonypy gseapy numba


2. To reproduce the real-world application which transfers the original subtypes from Jiang et al.'s cohort to SH, GZ, FZ and Gao et al.'s cohorts, you need to download the data which will be available after the paper publication and run following script

conda activate xtype_env

python cmd.sh
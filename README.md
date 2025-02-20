# REAL-XType

This repository contains the Python and TensorFlow implementation of the subtyping algorithm **REAL-XType**, as proposed in the paper *"REAL principle unveils metabolic vulnerabilities and robust prognostic stratification in hepatocellular carcinoma"*.

## Instructions

### 1. Set Up the Anaconda Environment

To ensure reproducibility, we recommend testing on a Windows system. Follow these steps to create the Anaconda environment:

1. Install Anaconda on your desktop/server if you haven't already.
2. Create the environment using the provided `real_env.yml` file:

   ```bash
   conda env create -f code/real_env.yml
   ```

### 2. Reproduce Benchmarking Results and Application Experiments

   ```bash
   conda activate real_env
   code/bash cmd.bat
   ```

### 3. Expected output
In the generated "data\benchmark" folder, you are expected to see all saved models with different hyperparameters, their best models and two images showing the benchmarking results.
In the generated "data\application" folder, you are expected to see survival curves on all test datasets.

Although it usually trains a single model within 1 min, the whole process will take several hours by grid-searching for all possible hyperparameters and benchmarking all algorithms.

## Contact
For questions, issues, or suggestions, please open an issue on the GitHub repository or contact the maintainers directly:
Linhai Xie: xielinhai@gmail.com

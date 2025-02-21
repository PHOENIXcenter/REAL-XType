# REAL-XType

This repository contains the Python and TensorFlow implementation of the subtyping algorithm **REAL-XType**, as proposed in the manuscript *"REAL principle unveils metabolic vulnerabilities and robust prognostic stratification in hepatocellular carcinoma"*.

## Instructions

### 1. Set Up the Anaconda Environment

To ensure the best reproducibility, we recommend testing on a Windows 11 desktop. Follow these steps to create the Anaconda environment:

1. Install Anaconda on your desktop/server if you haven't already.
2. Create the environment using the provided `real_env.yml` file. It may take from a few minutes to an hour, which depends on your network.

   ```bash
   conda env create -f code\real_env.yml
   ```

### 2. Reproduce Benchmarking Results and Application Experiments. 

The proteomic data after preprocessing and survival information are providede in the data folder.

Since the whole process will take several hours by grid-searching for all hyperparameters of all algorithms with different random seeds, here we simply provide the best parameter of each model for a quick reproduction, which may take an hour. 

   ```bash
   conda activate real_env
   code\cmd.bat
   ```

You can try the full searching process by changing the commented part in `para_space.py` and adjust the iteration range of `%%j ` in `cmd.bat`.

### 3.Functionanility of each command in the BAT script

`python code\benchmark.py --method %%i --para_id %%j --valid_fold %%k  --seed %%l`

It trains a model of `method` i with its hyperparameter set j, which is defined in `para_space.py`.
The `valid_fold` 0-4 indicates a regular training process of 5-fold cross validation and the `valid_fold` 5 means training the model using all data.
We repeat the training of each method with 5 different random seeds.
After all iterations of this command, you will see the folder of each method in the `data\benchmark` folder, containing their model weights and subtyping results of all samples.

`python code\benchmark.py --evaluate True`

It summarizes the test results of each method, selecting the best hyperparameter for each method and finally calculate the evaluation metrics on all test datasets, which is logged in `best_results.csv` and `best_vals.csv`.

`python code\plot.py`.

It generates two bar plots according to the `best_vals.csv` file, which are the benchmarking results in the manuscript.

`python code\training.py --para_id %%i --valid_fold %%j`

It repeats the training process of REAL-XType. You can simply pass this process by copying the folder of `XType` from `benchmark` to `application`

`python code\evaluation.py`

It evaluates the model on all testing datasets and generates the KM curves.

## Contact
For questions, issues, or suggestions, please open an issue on the GitHub repository or contact the maintainers directly:
xielinhai@gmail.com

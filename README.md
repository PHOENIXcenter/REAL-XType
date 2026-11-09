# REAL-XType

This repository contains the Python and TensorFlow implementation of the subtyping algorithm **REAL-XType**, as proposed in the manuscript *"REAL principle unveils metabolic vulnerabilities and robust prognostic stratification in hepatocellular carcinoma"*.

## Environment setup

The supplied workflow targets Windows 11 with Anaconda. Run all commands from the repository root (the directory containing this README).

```powershell
conda env create -f code/real_env.yml
conda activate real_env
```

Software and library versions are specified in `code/real_env.yml`, including Python 3.9, TensorFlow GPU 2.6.0, CUDA Toolkit 11.3.1, and cuDNN 8.2.1. XGBoost 1.6.2 is installed through the pip section. This XGBoost pin was added to complete the environment specification; the original file did not record the version used for the manuscript experiments. Agreement with the original XGBoost results needs verification.

To update an existing environment:

```powershell
conda env update -n real_env -f code/real_env.yml
conda activate real_env
```

Training defaults to CPU execution (`--device /cpu`). GPU execution requires a compatible NVIDIA GPU and driver and can be selected with `--device /GPU:0`. The batch workflow uses the default CPU setting. Runtime depends on hardware and environment; the complete workflow can take several hours or longer.

## Input data and scope

`data/HCC_datasets.csv` contains preprocessed proteomic measurements and survival information for 1,284 samples from the Jiang, SH, GZ, FZ, and Gao cohorts. It has two header rows: the first identifies `meta` and `proteome` columns; the second contains field names and protein identifiers. The scripts read it using `pandas.read_csv(..., header=[0, 1])`.

This workflow starts from the preprocessed data file. Raw-data preprocessing scripts are not included. Training uses Jiang, SH, and GZ samples with `ex_valid != 1`; held-out samples and FZ/Gao cohorts are evaluated separately.

## Run the complete workflow

After activating the environment, run:

```powershell
.\code\cmd.bat
```

The batch file runs benchmark training, benchmark evaluation, benchmark plotting, application training, and application evaluation in that order. It uses parameter ID 0 for each method, benchmark seeds 0–4, and application seed 0.

### 1. Benchmark training

Example of one training run:

```powershell
python code/benchmark.py --method XType --para_id 0 --valid_fold 0 --seed 0
```

The batch workflow includes `LogisticRegression`, `XGBoost`, `RandomForest`, `DNN`, and `XType`. Configurations are defined in `code/para_space.py`.

Fold settings 0–4 train cross-validation models, holding out the indicated fold. Setting 5 trains on all eligible training samples, excluding the held-out validation cohorts. Run all six settings for every method and seed before evaluation; predictions for the folds accumulate in the same CSV.

Outputs are saved under:

```text
data/benchmark/<method>/para-<parameter_id>/seed-<seed>/results.csv
```

Neural-network methods also save weights in the corresponding `model_weights/` directory. The scikit-learn/XGBoost branch saves predictions but does not save fitted models.

### 2. Benchmark evaluation and plotting

After completing all five methods, all six fold settings, and seeds 0–4:

```powershell
python code/benchmark.py --evaluate True
python code/plot.py
```

Evaluation selects the parameter configuration using the SH/GZ cross-validation DFS log-rank score and saves `results.csv`, `best_results.csv`, and `best_vals.csv` in `data/benchmark/`. Plotting reads `best_vals.csv` and generates `C-Index.pdf` and `Log-rank score.pdf` in that directory. The plotting code expects five benchmark seeds.

### 3. Application training

Run this command for each fold setting from 0 through 5, or use the batch workflow:

```powershell
python code/training.py --method XType --para_id 0 --valid_fold 0 --seed 0
```

The default prediction output is:

```text
data/application/XType/test-FZ_Gao/para-0/seed-0/results.csv
```

Application evaluation requires the `test-FZ_Gao` directory level. Use application training to create the required structure. Simply copying `benchmark/XType` directly into `application/` does not produce this structure.

### 4. Application evaluation and Kaplan-Meier curves

After training all six fold settings:

```powershell
python code/evaluation.py --method XType --seed 0
```

The script uses SH/GZ cross-validation metrics for parameter selection and generates OS/DFS Kaplan-Meier curves for the configured validation and low-risk groups. Summary metrics are stored under `data/application/XType/test-FZ_Gao/`. For the supplied single configuration, processed predictions and plots are stored in `data/application/XType/results_viz/para-0/seed-0/`.

## Hyperparameter search and reproducibility

`code/para_space.py` supplies one configuration per method and includes commented candidate grids. To expand the search, edit the parameter definitions and the parameter-ID loop bounds in `code/cmd.bat`: benchmark uses `%%j`; application uses `%%i`.

The current XGBoost constructor passes only the seed. Its `n_estimators` and `max_depth` entries in `para_space.py` are not passed to the estimator, so changing those entries alone does not change training. Verify the intended estimator settings before using an expanded search to reproduce manuscript results.

Use a fresh output directory or archive previous outputs before changing parameter grids or rerunning experiments: predictions accumulate across folds, and application evaluation reuses an existing summary metrics file. Training scripts set random seeds, but results can vary across hardware and library versions.

For manuscript submission, record the exact Git commit ID with `git rev-parse HEAD` and include the matching code archive as supplementary material.

## License and contact

Code is distributed under the MIT License; see `LICENSE`.

For questions, open an issue at <https://github.com/PHOENIXcenter/REAL-XType> or contact xielinhai@gmail.com.

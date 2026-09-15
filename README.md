# Turbofan Engine Remaining Useful Life Prediction

An NTI machine-learning project that predicts the Remaining Useful Life (RUL) of aircraft turbofan engines from multivariate sensor histories. The project compares four regression models and studies whether degradation-trend features improve predictions across increasingly difficult operating scenarios.

## Project Status

**Phase:** Four FD001 models complete; final comparison and scenario expansion

The repository foundation, local FD001 dataset, shared feature pipeline, four
trained model packages, and local comparison dashboard are ready.

## Problem Statement

Unexpected engine failure can cause safety risks, downtime, and expensive maintenance. Predictive maintenance estimates how many operating cycles remain before failure so maintenance can be scheduled earlier. Predictions that are too late are more dangerous than predictions that are slightly early, so the project evaluates both ordinary regression error and NASA's asymmetric scoring function.

## Dataset

The project uses the [NASA C-MAPSS Turbofan Jet Engine dataset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps), originally provided by the NASA Ames Prognostics Center of Excellence.

Each row represents one engine at one operating cycle and contains:

- Engine unit identifier
- Cycle number
- Three operating settings
- Twenty-one sensor measurements

The four scenarios increase in difficulty:

| Scenario | Training engines | Test engines | Operating conditions | Fault modes |
|---|---:|---:|---:|---:|
| FD001 | 100 | 100 | 1 | 1 |
| FD002 | 260 | 259 | 6 | 1 |
| FD003 | 100 | 100 | 1 | 2 |
| FD004 | 249 | 248 | 6 | 2 |

The dataset is not committed to Git. Each contributor should download it from Kaggle and extract it locally under `data/raw/cmapss/`.

## Research Question

Can degradation-trend feature engineering improve remaining-useful-life predictions and remain effective when engines operate under multiple conditions and fault modes?

## Models

The experiment will compare:

1. Ridge Regression
2. Random Forest Regressor
3. XGBoost Regressor
4. CatBoost Regressor

Every model will use the same engine-level partitions, features, target definition, and evaluation code.

## Planned Improvements

- Cap early-life RUL values to reduce unrealistic linear targets.
- Remove constant and low-information sensors using training data only.
- Normalize sensor values by operating condition.
- Create rolling means, standard deviations, changes, and degradation slopes.
- Tune model hyperparameters using group-aware validation by engine ID.
- Evaluate a weighted ensemble only after the four individual models are frozen.

## Evaluation

The final comparison will report:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of determination (R²)
- NASA asymmetric score
- Training and inference time

Engine IDs, not individual sensor rows, define validation groups. This prevents readings from the same engine appearing in both training and validation data.

## Model 1: FD001 Ridge Regression Results

Model 1 is a scaled Ridge regressor that provides a fast, interpretable linear
baseline. Its improved version uses the same selected sensors, causal rolling
features, validation engines, and capped target as Models 2–4. Alpha was chosen
using validation results only; the expanded search selected `alpha=1000`.

| Experiment | RMSE | MAE | R² | NASA score |
|---|---:|---:|---:|---:|
| Ridge baseline validation | 31.11 | 25.98 | -0.481 | 1006.40 |
| Ridge improved validation | **16.38** | **14.11** | **0.589** | **83.36** |
| Ridge improved official FD001 test | **21.98** | **17.33** | **0.720** | **961.65** |

Feature engineering and regularization reduced validation RMSE by **14.73
cycles (47.3%)**. Standardized coefficients are saved in
[`results/metrics/ridge_fd001_coefficients.csv`](results/metrics/ridge_fd001_coefficients.csv)
for interpretation.

The submitted Logistic Regression classified whether maintenance was needed
within 30 cycles. It was not reused as Model 1 because that binary target and its
accuracy/F1 metrics are incompatible with continuous RUL prediction and the
shared regression leaderboard.

To train or reuse the cached Ridge artifacts:

```powershell
conda activate nti-cmapss
python scripts/train_ridge.py
```

## Model 2: FD001 Scratch Random Forest Results

Model 2 intentionally uses a hand-written NumPy Random Forest rather than
`sklearn.ensemble.RandomForestRegressor`. Its bootstrap sampling, feature
subsampling, CART regression trees, prediction averaging, and split-count
feature importance are implemented in
[`src/cmapss_rul/random_forest_scratch.py`](src/cmapss_rul/random_forest_scratch.py).

| Experiment | RMSE | MAE | R² | NASA score |
|---|---:|---:|---:|---:|
| Scratch baseline validation | 26.94 | 21.15 | -0.110 | 470.50 |
| Scratch improved validation | **18.13** | **13.46** | **0.497** | **114.68** |
| Scratch improved official FD001 test | **20.70** | **15.71** | **0.752** | **841.82** |

The shared engineered features reduced validation RMSE by **8.82 cycles
(32.7%)**. The implementation is CPU-only and trains more slowly than optimized
library implementations, but uses the same split, target, features, and metrics
as Models 3 and 4 for a fair comparison.

To train or reuse its cached artifacts:

```powershell
conda activate nti-cmapss
python scripts/train_random_forest.py
```

## Model 3: FD001 XGBoost Results

The executed [XGBoost notebook](notebooks/03_xgboost_fd001.ipynb) compares a raw-feature baseline with an improved pipeline using RUL capping, training-only sensor selection, rolling statistics, changes, and degradation slopes. Validation engines are kept separate and truncated before failure to simulate realistic test histories.

| Experiment | RMSE | MAE | R² | NASA score |
|---|---:|---:|---:|---:|
| Baseline validation | 24.26 | 19.13 | 0.099 | 291.01 |
| Improved validation | **17.07** | **13.16** | **0.555** | **99.64** |
| Improved official FD001 test | **19.19** | **14.56** | **0.787** | **644.80** |

The improved pipeline reduced validation RMSE by **7.20 cycles (29.7%)**. It trained with XGBoost 3.4.1 on the NVIDIA GPU (`device=cuda`); the code automatically falls back to CPU when CUDA is unavailable. The executed notebook contains Matplotlib and Seaborn plots for engine lifetimes, sensor degradation, model comparison, residuals, official predictions, and feature importance.

To reproduce the complete experiment from the repository root:

```powershell
conda activate nti-cmapss
jupyter nbconvert --to notebook --execute --inplace notebooks/03_xgboost_fd001.ipynb --ExecutePreprocessor.timeout=1800
```

## Model 4: FD001 CatBoost Results

The executed [CatBoost notebook](notebooks/04_catboost_fd001.ipynb) repeats the same engine-level split, validation cutoffs, RUL target, engineered features, and metrics used by Model 3. This isolates the effect of changing the learning algorithm.

| Experiment | RMSE | MAE | R² | NASA score |
|---|---:|---:|---:|---:|
| CatBoost baseline validation | 25.65 | 20.25 | -0.006 | 436.20 |
| CatBoost improved validation | **17.91** | **13.91** | **0.509** | **109.23** |
| CatBoost improved official FD001 test | **18.94** | **14.31** | **0.792** | **624.29** |

Feature engineering and regularization reduced CatBoost validation RMSE by **7.74 cycles (30.2%)**. CatBoost used the NVIDIA GPU and stopped at iteration 177. On the official FD001 test, CatBoost slightly outperformed XGBoost: **18.94 versus 19.19 RMSE**, with lower MAE and NASA score as well.

To reproduce Model 4:

```powershell
conda activate nti-cmapss
jupyter nbconvert --to notebook --execute --inplace notebooks/04_catboost_fd001.ipynb --ExecutePreprocessor.timeout=1800
```

## Train Once, Explore Locally

Each model has an independent training entry point. Training saves the fitted model, predictions, metrics, and plots so it only needs to run once:

```powershell
conda activate nti-cmapss
python scripts/train_ridge.py
python scripts/train_random_forest.py
python scripts/train_xgboost.py
python scripts/train_catboost.py
```

All training scripts reuse valid cached artifacts. Add `--force` only when you intentionally want to retrain:

```powershell
python scripts/train_ridge.py --force
python scripts/train_random_forest.py --force
python scripts/train_xgboost.py --force
python scripts/train_catboost.py --force
```

The root [`main.py`](main.py) is a local Streamlit dashboard. It reads saved artifacts for model comparison, diagnostics, and uploaded-engine inference. It never imports a training script or trains a model:

```powershell
conda activate nti-cmapss
streamlit run main.py
```

Open `http://127.0.0.1:8501` if the browser does not open automatically. The app is bound to your own computer only, is not deployed or published, and stops when its terminal process stops.

Model binaries live in `models/`. Metrics, predictions, and figures live under `results/`. Metadata in `models/metadata/` tells Streamlit how to connect those artifacts. All four FD001 model packages are complete and load without retraining in the local dashboard.

## Workflow

1. Validate and document all FD001–FD004 files.
2. Explore engine lifetimes, operating conditions, and sensor behavior.
3. Build a shared engine-level split and evaluation pipeline.
4. Establish baseline results on FD001.
5. Add degradation-trend features and tune all four models.
6. Test the frozen method on FD002, FD003, and FD004.
7. Analyze prediction errors and model robustness by scenario.
8. Produce the final report and select the best model.

## Repository Structure

```text
data/                  Local raw and processed data; contents ignored by Git
models/                Fitted model binaries and inference metadata
notebooks/             Exploratory and presentation notebooks
results/               Saved metrics, predictions, and figures
scripts/               Independent model-training entry points
src/cmapss_rul/         Python package for data, features, models, and evaluation
main.py                 Local Streamlit dashboard; inference only
environment.yml        Reproducible Conda environment
```

## Collaboration

Each task should have a GitHub Issue, a focused branch, and a pull request reviewed by another team member. Shared data, splitting, feature, and evaluation utilities must be merged before individual model experiments begin.

## Roadmap

- [x] Select the NASA C-MAPSS dataset
- [x] Define four candidate regression models
- [x] Download and validate all four scenarios locally
- [x] Rename the Python package for the new project
- [x] Add data-loading and schema-validation code
- [x] Add exploratory analysis for FD001
- [x] Add group-aware validation and common metrics
- [x] Train and improve all four FD001 models
- [ ] Compare scenarios and prepare the final report

## Scope and Limitations

C-MAPSS contains simulated run-to-failure trajectories rather than measurements from deployed commercial aircraft. Results demonstrate predictive-maintenance methodology but do not establish production readiness. Sensor descriptions and operating scenarios must be interpreted according to the original NASA documentation.

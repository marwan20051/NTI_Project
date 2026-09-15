# Turbofan Engine Remaining Useful Life Prediction

An NTI machine-learning project that predicts the Remaining Useful Life (RUL) of aircraft turbofan engines from multivariate sensor histories. The project compares four regression models and studies whether degradation-trend features improve predictions across increasingly difficult operating scenarios.

## Project Status

**Phase:** Dataset validation and experiment design

The repository foundation and local dataset are ready. Data-loading, exploratory-analysis, feature-engineering, and baseline-model milestones follow.

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
python scripts/train_xgboost.py
python scripts/train_catboost.py
```

Both training scripts reuse valid cached artifacts. Add `--force` only when you intentionally want to retrain:

```powershell
python scripts/train_xgboost.py --force
python scripts/train_catboost.py --force
```

The root [`main.py`](main.py) is a local Streamlit dashboard. It reads saved artifacts for model comparison, diagnostics, and uploaded-engine inference. It never imports a training script or trains a model:

```powershell
conda activate nti-cmapss
streamlit run main.py
```

Open `http://127.0.0.1:8501` if the browser does not open automatically. The app is bound to your own computer only, is not deployed or published, and stops when its terminal process stops.

Model binaries live in `models/`. Metrics, predictions, and figures live under `results/`. Metadata in `models/metadata/` tells Streamlit how to connect those artifacts. Ridge and Random Forest appear as waiting cards until teammates add complete packages using the same convention.

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
- [ ] Train and improve all four models
- [ ] Compare scenarios and prepare the final report

## Scope and Limitations

C-MAPSS contains simulated run-to-failure trajectories rather than measurements from deployed commercial aircraft. Results demonstrate predictive-maintenance methodology but do not establish production readiness. Sensor descriptions and operating scenarios must be interpreted according to the original NASA documentation.

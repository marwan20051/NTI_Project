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
data/                 Local raw and processed data; contents ignored by Git
models/               Generated model artifacts; contents ignored by Git
notebooks/            Exploratory and presentation notebooks
reports/figures/      Generated charts; contents ignored by Git
src/cmapss_rul/        Python package for data, features, models, and evaluation
tests/                Automated validation and leakage tests
environment.yml       Reproducible Conda environment
```

## Collaboration

Each task should have a GitHub Issue, a focused branch, and a pull request reviewed by another team member. Shared data, splitting, feature, and evaluation utilities must be merged before individual model experiments begin.

## Roadmap

- [x] Select the NASA C-MAPSS dataset
- [x] Define four candidate regression models
- [x] Download and validate all four scenarios locally
- [x] Rename the Python package for the new project
- [ ] Add data-loading and schema-validation code
- [ ] Add exploratory analysis
- [ ] Add group-aware validation and common metrics
- [ ] Train and improve all four models
- [ ] Compare scenarios and prepare the final report

## Scope and Limitations

C-MAPSS contains simulated run-to-failure trajectories rather than measurements from deployed commercial aircraft. Results demonstrate predictive-maintenance methodology but do not establish production readiness. Sensor descriptions and operating scenarios must be interpreted according to the original NASA documentation.

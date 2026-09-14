# FD001 XGBoost RUL Experiment Design

## Objective

Build and execute an original, educational notebook for model 3 of the project: an XGBoost regressor that predicts the remaining useful life (RUL) of NASA C-MAPSS FD001 turbofan engines. The notebook will follow the clear learning progression of the supplied assignment without copying its implementation.

The experiment must demonstrate whether an improved XGBoost pipeline performs better than a simple XGBoost baseline under a leakage-safe evaluation.

## Scope

- Scenario: FD001 only.
- Model family: XGBoost regression only.
- Comparisons: baseline XGBoost versus improved XGBoost.
- Execution: run locally and save notebook outputs.
- Hardware: attempt NVIDIA GPU acceleration and fall back to CPU automatically if the installed XGBoost build or CUDA runtime does not support the GPU.
- Dataset files remain local and ignored by Git.

Training the other three project models and evaluating FD002-FD004 are outside this task.

## Notebook Structure

The notebook will use short explanations and executable sections:

1. Experiment goal and reproducibility settings.
2. Imports and runtime/GPU information.
3. FD001 file loading and schema validation.
4. Exploratory summaries and selected sensor plots.
5. RUL target construction and early-life target capping.
6. Leakage-safe engine-level train/validation split.
7. Baseline XGBoost training and evaluation.
8. Feature engineering and improved XGBoost training.
9. Baseline-versus-improved comparison.
10. Final retraining and official FD001 test evaluation.
11. Feature importance, error analysis, conclusions, and saved artifacts.

## Data and Target

The input schema contains an engine identifier, cycle number, three operating settings, and 21 sensor readings. Whitespace-separated source files will receive explicit column names, and empty trailing columns will be rejected.

For each training row:

`RUL = maximum cycle for that engine - current cycle`

The improved experiment will cap training RUL at 125 cycles. This reduces the influence of the early period, when degradation is not yet observable and exact large RUL values are difficult to distinguish from sensor readings.

The official test labels contain one RUL value per test engine at its final observed cycle. Official test metrics will therefore use exactly the last observed row of every test engine.

## Leakage Prevention

All rows belonging to one engine must stay in the same partition. A deterministic engine-level split will create training and validation engine sets. Preprocessing decisions, including sensor removal and any scaling or feature selection, will be learned using training engines only.

The validation report will emphasize predictions at each validation engine's final observed cycle because this matches the official FD001 test protocol. Row-level validation metrics may be shown as diagnostics but will not replace last-cycle metrics.

## Baseline XGBoost

The baseline will use cycle number, operating settings, and non-constant raw sensors. It will use conservative fixed hyperparameters and no rolling or trend features. This establishes a transparent reference result.

## Improved XGBoost

The improved version will use the same engine split and target, then add:

- RUL capping at 125 cycles.
- Training-only removal of constant or near-constant sensor columns.
- Per-engine rolling means and standard deviations over short and medium windows.
- Per-engine sensor changes and rolling degradation slopes for selected informative sensors.
- Regularized XGBoost hyperparameters with early stopping.

Feature calculations will be causal: a row may use its current and previous cycles but never future sensor values.

## GPU Strategy

The notebook will make a small, explicit GPU capability attempt. If it succeeds, XGBoost will train using histogram-based GPU acceleration. If it fails because CUDA or a GPU-enabled XGBoost build is unavailable, the notebook will print the reason and continue with histogram-based CPU training. Evaluation and saved outputs will be identical in either mode.

## Metrics

Both versions will be compared using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of determination (R²)
- NASA asymmetric score
- Training time and prediction time

The NASA score penalizes late predictions more heavily than early predictions, reflecting the higher risk of overestimating an engine's remaining life.

## Visual Outputs

The executed notebook will include:

- Engine lifetime distribution.
- Example sensor degradation trajectories.
- Baseline and improved metric comparison.
- True-versus-predicted RUL plot.
- Residual distribution or residual-versus-prediction plot.
- Top XGBoost feature importances.

Generated figures will be saved under `reports/figures/`.

## Artifacts

- Executed notebook under `notebooks/`.
- Final XGBoost model under `models/`.
- Validation and official-test metrics under `reports/`.
- Official-test predictions under `reports/`.
- Reusable data, feature, metric, and modeling functions under `src/cmapss_rul/` where extracting them improves clarity.

Generated binary and data artifacts remain ignored when required by the repository policy. The notebook and reusable source code will be tracked.

## Error Handling

The workflow will fail with a clear message when FD001 files are missing, the schema is invalid, engine/RUL counts disagree, target values are invalid, or a split contains overlapping engine IDs. GPU unavailability is the only expected condition that triggers an automatic fallback instead of stopping execution.

## Verification

- Run schema and leakage checks before model fitting.
- Confirm train and validation engine identifiers are disjoint.
- Confirm engineered features contain no infinite values and handle initial rolling-window missing values deterministically.
- Execute the notebook from a clean kernel to completion.
- Confirm all expected metrics, predictions, figures, and the fitted model are produced.
- Re-run repository tests and Git whitespace checks before proposing the implementation for review.

## Success Criteria

The task is complete when the notebook executes locally without manual intervention, the official FD001 test metrics are reported, the improved pipeline is compared fairly with the baseline, GPU/CPU behavior is recorded, and all reproducible code and documentation are ready for team review. Improvement is measured and reported honestly; the notebook will not claim success if the improved model fails to outperform the baseline.

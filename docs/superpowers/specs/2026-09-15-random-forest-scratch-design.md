# FD001 Scratch Random Forest Design

## Goal

Turn the root `random forest.py` learning implementation into Model 2 of the
NASA C-MAPSS project while preserving its hand-written CART and bootstrap
forest algorithm. The result must be comparable with Models 3 and 4 and must
integrate with the local Streamlit dashboard without retraining in the app.

## Architecture

- `src/cmapss_rul/random_forest_scratch.py` contains the reusable estimator,
  tree nodes, fitting logic, prediction logic, and feature importance.
- `scripts/train_random_forest.py` owns data preparation, baseline and improved
  experiments, timing, evaluation, caching, serialization, and artifact output.
- The incompatible root `random forest.py` is removed after its useful logic is
  incorporated.

Keeping the estimator in the importable package ensures that a Joblib model
created by the training script can later be loaded by Streamlit.

## Data and Experiment Contract

The trainer uses the existing shared project utilities:

- FD001 files from `data/raw/cmapss/CMaps/`.
- Engine-level train/validation separation to prevent leakage.
- Truncated validation histories that simulate real prediction conditions.
- A capped RUL target of 125 cycles.
- The same raw-feature baseline and engineered-feature pipeline used by
  XGBoost and CatBoost.
- RMSE, MAE, R-squared, NASA score, training time, and prediction time.

The final fitted model trains on all FD001 training engines and is evaluated
once on the official FD001 test labels. Model selection uses validation RMSE,
not official-test performance.

## Scratch Forest Behavior

The implementation retains bootstrap aggregation and recursive CART regression
trees written with NumPy. Each estimator owns and resets its random generator
so repeated training is deterministic. Runtime is bounded with configurable
tree count, depth, minimum leaf size, feature subsampling, row subsampling, and
a limited number of candidate split thresholds. Training remains CPU-only.

The estimator exposes `fit`, `predict`, and `feature_importances_`-compatible
behavior needed by the trainer and dashboard. Invalid inputs and prediction
before fitting produce clear errors.

## Saved Artifacts

Training creates:

```text
models/random_forest_fd001.joblib
models/metadata/random_forest_fd001.json
results/metrics/random_forest_fd001_metrics.csv
results/predictions/random_forest_fd001_predictions.csv
results/figures/random_forest_fd001_run_summary.png
```

Metadata uses serializer `joblib` and records the exact feature columns,
selected sensors, rolling windows, RUL cap, and artifact paths. Once all files
pass the dashboard's validation, the Random Forest slot becomes ready
automatically.

## Caching and Failure Handling

The normal command reuses a complete, loadable artifact package. `--force`
explicitly retrains. Partial or malformed artifacts trigger a fresh training
run, while missing data produces a direct file error. Directories are created
only when needed.

## Verification

- Compile the estimator and trainer.
- Exercise deterministic fitting and prediction on a small synthetic dataset.
- Run the FD001 trainer locally and confirm all five artifacts are created.
- Run it a second time and confirm cached training is skipped.
- Verify the dashboard discovers three ready models and raises no AppTest
  exceptions.
- Confirm `main.py` contains no training call or training-script import.

## Limitations

The educational NumPy implementation will be slower and may be less accurate
than scikit-learn's optimized Random Forest. It does not use the GPU. These
limitations will be documented rather than hidden, because preserving the
scratch implementation is an explicit project choice.

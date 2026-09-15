# FD001 Ridge Regression Design

## Goal

Complete Model 1 as a continuous Remaining Useful Life regressor that can be
compared fairly with the scratch Random Forest, XGBoost, and CatBoost models.
The submitted Logistic Regression work informs the scaled linear-model and
coefficient-analysis approach, but its binary maintenance target is not used in
the four-model RUL leaderboard.

## Compatibility Decision

The supplied Logistic Regression predicts whether RUL is at most 30 cycles and
reports classification metrics. The project leaderboard predicts the number of
remaining cycles and reports RMSE, MAE, R-squared, and NASA score. Therefore,
Model 1 will use `sklearn.linear_model.Ridge`, not Logistic Regression.

The external Logistic Regression folder remains unchanged. Its pickle is not
loaded or copied because it implements a different task and serialized pickle
files should only be loaded when their origin is trusted and their contract is
compatible.

## Experiment

`scripts/train_ridge.py` will reuse the project's shared FD001 preparation:

- Engine-level train/validation separation.
- Realistically truncated validation histories.
- The same raw baseline features as Models 2–4.
- The same selected sensors and causal rolling features for the improved run.
- A 125-cycle cap for the improved target.
- Shared regression and NASA metrics.

Both baseline and improved models use a scikit-learn pipeline containing
`StandardScaler` followed by `Ridge`. The baseline uses raw features and a fixed
alpha of 1. The improved model evaluates alpha values `0.01`, `0.1`, `1`, `10`,
`100`, `1000`, `2000`, `5000`, and `10000` using validation RMSE, with NASA
score and MAE as tie-breakers. Values above 1000 ensure the selected alpha is
not merely the upper boundary of the search.
Official test data does not influence alpha selection.

After selecting alpha, the final pipeline is fitted on all FD001 training rows
with the shared engineered feature columns and capped target, then evaluated on
the official final-cycle test rows.

## Outputs

The trainer creates:

```text
models/ridge_fd001.joblib
models/metadata/ridge_fd001.json
results/metrics/ridge_fd001_metrics.csv
results/metrics/ridge_fd001_coefficients.csv
results/predictions/ridge_fd001_predictions.csv
results/figures/ridge_fd001_run_summary.png
```

The main artifact metadata uses serializer `joblib` and records the model path,
metrics path, prediction path, summary figure, selected sensors, exact feature
columns, rolling windows, RUL cap, and selected alpha. The coefficient CSV stores
the standardized Ridge coefficient for every final feature for interpretability.

When the required model, metadata, metrics, and prediction files validate,
Streamlit automatically changes Model 1 from waiting to ready and includes it in
the validation-ranked comparison. The dashboard only loads the saved pipeline
for inference and never fits it.

## Caching and Errors

The trainer validates the full cached package before reusing it. Missing,
malformed, or mismatched artifacts cause a clean retraining run. `--force`
explicitly rebuilds a valid package. Missing FD001 data and model-loading errors
produce direct failures rather than partial results.

## Verification

- Compile the trainer.
- Check that alpha selection uses validation results only.
- Execute the FD001 run and verify the model plus five result artifacts.
- Re-run and confirm cached training is skipped.
- Recompute official metrics from the saved prediction CSV.
- Load the Joblib pipeline and run inference for all 100 FD001 test engines.
- Run Streamlit AppTest and confirm `4/4 model packages ready` with zero
  exceptions.
- Confirm `main.py` contains no fitting call or training-script import.

## Limitations

Ridge is a linear model, so it cannot represent complex sensor interactions as
directly as the tree models. Its value is as a fast, interpretable regularized
baseline. Coefficients describe associations in standardized engineered
features; they are not causal effects.

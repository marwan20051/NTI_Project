"""Train FD001 Model 3 (XGBoost) and save reusable artifacts."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cmapss_rul.evaluation import regression_metrics  # noqa: E402
from cmapss_rul.modeling import (  # noqa: E402
    baseline_parameters,
    improved_parameters,
    resolve_device,
)
from cmapss_rul.training_utils import (  # noqa: E402
    RUL_CAP,
    baseline_data,
    cache_is_complete,
    engineered_final_data,
    engineered_validation_data,
    prediction_frame,
    prepare_experiment_data,
    save_training_plot,
)

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cmapss" / "CMaps"
ARTIFACTS = {
    "model": MODEL_DIR / "xgboost_fd001.json",
    "metrics": MODEL_DIR / "xgboost_fd001_metrics.csv",
    "predictions": MODEL_DIR / "xgboost_fd001_predictions.csv",
}
SUMMARY_FIGURE = MODEL_DIR / "xgboost_fd001_run_summary.png"


def _validate_saved_model(path: Path) -> None:
    model = xgb.XGBRegressor()
    model.load_model(path)


def train(force: bool = False) -> pd.DataFrame:
    """Train XGBoost unless a complete cache exists."""
    if cache_is_complete(
        ARTIFACTS,
        model_validator=_validate_saved_model,
        expected_model_prefix="XGBoost",
    ) and not force:
        print("XGBoost is already trained. Add --force to retrain it.")
        cached_metrics = pd.read_csv(ARTIFACTS["metrics"])
        if not SUMMARY_FIGURE.is_file():
            cached_predictions = pd.read_csv(ARTIFACTS["predictions"])
            save_training_plot(
                cached_metrics,
                cached_predictions,
                "XGBoost",
                "#2A9D8F",
                MODEL_DIR,
            )
        return cached_metrics

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (
        train_df,
        test_df,
        official_test,
        train_split,
        validation_history,
        validation_snapshots,
    ) = prepare_experiment_data(DATA_DIR)
    device = resolve_device(prefer_gpu=True)
    print(f"Training XGBoost on {device}...")

    baseline_train, baseline_valid = baseline_data(
        train_split,
        validation_snapshots,
    )
    baseline_model = xgb.XGBRegressor(**baseline_parameters(device))
    start = time.perf_counter()
    baseline_model.fit(baseline_train, train_split["rul"], verbose=False)
    baseline_train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    baseline_predictions = np.clip(baseline_model.predict(baseline_valid), 0, None)
    baseline_predict_seconds = time.perf_counter() - start
    baseline_metrics = regression_metrics(
        validation_snapshots["rul"],
        baseline_predictions,
    )

    sensors, train_features, validation_features, validation_last = (
        engineered_validation_data(train_split, validation_history)
    )
    print(f"Selected sensors: {sensors}")
    improved_model = xgb.XGBRegressor(**improved_parameters(device))
    start = time.perf_counter()
    improved_model.fit(
        train_features,
        train_split["rul"].clip(upper=RUL_CAP),
        eval_set=[
            (
                validation_features,
                validation_history["rul"].clip(upper=RUL_CAP),
            )
        ],
        verbose=False,
    )
    improved_train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    improved_predictions = np.clip(improved_model.predict(validation_last), 0, None)
    improved_predict_seconds = time.perf_counter() - start
    improved_metrics = regression_metrics(
        validation_snapshots["rul"],
        improved_predictions,
    )

    _, full_train_features, test_last = engineered_final_data(train_df, test_df)
    final_parameters = improved_parameters(device)
    final_parameters.pop("early_stopping_rounds")
    final_parameters["n_estimators"] = max(1, improved_model.best_iteration + 1)
    final_model = xgb.XGBRegressor(**final_parameters)
    start = time.perf_counter()
    final_model.fit(
        full_train_features,
        train_df["rul"].clip(upper=RUL_CAP),
        verbose=False,
    )
    final_train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    test_predictions = np.clip(final_model.predict(test_last), 0, None)
    test_predict_seconds = time.perf_counter() - start
    official_metrics = regression_metrics(official_test["rul"], test_predictions)

    metrics = pd.DataFrame(
        [
            {
                "model": "XGBoost baseline",
                "split": "validation",
                "device": device,
                "train_seconds": baseline_train_seconds,
                "predict_seconds": baseline_predict_seconds,
                **baseline_metrics,
            },
            {
                "model": "XGBoost improved",
                "split": "validation",
                "device": device,
                "train_seconds": improved_train_seconds,
                "predict_seconds": improved_predict_seconds,
                **improved_metrics,
            },
            {
                "model": "XGBoost improved",
                "split": "official_test",
                "device": device,
                "train_seconds": final_train_seconds,
                "predict_seconds": test_predict_seconds,
                **official_metrics,
            },
        ]
    )
    predictions = prediction_frame(official_test, test_predictions)
    metrics.to_csv(ARTIFACTS["metrics"], index=False)
    predictions.to_csv(ARTIFACTS["predictions"], index=False)
    final_model.save_model(ARTIFACTS["model"])
    save_training_plot(
        metrics,
        predictions,
        "XGBoost",
        "#2A9D8F",
        MODEL_DIR,
    )
    print(f"XGBoost complete. Official RMSE: {official_metrics['rmse']:.3f}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even when valid saved artifacts already exist.",
    )
    args = parser.parse_args(argv)
    train(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

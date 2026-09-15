"""Train FD001 Model 2 (scratch Random Forest) and save reusable artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cmapss_rul.evaluation import regression_metrics  # noqa: E402
from cmapss_rul.random_forest_scratch import RandomForestScratch  # noqa: E402
from cmapss_rul.training_utils import (  # noqa: E402
    RUL_CAP,
    WINDOWS,
    baseline_data,
    cache_is_complete,
    engineered_final_data,
    engineered_validation_data,
    prediction_frame,
    prepare_experiment_data,
    save_training_plot,
)

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cmapss" / "CMaps"
MODEL_DIR = PROJECT_ROOT / "models"
METRIC_DIR = PROJECT_ROOT / "results" / "metrics"
PREDICTION_DIR = PROJECT_ROOT / "results" / "predictions"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
METADATA_DIR = MODEL_DIR / "metadata"
ARTIFACTS = {
    "model": MODEL_DIR / "random_forest_fd001.joblib",
    "metrics": METRIC_DIR / "random_forest_fd001_metrics.csv",
    "predictions": PREDICTION_DIR / "random_forest_fd001_predictions.csv",
    "metadata": METADATA_DIR / "random_forest_fd001.json",
}
SUMMARY_FIGURE = FIGURE_DIR / "random_forest_fd001_run_summary.png"

BASELINE_PARAMETERS = {
    "n_estimators": 30,
    "max_depth": 10,
    "min_samples_leaf": 8,
    "max_features": 0.5,
    "max_samples": 0.7,
    "max_thresholds": 48,
    "random_state": 42,
    "verbose": True,
}
IMPROVED_PARAMETERS = {
    "n_estimators": 60,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "max_features": 0.5,
    "max_samples": 0.8,
    "max_thresholds": 64,
    "random_state": 42,
    "verbose": True,
}


def _validate_saved_model(path: Path) -> None:
    model = joblib.load(path)
    if not isinstance(model, RandomForestScratch) or not model.trees_:
        raise ValueError("Saved artifact is not a fitted RandomForestScratch")


def _save_metadata(
    feature_columns: list[str],
    selected_sensors: list[str],
) -> None:
    metadata = {
        "id": "random_forest",
        "display_name": "Random Forest",
        "dataset": "FD001",
        "serializer": "joblib",
        "model_file": "models/random_forest_fd001.joblib",
        "metrics_file": "results/metrics/random_forest_fd001_metrics.csv",
        "predictions_file": (
            "results/predictions/random_forest_fd001_predictions.csv"
        ),
        "summary_figure": (
            "results/figures/random_forest_fd001_run_summary.png"
        ),
        "selected_sensors": selected_sensors,
        "feature_columns": feature_columns,
        "windows": list(WINDOWS),
        "rul_cap": RUL_CAP,
    }
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS["metadata"].write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _timed_fit(
    model: RandomForestScratch,
    features: pd.DataFrame,
    target: pd.Series,
) -> float:
    started = time.perf_counter()
    model.fit(features, target)
    return time.perf_counter() - started


def _timed_predict(
    model: RandomForestScratch,
    features: pd.DataFrame,
    upper_limit: float | None = RUL_CAP,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    predictions = np.clip(model.predict(features), 0.0, upper_limit)
    return predictions, time.perf_counter() - started


def train(force: bool = False) -> pd.DataFrame:
    """Train the scratch Random Forest unless a complete cache exists."""
    if cache_is_complete(
        ARTIFACTS,
        model_validator=_validate_saved_model,
        expected_model_prefix="Random Forest",
    ) and not force:
        print("Random Forest is already trained. Add --force to retrain it.")
        cached_metrics = pd.read_csv(ARTIFACTS["metrics"])
        if not SUMMARY_FIGURE.is_file():
            cached_predictions = pd.read_csv(ARTIFACTS["predictions"])
            save_training_plot(
                cached_metrics,
                cached_predictions,
                "Random Forest",
                "#34D399",
                FIGURE_DIR,
            )
        return cached_metrics

    for directory in (
        MODEL_DIR,
        METRIC_DIR,
        PREDICTION_DIR,
        FIGURE_DIR,
        METADATA_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    (
        train_df,
        test_df,
        official_test,
        train_split,
        validation_history,
        validation_snapshots,
    ) = prepare_experiment_data(DATA_DIR)
    print("Training scratch Random Forest on CPU...")

    baseline_train, baseline_valid = baseline_data(
        train_split,
        validation_snapshots,
    )
    print("Baseline model:")
    baseline_model = RandomForestScratch(**BASELINE_PARAMETERS)
    baseline_train_seconds = _timed_fit(
        baseline_model,
        baseline_train,
        train_split["rul"],
    )
    baseline_predictions, baseline_predict_seconds = _timed_predict(
        baseline_model,
        baseline_valid,
        upper_limit=None,
    )
    baseline_metrics = regression_metrics(
        validation_snapshots["rul"],
        baseline_predictions,
    )

    sensors, train_features, _validation_features, validation_last = (
        engineered_validation_data(train_split, validation_history)
    )
    print(f"Selected sensors: {sensors}")
    print("Improved model:")
    improved_model = RandomForestScratch(**IMPROVED_PARAMETERS)
    improved_train_seconds = _timed_fit(
        improved_model,
        train_features,
        train_split["rul"].clip(upper=RUL_CAP),
    )
    improved_predictions, improved_predict_seconds = _timed_predict(
        improved_model,
        validation_last,
    )
    improved_metrics = regression_metrics(
        validation_snapshots["rul"],
        improved_predictions,
    )

    final_sensors, full_train_features, test_last = engineered_final_data(
        train_df,
        test_df,
    )
    print("Final model:")
    final_model = RandomForestScratch(**IMPROVED_PARAMETERS)
    final_train_seconds = _timed_fit(
        final_model,
        full_train_features,
        train_df["rul"].clip(upper=RUL_CAP),
    )
    test_predictions, test_predict_seconds = _timed_predict(
        final_model,
        test_last,
    )
    official_metrics = regression_metrics(
        official_test["rul"],
        test_predictions,
    )

    metrics = pd.DataFrame(
        [
            {
                "model": "Random Forest baseline",
                "split": "validation",
                "device": "CPU (NumPy scratch)",
                "train_seconds": baseline_train_seconds,
                "predict_seconds": baseline_predict_seconds,
                **baseline_metrics,
            },
            {
                "model": "Random Forest improved",
                "split": "validation",
                "device": "CPU (NumPy scratch)",
                "train_seconds": improved_train_seconds,
                "predict_seconds": improved_predict_seconds,
                **improved_metrics,
            },
            {
                "model": "Random Forest improved",
                "split": "official_test",
                "device": "CPU (NumPy scratch)",
                "train_seconds": final_train_seconds,
                "predict_seconds": test_predict_seconds,
                **official_metrics,
            },
        ]
    )
    predictions = prediction_frame(official_test, test_predictions)

    metrics.to_csv(ARTIFACTS["metrics"], index=False)
    predictions.to_csv(ARTIFACTS["predictions"], index=False)
    joblib.dump(final_model, ARTIFACTS["model"])
    _save_metadata(list(full_train_features.columns), final_sensors)
    save_training_plot(
        metrics,
        predictions,
        "Random Forest",
        "#34D399",
        FIGURE_DIR,
    )
    print(
        "Random Forest complete. "
        f"Validation RMSE: {improved_metrics['rmse']:.3f}; "
        f"official RMSE: {official_metrics['rmse']:.3f}"
    )
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

"""Train FD001 Model 1 (Ridge regression) and save reusable artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cmapss_rul.evaluation import regression_metrics  # noqa: E402
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
    "model": MODEL_DIR / "ridge_fd001.joblib",
    "metrics": METRIC_DIR / "ridge_fd001_metrics.csv",
    "predictions": PREDICTION_DIR / "ridge_fd001_predictions.csv",
    "metadata": METADATA_DIR / "ridge_fd001.json",
}
COEFFICIENTS_FILE = METRIC_DIR / "ridge_fd001_coefficients.csv"
SUMMARY_FIGURE = FIGURE_DIR / "ridge_fd001_run_summary.png"
ALPHA_CANDIDATES = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def _ridge_pipeline(alpha: float) -> Pipeline:
    """Create a scaled Ridge pipeline with the requested regularization."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge alpha must be a positive finite number")
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def _validate_saved_model(path: Path) -> None:
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise ValueError("Saved Ridge artifact is not a scikit-learn Pipeline")
    if not isinstance(model.named_steps.get("scaler"), StandardScaler):
        raise ValueError("Saved Ridge pipeline is missing StandardScaler")
    if not isinstance(model.named_steps.get("ridge"), Ridge):
        raise ValueError("Saved Ridge pipeline is missing Ridge")
    if not hasattr(model.named_steps["ridge"], "coef_"):
        raise ValueError("Saved Ridge pipeline is not fitted")


def _extra_cache_files_valid() -> bool:
    if not all(
        path.is_file() and path.stat().st_size > 0
        for path in (COEFFICIENTS_FILE, SUMMARY_FIGURE)
    ):
        return False
    try:
        coefficients = pd.read_csv(COEFFICIENTS_FILE)
    except Exception:
        return False
    return bool(
        not coefficients.empty
        and {"feature", "coefficient", "absolute_coefficient"}.issubset(
            coefficients.columns
        )
    )


def _save_metadata(
    feature_columns: list[str],
    selected_sensors: list[str],
    selected_alpha: float,
) -> None:
    metadata = {
        "id": "ridge",
        "display_name": "Ridge Regression",
        "dataset": "FD001",
        "serializer": "joblib",
        "model_file": "models/ridge_fd001.joblib",
        "metrics_file": "results/metrics/ridge_fd001_metrics.csv",
        "predictions_file": "results/predictions/ridge_fd001_predictions.csv",
        "summary_figure": "results/figures/ridge_fd001_run_summary.png",
        "selected_sensors": selected_sensors,
        "feature_columns": feature_columns,
        "windows": list(WINDOWS),
        "rul_cap": RUL_CAP,
        "selected_alpha": selected_alpha,
    }
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS["metadata"].write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _timed_fit(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> float:
    started = time.perf_counter()
    model.fit(features, target)
    return time.perf_counter() - started


def _timed_predict(
    model: Pipeline,
    features: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    predictions = np.clip(model.predict(features), 0.0, RUL_CAP)
    return predictions, time.perf_counter() - started


def _save_coefficients(model: Pipeline, feature_columns: list[str]) -> None:
    ridge = model.named_steps["ridge"]
    coefficients = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": np.asarray(ridge.coef_, dtype=float),
        }
    )
    coefficients["absolute_coefficient"] = coefficients["coefficient"].abs()
    coefficients.sort_values(
        "absolute_coefficient",
        ascending=False,
        ignore_index=True,
    ).to_csv(COEFFICIENTS_FILE, index=False)


def train(force: bool = False) -> pd.DataFrame:
    """Train Ridge unless a complete cached experiment already exists."""
    cache_valid = cache_is_complete(
        ARTIFACTS,
        model_validator=_validate_saved_model,
        expected_model_prefix="Ridge",
    ) and _extra_cache_files_valid()
    if cache_valid and not force:
        print("Ridge is already trained. Add --force to retrain it.")
        return pd.read_csv(ARTIFACTS["metrics"])

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
    print("Training Ridge baseline on CPU...")

    baseline_train, baseline_valid = baseline_data(
        train_split,
        validation_snapshots,
    )
    baseline_model = _ridge_pipeline(alpha=1.0)
    baseline_train_seconds = _timed_fit(
        baseline_model,
        baseline_train,
        train_split["rul"],
    )
    baseline_predictions, baseline_predict_seconds = _timed_predict(
        baseline_model,
        baseline_valid,
    )
    baseline_metrics = regression_metrics(
        validation_snapshots["rul"],
        baseline_predictions,
    )

    sensors, train_features, _validation_features, validation_last = (
        engineered_validation_data(train_split, validation_history)
    )
    print(f"Selected sensors: {sensors}")
    print("Selecting Ridge alpha using validation RMSE...")
    best_result: dict[str, object] | None = None
    for alpha in ALPHA_CANDIDATES:
        candidate = _ridge_pipeline(alpha)
        train_seconds = _timed_fit(
            candidate,
            train_features,
            train_split["rul"].clip(upper=RUL_CAP),
        )
        predictions, predict_seconds = _timed_predict(
            candidate,
            validation_last,
        )
        candidate_metrics = regression_metrics(
            validation_snapshots["rul"],
            predictions,
        )
        print(
            f"  alpha={alpha:g}: RMSE={candidate_metrics['rmse']:.3f}, "
            f"MAE={candidate_metrics['mae']:.3f}, "
            f"NASA={candidate_metrics['nasa_score']:.3f}"
        )
        sort_key = (
            candidate_metrics["rmse"],
            candidate_metrics["nasa_score"],
            candidate_metrics["mae"],
        )
        if best_result is None or sort_key < best_result["sort_key"]:
            best_result = {
                "alpha": alpha,
                "model": candidate,
                "metrics": candidate_metrics,
                "train_seconds": train_seconds,
                "predict_seconds": predict_seconds,
                "sort_key": sort_key,
            }

    if best_result is None:
        raise RuntimeError("Ridge alpha selection produced no candidate")
    selected_alpha = float(best_result["alpha"])
    improved_metrics = best_result["metrics"]
    print(f"Selected alpha: {selected_alpha:g}")

    final_sensors, full_train_features, test_last = engineered_final_data(
        train_df,
        test_df,
    )
    final_model = _ridge_pipeline(selected_alpha)
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
    improved_name = f"Ridge improved (alpha={selected_alpha:g})"
    metrics = pd.DataFrame(
        [
            {
                "model": "Ridge baseline",
                "split": "validation",
                "device": "CPU",
                "train_seconds": baseline_train_seconds,
                "predict_seconds": baseline_predict_seconds,
                **baseline_metrics,
            },
            {
                "model": improved_name,
                "split": "validation",
                "device": "CPU",
                "train_seconds": float(best_result["train_seconds"]),
                "predict_seconds": float(best_result["predict_seconds"]),
                **improved_metrics,
            },
            {
                "model": improved_name,
                "split": "official_test",
                "device": "CPU",
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
    _save_coefficients(final_model, list(full_train_features.columns))
    _save_metadata(
        list(full_train_features.columns),
        final_sensors,
        selected_alpha,
    )
    save_training_plot(
        metrics,
        predictions,
        "Ridge",
        "#38BDF8",
        FIGURE_DIR,
    )
    print(
        "Ridge complete. "
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

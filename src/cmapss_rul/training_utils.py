"""Shared preparation, caching, and plotting helpers for model trainers."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cmapss_rul.data import load_fd001, make_validation_subset, split_by_engine
from cmapss_rul.features import (
    engineer_features,
    nonconstant_sensor_columns,
    select_sensors,
)

RANDOM_STATE = 42
RUL_CAP = 125
WINDOWS = (5, 15)
REQUIRED_METRIC_COLUMNS = {
    "model",
    "split",
    "device",
    "train_seconds",
    "predict_seconds",
    "rmse",
    "mae",
    "r2",
    "nasa_score",
}


def cache_is_complete(
    artifacts: dict[str, Path],
    model_validator: Callable[[Path], None],
    expected_model_prefix: str,
    expected_prediction_rows: int = 100,
) -> bool:
    """Return whether model, metrics, and prediction artifacts are valid."""
    if not all(
        path.is_file() and path.stat().st_size > 0
        for path in artifacts.values()
    ):
        return False
    try:
        metrics = pd.read_csv(artifacts["metrics"])
        predictions = pd.read_csv(artifacts["predictions"])
        metadata = json.loads(artifacts["metadata"].read_text(encoding="utf-8"))
        model_validator(artifacts["model"])
    except Exception:
        return False

    if not isinstance(metadata, dict):
        return False
    required_metadata = {
        "id",
        "serializer",
        "model_file",
        "metrics_file",
        "predictions_file",
        "selected_sensors",
        "windows",
        "rul_cap",
    }
    if not required_metadata.issubset(metadata):
        return False
    if not isinstance(metadata["id"], str) or not metadata["id"]:
        return False
    if metadata["serializer"] not in {"xgboost", "catboost", "joblib"}:
        return False
    if not (
        isinstance(metadata["selected_sensors"], list)
        and metadata["selected_sensors"]
        and all(
            isinstance(sensor, str) and sensor
            for sensor in metadata["selected_sensors"]
        )
    ):
        return False
    if not (
        isinstance(metadata["windows"], list)
        and metadata["windows"]
        and all(
            isinstance(window, int) and window > 0
            for window in metadata["windows"]
        )
    ):
        return False
    try:
        rul_cap = float(metadata["rul_cap"])
    except (TypeError, ValueError):
        return False
    if not np.isfinite(rul_cap) or rul_cap <= 0:
        return False
    feature_columns = metadata.get("feature_columns")
    if feature_columns is not None and not (
        isinstance(feature_columns, list)
        and feature_columns
        and all(isinstance(column, str) and column for column in feature_columns)
    ):
        return False

    metadata_path = artifacts["metadata"].resolve()
    try:
        project_root = metadata_path.parents[2]
    except IndexError:
        return False
    metadata_artifacts = {
        "model": "model_file",
        "metrics": "metrics_file",
        "predictions": "predictions_file",
    }
    for artifact_key, metadata_key in metadata_artifacts.items():
        relative_path = metadata[metadata_key]
        if not isinstance(relative_path, str) or not relative_path:
            return False
        declared_path = Path(relative_path)
        if declared_path.is_absolute():
            return False
        if (project_root / declared_path).resolve() != artifacts[
            artifact_key
        ].resolve():
            return False

    numeric_metric_columns = [
        "train_seconds",
        "predict_seconds",
        "rmse",
        "mae",
        "r2",
        "nasa_score",
    ]
    metrics_valid = (
        not metrics.empty
        and REQUIRED_METRIC_COLUMNS.issubset(metrics.columns)
        and {"validation", "official_test"}.issubset(set(metrics["split"]))
        and metrics["model"].astype(str).str.startswith(expected_model_prefix).all()
    )
    if metrics_valid:
        numeric_metrics = metrics[numeric_metric_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        metrics_valid = bool(np.isfinite(numeric_metrics.to_numpy()).all())

    prediction_columns = {"unit_id", "true_rul", "predicted_rul", "residual"}
    predictions_valid = (
        len(predictions) == expected_prediction_rows
        and prediction_columns.issubset(predictions.columns)
        and predictions["unit_id"].is_unique
    )
    if predictions_valid:
        numeric_predictions = predictions[list(prediction_columns)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        predictions_valid = bool(
            np.isfinite(numeric_predictions.to_numpy()).all()
        )
    return metrics_valid and predictions_valid


def prepare_experiment_data(data_dir: Path):
    """Load FD001 and create the shared leakage-safe validation split."""
    train_df, test_df, official_test = load_fd001(data_dir)
    train_split, validation_full = split_by_engine(
        train_df,
        validation_size=0.20,
        random_state=RANDOM_STATE,
    )
    validation_history, validation_snapshots = make_validation_subset(
        validation_full,
        min_rul=10,
        max_rul=100,
        random_state=RANDOM_STATE,
    )
    return (
        train_df,
        test_df,
        official_test,
        train_split,
        validation_history,
        validation_snapshots,
    )


def baseline_data(
    train_split: pd.DataFrame,
    validation_snapshots: pd.DataFrame,
):
    raw_sensors = nonconstant_sensor_columns(train_split)
    columns = ["cycle", "setting_1", "setting_2", "setting_3", *raw_sensors]
    return train_split[columns], validation_snapshots[columns]


def engineered_validation_data(
    train_split: pd.DataFrame,
    validation_history: pd.DataFrame,
):
    selection_frame = train_split.assign(
        rul=train_split["rul"].clip(upper=RUL_CAP)
    )
    selected_sensors = select_sensors(selection_frame, top_n=8)
    train_features = engineer_features(train_split, selected_sensors, WINDOWS)
    validation_features = engineer_features(
        validation_history,
        selected_sensors,
        WINDOWS,
    )
    last_indices = validation_history.groupby("unit_id")["cycle"].idxmax()
    validation_last = validation_features.loc[last_indices].reset_index(drop=True)
    return selected_sensors, train_features, validation_features, validation_last


def engineered_final_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    selection_frame = train_df.assign(rul=train_df["rul"].clip(upper=RUL_CAP))
    sensors = select_sensors(selection_frame, top_n=8)
    train_features = engineer_features(train_df, sensors, WINDOWS)
    test_features = engineer_features(test_df, sensors, WINDOWS)
    last_indices = test_df.groupby("unit_id")["cycle"].idxmax()
    test_last = test_features.loc[last_indices].reset_index(drop=True)
    return sensors, train_features, test_last


def prediction_frame(
    official_test: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "unit_id": official_test["unit_id"].astype(int),
            "true_rul": official_test["rul"].to_numpy(),
            "predicted_rul": np.clip(predictions, 0.0, None),
        }
    )
    frame["residual"] = frame["predicted_rul"] - frame["true_rul"]
    return frame


def save_training_plot(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    algorithm: str,
    color: str,
    figure_dir: Path,
) -> Path:
    """Save a compact visual summary for a freshly trained model."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    validation = metrics.loc[metrics["split"].eq("validation")]
    plot_metrics = validation.melt(
        id_vars="model",
        value_vars=["rmse", "mae"],
        var_name="metric",
        value_name="cycles",
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(
        data=plot_metrics,
        x="metric",
        y="cycles",
        hue="model",
        ax=axes[0],
    )
    axes[0].set(
        title=f"{algorithm}: validation error",
        xlabel="",
        ylabel="Cycles (lower is better)",
    )
    limit = max(
        predictions["true_rul"].max(),
        predictions["predicted_rul"].max(),
    ) + 5
    sns.scatterplot(
        data=predictions,
        x="true_rul",
        y="predicted_rul",
        color=color,
        s=55,
        ax=axes[1],
    )
    axes[1].plot([0, limit], [0, limit], "--", color="black")
    axes[1].set(
        title=f"{algorithm}: official FD001 predictions",
        xlabel="True RUL",
        ylabel="Predicted RUL",
        xlim=(0, limit),
        ylim=(0, limit),
    )
    figure.tight_layout()
    file_stem = algorithm.lower().replace(" ", "_")
    path = figure_dir / f"{file_stem}_fd001_run_summary.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path

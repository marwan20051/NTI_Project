"""Training-only selection and causal degradation features for C-MAPSS."""

from collections.abc import Iterable

import numpy as np
import pandas as pd


def select_sensors(
    train_frame: pd.DataFrame,
    top_n: int = 8,
    variance_floor: float = 1e-10,
) -> list[str]:
    """Select variable sensors most associated with RUL using training data."""
    sensors = [
        column for column in train_frame.columns if column.startswith("sensor_")
    ]
    variable = [
        column
        for column in sensors
        if train_frame[column].var() > variance_floor
    ]
    correlations = train_frame[variable].corrwith(train_frame["rul"]).abs()
    return correlations.sort_values(ascending=False).head(top_n).index.tolist()


def nonconstant_sensor_columns(
    train_frame: pd.DataFrame,
    variance_floor: float = 1e-10,
) -> list[str]:
    """Return raw sensor columns with useful variation in training engines."""
    return [
        column
        for column in train_frame.columns
        if column.startswith("sensor_")
        and train_frame[column].var() > variance_floor
    ]


def _rolling_slope(values: np.ndarray) -> float:
    """Calculate the least-squares slope over one trailing sensor window."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    centered_x = x - x.mean()
    denominator = np.square(centered_x).sum()
    if denominator == 0:
        return 0.0
    return float((centered_x * (values - values.mean())).sum() / denominator)


def engineer_features(
    frame: pd.DataFrame,
    selected_sensors: list[str],
    windows: Iterable[int] = (5, 15),
) -> pd.DataFrame:
    """Create causal rolling features independently within each engine."""
    missing = set(selected_sensors).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing selected sensors: {sorted(missing)}")

    ordered = frame.sort_values(["unit_id", "cycle"]).copy()
    base_columns = [
        "cycle",
        "setting_1",
        "setting_2",
        "setting_3",
        *selected_sensors,
    ]
    output = ordered[base_columns].copy()
    grouped = ordered.groupby("unit_id", sort=False)

    for sensor in selected_sensors:
        output[f"{sensor}_diff_1"] = grouped[sensor].diff().fillna(0.0)
        for window in windows:
            output[f"{sensor}_roll_mean_{window}"] = grouped[sensor].transform(
                lambda series: series.rolling(window, min_periods=1).mean()
            )
            output[f"{sensor}_roll_std_{window}"] = grouped[sensor].transform(
                lambda series: series.rolling(window, min_periods=1).std(ddof=0)
            )
            output[f"{sensor}_slope_{window}"] = grouped[sensor].transform(
                lambda series: series.rolling(window, min_periods=1).apply(
                    _rolling_slope,
                    raw=True,
                )
            )

    output = output.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return output.loc[frame.index]

"""Local Streamlit dashboard for cached NASA C-MAPSS FD001 models."""

from __future__ import annotations

import io
import json
import sys
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cmapss_rul.features import engineer_features  # noqa: E402
from cmapss_rul.schema import CMAPSS_COLUMNS  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cmapss" / "CMaps"
MODEL_DIR = PROJECT_ROOT / "models"
METADATA_DIR = MODEL_DIR / "metadata"
METRIC_DIR = PROJECT_ROOT / "results" / "metrics"
PREDICTION_DIR = PROJECT_ROOT / "results" / "predictions"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

MODEL_SLOTS = [
    {
        "id": "ridge",
        "name": "Ridge Regression",
        "number": 1,
        "color": "#38BDF8",
        "description": "Regularized linear baseline",
    },
    {
        "id": "random_forest",
        "name": "Random Forest",
        "number": 2,
        "color": "#34D399",
        "description": "Bagged nonlinear trees",
    },
    {
        "id": "xgboost",
        "name": "XGBoost",
        "number": 3,
        "color": "#F59E0B",
        "description": "Gradient-boosted decision trees",
    },
    {
        "id": "catboost",
        "name": "CatBoost",
        "number": 4,
        "color": "#A78BFA",
        "description": "Ordered gradient boosting",
    },
]

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
REQUIRED_PREDICTION_COLUMNS = {
    "unit_id",
    "true_rul",
    "predicted_rul",
    "residual",
}
SUPPORTED_SERIALIZERS = {"xgboost", "catboost", "joblib"}


def _file_signature(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


@st.cache_data(show_spinner=False)
def read_json(signature: tuple[str, int, int]) -> dict[str, Any]:
    path = Path(signature[0])
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_csv(signature: tuple[str, int, int]) -> pd.DataFrame:
    return pd.read_csv(signature[0])


def _artifact_path(relative_path: Any) -> Path:
    """Resolve a metadata artifact path without allowing it outside the project."""
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("artifact path must be a non-empty relative string")
    artifact = Path(relative_path)
    if artifact.is_absolute():
        raise ValueError("artifact path must be relative to the project")
    candidate = (PROJECT_ROOT / artifact).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError("artifact path points outside the project") from error
    return candidate


def _validate_package(
    metadata: dict[str, Any],
    paths: dict[str, Path],
) -> list[str]:
    """Validate an inference package before exposing it as dashboard-ready."""
    problems: list[str] = []
    serializer = metadata.get("serializer")
    if serializer not in SUPPORTED_SERIALIZERS:
        problems.append(f"unsupported serializer: {serializer!r}")

    feature_columns = metadata.get("feature_columns")
    sensors = metadata.get("selected_sensors")
    windows = metadata.get("windows")
    if feature_columns is not None and not (
        isinstance(feature_columns, list)
        and bool(feature_columns)
        and all(isinstance(column, str) and column for column in feature_columns)
    ):
        problems.append("feature_columns must be a non-empty list of names")
    has_feature_recipe = (
        isinstance(sensors, list)
        and bool(sensors)
        and all(isinstance(sensor, str) and sensor for sensor in sensors)
        and isinstance(windows, list)
        and bool(windows)
        and all(isinstance(window, int) and window > 0 for window in windows)
    )
    if not has_feature_recipe:
        problems.append("missing a valid selected_sensors/windows feature recipe")

    if paths["model_file"].stat().st_size == 0:
        problems.append("model file is empty")

    rul_cap = metadata.get("rul_cap", 125)
    try:
        rul_cap = float(rul_cap)
        if not np.isfinite(rul_cap) or rul_cap <= 0:
            raise ValueError
    except (TypeError, ValueError):
        problems.append("rul_cap must be a positive finite number")

    try:
        metrics = read_csv(_file_signature(paths["metrics_file"]))
        missing_columns = REQUIRED_METRIC_COLUMNS.difference(metrics.columns)
        if missing_columns:
            problems.append(
                "metrics missing columns: " + ", ".join(sorted(missing_columns))
            )
        elif metrics.empty or not metrics["split"].eq("validation").any():
            problems.append("metrics must contain at least one validation row")
        else:
            numeric_columns = [
                "train_seconds",
                "predict_seconds",
                "rmse",
                "mae",
                "r2",
                "nasa_score",
            ]
            numeric = metrics[numeric_columns].apply(pd.to_numeric, errors="coerce")
            if not np.isfinite(numeric.to_numpy(dtype=float)).all():
                problems.append("metrics contain missing or nonnumeric values")
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        problems.append(f"invalid metrics file: {error}")

    try:
        predictions = read_csv(_file_signature(paths["predictions_file"]))
        missing_columns = REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns)
        if missing_columns:
            problems.append(
                "predictions missing columns: " + ", ".join(sorted(missing_columns))
            )
        elif predictions.empty:
            problems.append("predictions file is empty")
        else:
            numeric = predictions[list(REQUIRED_PREDICTION_COLUMNS)].apply(
                pd.to_numeric,
                errors="coerce",
            )
            if not np.isfinite(numeric.to_numpy(dtype=float)).all():
                problems.append("predictions contain missing or nonnumeric values")
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        problems.append(f"invalid predictions file: {error}")

    return problems


def discover_models() -> list[dict[str, Any]]:
    """Return all four dashboard slots and their artifact availability."""
    registry: list[dict[str, Any]] = []
    for slot in MODEL_SLOTS:
        entry = dict(slot)
        metadata_path = METADATA_DIR / f"{slot['id']}_fd001.json"
        entry["metadata_path"] = metadata_path
        entry["metadata"] = None
        entry["missing"] = [metadata_path.name]
        entry["available"] = False

        if metadata_path.is_file():
            try:
                metadata = read_json(_file_signature(metadata_path))
                if not isinstance(metadata, dict):
                    raise ValueError("metadata root must be a JSON object")
                required = ("model_file", "metrics_file", "predictions_file")
                paths = {
                    key: _artifact_path(metadata[key])
                    for key in required
                    if key in metadata
                }
                missing = [
                    f"missing {key}"
                    for key in required
                    if key not in paths or not paths[key].is_file()
                ]
                if not missing:
                    missing.extend(_validate_package(metadata, paths))
                entry.update(metadata)
                entry["paths"] = paths
                entry["metadata"] = metadata
                entry["missing"] = missing
                entry["available"] = not missing
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                json.JSONDecodeError,
            ) as error:
                entry["missing"] = [f"invalid metadata: {error}"]
        registry.append(entry)
    return registry


def _best_rows(metrics: pd.DataFrame) -> tuple[pd.Series, pd.Series | None]:
    validation = metrics.loc[metrics["split"].eq("validation")].sort_values(
        ["rmse", "nasa_score", "mae"]
    )
    if validation.empty:
        raise ValueError("No validation result is available")
    best_validation = validation.iloc[0]
    official = metrics.loc[
        metrics["split"].eq("official_test")
        & metrics["model"].eq(best_validation["model"])
    ].sort_values(["rmse", "nasa_score", "mae"])
    return best_validation, None if official.empty else official.iloc[0]


def build_comparison(registry: list[dict[str, Any]]) -> pd.DataFrame:
    """Rank every available model using saved validation metrics."""
    rows: list[dict[str, Any]] = []
    for entry in registry:
        if not entry["available"]:
            continue
        metrics = read_csv(_file_signature(entry["paths"]["metrics_file"]))
        missing = REQUIRED_METRIC_COLUMNS.difference(metrics.columns)
        if missing:
            continue
        validation, official = _best_rows(metrics)
        row = {
            "model_id": entry["id"],
            "model": entry["name"],
            "variant": validation["model"],
            "device": validation["device"],
            "validation_rmse": float(validation["rmse"]),
            "validation_mae": float(validation["mae"]),
            "validation_r2": float(validation["r2"]),
            "validation_nasa": float(validation["nasa_score"]),
            "train_seconds": float(validation["train_seconds"]),
            "predict_seconds": float(validation["predict_seconds"]),
            "test_rmse": np.nan,
            "test_mae": np.nan,
            "test_r2": np.nan,
            "test_nasa": np.nan,
        }
        if official is not None:
            row.update(
                {
                    "test_rmse": float(official["rmse"]),
                    "test_mae": float(official["mae"]),
                    "test_r2": float(official["r2"]),
                    "test_nasa": float(official["nasa_score"]),
                }
            )
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    comparison = pd.DataFrame(rows).sort_values(
        ["validation_rmse", "validation_nasa", "validation_mae"],
        ignore_index=True,
    )
    comparison.insert(0, "rank", np.arange(1, len(comparison) + 1))
    return comparison


@st.cache_data(show_spinner=False)
def dataset_summary(signatures: tuple[tuple[str, int, int], ...]) -> dict[str, int]:
    train_path = Path(signatures[0][0])
    test_path = Path(signatures[1][0])
    train = pd.read_csv(train_path, sep=r"\s+", header=None, usecols=[0, 1])
    test = pd.read_csv(test_path, sep=r"\s+", header=None, usecols=[0, 1])
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_engines": int(train[0].nunique()),
        "test_engines": int(test[0].nunique()),
        "max_cycle": int(train[1].max()),
    }


@st.cache_resource(show_spinner="Loading the selected fitted model...")
def load_fitted_model(
    model_id: str,
    serializer: str,
    signature: tuple[str, int, int],
):
    """Load one fitted artifact. This function performs inference setup only."""
    path = signature[0]
    if serializer == "xgboost":
        import xgboost as xgb

        model = xgb.XGBRegressor()
        model.load_model(path)
        return model
    if serializer == "catboost":
        from catboost import CatBoostRegressor

        model = CatBoostRegressor()
        model.load_model(path)
        return model
    if serializer == "joblib":
        import joblib

        return joblib.load(path)
    raise ValueError(f"Unsupported serializer for {model_id}: {serializer}")


@st.cache_data(show_spinner=False)
def parse_uploaded_history(file_name: str, content: bytes) -> pd.DataFrame:
    """Parse a C-MAPSS text file or a CSV containing its 26 columns."""
    buffer = io.BytesIO(content)
    if Path(file_name).suffix.lower() == ".txt":
        frame = pd.read_csv(buffer, sep=r"\s+", header=None)
    else:
        frame = pd.read_csv(buffer)
        if not set(CMAPSS_COLUMNS).issubset(frame.columns):
            buffer.seek(0)
            frame = pd.read_csv(buffer, header=None)

    if frame.shape[1] != len(CMAPSS_COLUMNS):
        raise ValueError(
            f"Expected {len(CMAPSS_COLUMNS)} columns, found {frame.shape[1]}."
        )
    if not set(CMAPSS_COLUMNS).issubset(frame.columns):
        frame.columns = CMAPSS_COLUMNS
    frame = frame[CMAPSS_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if frame.isna().any().any():
        raise ValueError("The uploaded file contains missing or nonnumeric values.")
    if frame.empty:
        raise ValueError("The uploaded file is empty.")
    frame[["unit_id", "cycle"]] = frame[["unit_id", "cycle"]].astype(int)
    return frame.sort_values(["unit_id", "cycle"]).reset_index(drop=True)


def _feature_columns(metadata: dict[str, Any]) -> list[str]:
    if metadata.get("feature_columns"):
        return list(metadata["feature_columns"])
    columns = [
        "cycle",
        "setting_1",
        "setting_2",
        "setting_3",
        *metadata["selected_sensors"],
    ]
    for sensor in metadata["selected_sensors"]:
        columns.append(f"{sensor}_diff_1")
        for window in metadata["windows"]:
            columns.extend(
                [
                    f"{sensor}_roll_mean_{window}",
                    f"{sensor}_roll_std_{window}",
                    f"{sensor}_slope_{window}",
                ]
            )
    return columns


def predict_uploaded_history(
    frame: pd.DataFrame,
    entry: dict[str, Any],
) -> pd.DataFrame:
    metadata = entry["metadata"]
    model_path = entry["paths"]["model_file"]
    model = load_fitted_model(
        entry["id"],
        metadata["serializer"],
        _file_signature(model_path),
    )
    features = engineer_features(
        frame,
        list(metadata["selected_sensors"]),
        tuple(metadata["windows"]),
    )
    last_indices = frame.groupby("unit_id")["cycle"].idxmax()
    last_features = features.loc[last_indices]
    expected_columns = _feature_columns(metadata)
    missing = set(expected_columns).difference(last_features.columns)
    if missing:
        raise ValueError(f"Cannot build required features: {sorted(missing)}")
    upper_limit = float(metadata.get("rul_cap", 125))
    prediction = np.clip(
        np.asarray(model.predict(last_features[expected_columns])).reshape(-1),
        0.0,
        upper_limit,
    )
    return pd.DataFrame(
        {
            "unit_id": frame.loc[last_indices, "unit_id"].to_numpy(dtype=int),
            "last_cycle": frame.loc[last_indices, "cycle"].to_numpy(dtype=int),
            "predicted_rul": prediction,
        }
    ).sort_values("unit_id", ignore_index=True)


def _status_label(rul: float) -> tuple[str, str]:
    if rul < 30:
        return "Critical", "#EF4444"
    if rul < 60:
        return "Maintenance soon", "#F59E0B"
    return "Stable", "#22C55E"


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {background: linear-gradient(145deg, #07111f 0%, #0b1628 55%, #101827 100%);}
        [data-testid="stHeader"] {background: transparent;}
        [data-testid="stToolbar"] {visibility: hidden;}
        .hero {padding: 2.2rem; border-radius: 24px; margin-bottom: 1.2rem;
               background: radial-gradient(circle at top right, #164e63 0%, #111827 45%, #0f172a 100%);
               border: 1px solid rgba(125, 211, 252, .2); box-shadow: 0 18px 50px rgba(0,0,0,.25);}
        .hero h1 {margin: 0; font-size: 2.45rem; color: #f8fafc; letter-spacing: -.03em;}
        .hero p {color: #bae6fd; margin: .55rem 0 0; font-size: 1.03rem;}
        .model-card {padding: 1.15rem; border-radius: 18px; min-height: 165px;
                     background: rgba(15, 23, 42, .78); border: 1px solid rgba(148,163,184,.18);}
        .model-number {font-size: .75rem; letter-spacing: .12em; color: #94a3b8; font-weight: 700;}
        .model-name {font-size: 1.15rem; color: #f8fafc; font-weight: 750; margin: .35rem 0;}
        .model-description {font-size: .88rem; color: #94a3b8; min-height: 45px;}
        .model-reason {font-size: .72rem; color: #94a3b8; margin-top: .35rem; line-height: 1.2;}
        .ready {color: #4ade80; font-weight: 700;} .waiting {color: #fbbf24; font-weight: 700;}
        .section-note {padding: .8rem 1rem; border-left: 3px solid #38bdf8;
                       background: rgba(14, 165, 233, .08); border-radius: 0 10px 10px 0;}
        div[data-testid="stMetric"] {background: rgba(15,23,42,.7); border: 1px solid rgba(148,163,184,.16);
                                     padding: .8rem; border-radius: 14px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_overview(
    registry: list[dict[str, Any]],
    comparison: pd.DataFrame,
) -> None:
    st.subheader("Model readiness")
    columns = st.columns(4)
    for column, entry in zip(columns, registry):
        state_class = "ready" if entry["available"] else "waiting"
        state_text = "● Ready" if entry["available"] else "○ Waiting for teammate"
        reason = ""
        if not entry["available"] and entry.get("missing"):
            reason = f'<div class="model-reason">{escape(str(entry["missing"][0]))}</div>'
        with column:
            st.markdown(
                f"""
                <div class="model-card" style="border-top: 3px solid {entry['color']}">
                  <div class="model-number">MODEL {entry['number']}</div>
                  <div class="model-name">{entry['name']}</div>
                  <div class="model-description">{entry['description']}</div>
                  <div class="{state_class}">{state_text}</div>
                  {reason}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("FD001 at a glance")
    train_path = DATA_DIR / "train_FD001.txt"
    test_path = DATA_DIR / "test_FD001.txt"
    if train_path.is_file() and test_path.is_file():
        summary = dataset_summary(
            (_file_signature(train_path), _file_signature(test_path))
        )
        metrics = st.columns(5)
        values = [
            ("Training engines", summary["train_engines"]),
            ("Test engines", summary["test_engines"]),
            ("Training rows", f"{summary['train_rows']:,}"),
            ("Test rows", f"{summary['test_rows']:,}"),
            ("Longest history", f"{summary['max_cycle']} cycles"),
        ]
        for column, (label, value) in zip(metrics, values):
            column.metric(label, value)
    else:
        st.info("Local FD001 data is not present. Saved model comparisons still work.")

    if not comparison.empty:
        winner = comparison.iloc[0]
        official = comparison.dropna(subset=["test_rmse"])
        left, right = st.columns(2)
        left.success(
            f"Validation selection: **{winner['model']}** — "
            f"RMSE {winner['validation_rmse']:.2f} cycles"
        )
        if not official.empty:
            official_winner = official.loc[official["test_rmse"].idxmin()]
            right.info(
                f"Lowest official-test error: **{official_winner['model']}** — "
                f"RMSE {official_winner['test_rmse']:.2f} cycles"
            )


def render_comparison(comparison: pd.DataFrame) -> None:
    if comparison.empty:
        st.warning("No complete model result package is available yet.")
        return

    st.markdown(
        '<div class="section-note">Models are ranked using validation RMSE. '
        "Official-test performance is reported separately to avoid selecting a model on test data.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    leaderboard = comparison[
        [
            "rank",
            "model",
            "device",
            "validation_rmse",
            "validation_mae",
            "validation_r2",
            "validation_nasa",
            "test_rmse",
            "test_mae",
            "test_r2",
            "test_nasa",
        ]
    ].copy()
    st.dataframe(
        leaderboard.style.format(precision=3).background_gradient(
            subset=["validation_rmse", "validation_mae"],
            cmap="YlGn_r",
        ),
        width="stretch",
        hide_index=True,
    )

    left, right = st.columns(2)
    error_data = comparison.melt(
        id_vars="model",
        value_vars=["validation_rmse", "validation_mae"],
        var_name="metric",
        value_name="cycles",
    )
    error_data["metric"] = error_data["metric"].map(
        {"validation_rmse": "RMSE", "validation_mae": "MAE"}
    )
    error_chart = px.bar(
        error_data,
        x="model",
        y="cycles",
        color="metric",
        barmode="group",
        title="Validation error (lower is better)",
        color_discrete_sequence=["#38BDF8", "#A78BFA"],
    )
    left.plotly_chart(error_chart, width="stretch")

    nasa_chart = px.bar(
        comparison,
        x="model",
        y="validation_nasa",
        color="model",
        title="NASA asymmetric score (lower is safer)",
        color_discrete_sequence=["#F59E0B", "#8B5CF6", "#22C55E", "#38BDF8"],
    )
    nasa_chart.update_layout(showlegend=False)
    right.plotly_chart(nasa_chart, width="stretch")

    speed_chart = px.scatter(
        comparison,
        x="train_seconds",
        y="validation_rmse",
        color="model",
        size=np.maximum(comparison["predict_seconds"], 0.01),
        hover_data=["device", "validation_mae", "validation_r2"],
        title="Accuracy versus training time",
        labels={
            "train_seconds": "Training time (seconds)",
            "validation_rmse": "Validation RMSE",
        },
    )
    st.plotly_chart(speed_chart, width="stretch")


def render_model_explorer(registry: list[dict[str, Any]]) -> None:
    available = [entry for entry in registry if entry["available"]]
    if not available:
        st.warning("No complete model package is available.")
        return
    by_name = {entry["name"]: entry for entry in available}
    selected_name = st.selectbox("Choose a model", list(by_name))
    entry = by_name[selected_name]
    metrics = read_csv(_file_signature(entry["paths"]["metrics_file"]))
    predictions = read_csv(_file_signature(entry["paths"]["predictions_file"]))
    validation, official = _best_rows(metrics)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Validation RMSE", f"{validation['rmse']:.2f}")
    metric_columns[1].metric("Validation MAE", f"{validation['mae']:.2f}")
    metric_columns[2].metric("Validation R²", f"{validation['r2']:.3f}")
    metric_columns[3].metric("NASA score", f"{validation['nasa_score']:.1f}")

    left, right = st.columns(2)
    scatter = px.scatter(
        predictions,
        x="true_rul",
        y="predicted_rul",
        color="residual",
        color_continuous_scale="RdBu_r",
        hover_data=["unit_id"],
        title="True versus predicted RUL",
    )
    limit = float(
        max(predictions["true_rul"].max(), predictions["predicted_rul"].max())
    )
    scatter.add_shape(type="line", x0=0, y0=0, x1=limit, y1=limit, line_dash="dash")
    left.plotly_chart(scatter, width="stretch")

    residuals = px.histogram(
        predictions,
        x="residual",
        nbins=18,
        marginal="box",
        title="Residual distribution",
        color_discrete_sequence=[entry["color"]],
    )
    residuals.add_vline(x=0, line_dash="dash", line_color="#F8FAFC")
    right.plotly_chart(residuals, width="stretch")

    st.dataframe(
        predictions.sort_values("residual", key=lambda values: values.abs(), ascending=False)
        .head(15)
        .style.format({"true_rul": "{:.1f}", "predicted_rul": "{:.1f}", "residual": "{:+.1f}"}),
        width="stretch",
        hide_index=True,
    )
    figure_path = PROJECT_ROOT / entry.get("summary_figure", "")
    if figure_path.is_file():
        with st.expander("Saved training summary"):
            st.image(str(figure_path), width="stretch")


def render_prediction(registry: list[dict[str, Any]]) -> None:
    available = [entry for entry in registry if entry["available"]]
    if not available:
        st.warning("Prediction unlocks when at least one complete model is available.")
        return
    by_name = {entry["name"]: entry for entry in available}
    selected_name = st.selectbox(
        "Prediction model",
        list(by_name),
        key="prediction_model",
    )
    uploaded = st.file_uploader(
        "Upload C-MAPSS engine history",
        type=["txt", "csv"],
        help="Use the original 26-column C-MAPSS format or a CSV with the project column names.",
    )
    st.caption(
        "Inference only: the selected fitted artifact is loaded from disk. "
        "This dashboard never launches a training script."
    )
    if uploaded is None:
        st.info("Upload an engine-history file to enable prediction.")
        return

    try:
        history = parse_uploaded_history(uploaded.name, uploaded.getvalue())
        st.success(
            f"Validated {len(history):,} cycles from "
            f"{history['unit_id'].nunique()} engine(s)."
        )
        with st.expander("Preview uploaded data"):
            st.dataframe(history.head(25), width="stretch", hide_index=True)
    except ValueError as error:
        st.error(str(error))
        return

    if not st.button("Predict remaining useful life", type="primary"):
        return
    try:
        predictions = predict_uploaded_history(history, by_name[selected_name])
    except Exception as error:
        st.error(f"Prediction could not be completed: {error}")
        return

    engine_id = st.selectbox("Inspect engine", predictions["unit_id"].tolist())
    selected = predictions.loc[predictions["unit_id"].eq(engine_id)].iloc[0]
    rul = float(selected["predicted_rul"])
    status, color = _status_label(rul)
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=rul,
            number={"suffix": " cycles"},
            title={"text": f"Engine {engine_id} · {status}"},
            gauge={
                "axis": {"range": [0, max(125, rul + 10)]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "rgba(239,68,68,.22)"},
                    {"range": [30, 60], "color": "rgba(245,158,11,.22)"},
                    {"range": [60, 125], "color": "rgba(34,197,94,.20)"},
                ],
            },
        )
    )
    st.plotly_chart(gauge, width="stretch")
    st.dataframe(
        predictions.style.format({"predicted_rul": "{:.2f}"}),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download predictions",
        predictions.to_csv(index=False).encode("utf-8"),
        file_name=f"{by_name[selected_name]['id']}_uploaded_predictions.csv",
        mime="text/csv",
    )
    st.warning(
        "This is an educational prediction from simulated C-MAPSS data, "
        "not a production aviation-maintenance decision."
    )


def render_app() -> None:
    st.set_page_config(
        page_title="Turbofan RUL Intelligence",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_style()
    st.markdown(
        """
        <div class="hero">
          <h1>Turbofan RUL Intelligence</h1>
          <p>Local predictive-maintenance dashboard · NASA C-MAPSS FD001 · cached inference only</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    registry = discover_models()
    comparison = build_comparison(registry)
    ready_count = sum(entry["available"] for entry in registry)
    st.caption(
        f"{ready_count}/4 model packages ready · Runs only on this laptop · No model retraining"
    )

    overview_tab, comparison_tab, explorer_tab, prediction_tab = st.tabs(
        ["Overview", "Compare models", "Model explorer", "Predict RUL"]
    )
    with overview_tab:
        render_overview(registry, comparison)
    with comparison_tab:
        render_comparison(comparison)
    with explorer_tab:
        render_model_explorer(registry)
    with prediction_tab:
        render_prediction(registry)


if __name__ == "__main__":
    render_app()

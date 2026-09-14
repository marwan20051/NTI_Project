"""XGBoost configuration and optional CUDA detection."""

import warnings

import numpy as np
import xgboost as xgb


def resolve_device(prefer_gpu: bool = True) -> str:
    """Return ``cuda`` only when this XGBoost build can train on the GPU."""
    if not prefer_gpu:
        return "cpu"

    build_info = xgb.build_info() if hasattr(xgb, "build_info") else {}
    if not build_info.get("USE_CUDA", False):
        return "cpu"

    try:
        probe = xgb.XGBRegressor(
            n_estimators=1,
            max_depth=1,
            tree_method="hist",
            device="cuda",
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe.fit(
                np.array([[0.0], [1.0]], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                verbose=False,
            )
        return "cuda"
    except (xgb.core.XGBoostError, ValueError, OSError):
        return "cpu"


def _common_parameters(device: str) -> dict:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": device,
        "random_state": 42,
        "n_jobs": -1,
    }


def baseline_parameters(device: str) -> dict:
    """Return a transparent, moderately sized XGBoost baseline."""
    return {
        **_common_parameters(device),
        "n_estimators": 350,
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
    }


def improved_parameters(device: str) -> dict:
    """Return a slower-learning, regularized configuration."""
    return {
        **_common_parameters(device),
        "n_estimators": 1200,
        "max_depth": 4,
        "learning_rate": 0.025,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 5.0,
        "early_stopping_rounds": 75,
    }

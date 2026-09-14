"""CatBoost configuration with a safe NVIDIA GPU fallback."""

import numpy as np
from catboost import CatBoostError, CatBoostRegressor


def resolve_catboost_device(prefer_gpu: bool = True) -> str:
    """Return ``GPU`` only when CatBoost can fit on the first CUDA device."""
    if not prefer_gpu:
        return "CPU"

    try:
        probe = CatBoostRegressor(
            iterations=1,
            depth=1,
            task_type="GPU",
            devices="0",
            verbose=False,
            allow_writing_files=False,
            random_seed=42,
        )
        probe.fit(
            np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
            np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        )
        return "GPU"
    except (CatBoostError, OSError, RuntimeError):
        return "CPU"


def _common_parameters(device: str) -> dict:
    parameters = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "task_type": device,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    }
    if device == "GPU":
        parameters["devices"] = "0"
    return parameters


def catboost_baseline_parameters(device: str) -> dict:
    """Return a clear raw-feature CatBoost baseline configuration."""
    return {
        **_common_parameters(device),
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
    }


def catboost_improved_parameters(device: str) -> dict:
    """Return a regularized configuration with overfitting detection."""
    return {
        **_common_parameters(device),
        "iterations": 1800,
        "depth": 7,
        "learning_rate": 0.025,
        "l2_leaf_reg": 8.0,
        "random_strength": 0.5,
        "od_type": "Iter",
        "od_wait": 100,
        "use_best_model": True,
    }

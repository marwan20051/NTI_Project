import numpy as np
from catboost import CatBoostRegressor

from cmapss_rul.catboost_modeling import (
    catboost_baseline_parameters,
    catboost_improved_parameters,
    resolve_catboost_device,
)


def test_catboost_parameter_sets_are_reproducible():
    assert catboost_baseline_parameters("CPU")["random_seed"] == 42
    assert catboost_improved_parameters("CPU")["random_seed"] == 42


def test_improved_catboost_is_regularized_and_slower_learning():
    baseline = catboost_baseline_parameters("CPU")
    improved = catboost_improved_parameters("CPU")
    assert improved["learning_rate"] < baseline["learning_rate"]
    assert improved["l2_leaf_reg"] > baseline["l2_leaf_reg"]


def test_catboost_device_can_force_cpu():
    assert resolve_catboost_device(prefer_gpu=False) == "CPU"


def test_catboost_cpu_baseline_can_fit_and_predict():
    parameters = catboost_baseline_parameters("CPU")
    parameters["iterations"] = 2
    model = CatBoostRegressor(**parameters)
    features = np.arange(12, dtype=float).reshape(-1, 1)
    target = np.linspace(11, 0, num=12)
    model.fit(features, target)
    assert model.predict(features[:3]).shape == (3,)

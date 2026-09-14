from cmapss_rul.modeling import (
    baseline_parameters,
    improved_parameters,
    resolve_device,
)


def test_parameter_sets_are_reproducible():
    assert baseline_parameters("cpu")["random_state"] == 42
    assert improved_parameters("cpu")["random_state"] == 42


def test_improved_model_is_regularized_and_slower_learning():
    baseline = baseline_parameters("cpu")
    improved = improved_parameters("cpu")
    assert improved["learning_rate"] < baseline["learning_rate"]
    assert improved["reg_lambda"] > baseline["reg_lambda"]


def test_resolve_device_can_force_cpu():
    assert resolve_device(prefer_gpu=False) == "cpu"

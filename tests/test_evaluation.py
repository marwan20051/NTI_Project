import numpy as np

from cmapss_rul.evaluation import nasa_score, regression_metrics


def test_nasa_score_is_zero_for_perfect_predictions():
    assert nasa_score(np.array([10, 20]), np.array([10, 20])) == 0.0


def test_nasa_score_penalizes_late_prediction_more():
    true = np.array([50.0])
    early = nasa_score(true, np.array([40.0]))
    late = nasa_score(true, np.array([60.0]))
    assert late > early


def test_regression_metrics_contains_expected_keys():
    metrics = regression_metrics(
        np.array([1, 2, 3]),
        np.array([1, 2, 3]),
    )
    assert metrics == {
        "rmse": 0.0,
        "mae": 0.0,
        "r2": 1.0,
        "nasa_score": 0.0,
    }

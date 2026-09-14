import numpy as np
import pandas as pd

from cmapss_rul.features import engineer_features, select_sensors


def test_select_sensors_drops_constant_columns_and_limits_count():
    frame = pd.DataFrame(
        {
            "sensor_1": [1.0] * 6,
            "sensor_2": [0, 0, 0, 3, 3, 3],
            "sensor_3": [5, 4, 3, 2, 1, 0],
            "rul": [5, 4, 3, 2, 1, 0],
        }
    )
    assert select_sensors(frame, top_n=1) == ["sensor_3"]


def test_engineering_is_causal_and_finite():
    frame = pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
            "setting_1": [0.0] * 5,
            "setting_2": [0.0] * 5,
            "setting_3": [0.0] * 5,
            "sensor_2": [1.0, 2.0, 100.0, 4.0, 6.0],
        }
    )
    features = engineer_features(frame, ["sensor_2"], windows=(2,))
    assert features.loc[1, "sensor_2_roll_mean_2"] == 1.5
    assert features.loc[1, "sensor_2_diff_1"] == 1.0
    assert features.loc[1, "sensor_2_slope_2"] == 1.0
    assert np.isfinite(features.to_numpy()).all()

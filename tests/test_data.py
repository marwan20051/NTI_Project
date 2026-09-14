import pandas as pd
import pytest

from cmapss_rul.data import (
    CMAPSS_COLUMNS,
    add_train_rul,
    attach_test_rul,
    last_cycle_rows,
    split_by_engine,
)


def sample_frame():
    rows = []
    for unit, maximum in [(1, 3), (2, 2), (3, 4), (4, 3)]:
        for cycle in range(1, maximum + 1):
            row = {column: 0.0 for column in CMAPSS_COLUMNS}
            row.update(unit_id=unit, cycle=cycle, sensor_2=float(unit + cycle))
            rows.append(row)
    return pd.DataFrame(rows)


def test_add_train_rul_uses_each_engines_final_cycle():
    result = add_train_rul(sample_frame())
    assert result.loc[result.unit_id.eq(1), "rul"].tolist() == [2, 1, 0]
    assert result.loc[result.unit_id.eq(2), "rul"].tolist() == [1, 0]


def test_last_cycle_rows_returns_one_row_per_engine():
    result = last_cycle_rows(sample_frame())
    assert result[["unit_id", "cycle"]].values.tolist() == [
        [1, 3],
        [2, 2],
        [3, 4],
        [4, 3],
    ]


def test_attach_test_rul_requires_one_label_per_engine():
    with pytest.raises(ValueError, match="one RUL value per test engine"):
        attach_test_rul(sample_frame(), pd.Series([10, 20]))


def test_split_by_engine_has_no_overlap():
    train, valid = split_by_engine(
        sample_frame(), validation_size=0.25, random_state=42
    )
    assert set(train.unit_id).isdisjoint(set(valid.unit_id))
    assert len(set(valid.unit_id)) == 1

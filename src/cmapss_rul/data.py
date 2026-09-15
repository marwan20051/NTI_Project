"""Loading and leakage-safe splitting helpers for NASA C-MAPSS data."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from cmapss_rul.schema import CMAPSS_COLUMNS


def load_sensor_file(path: Path) -> pd.DataFrame:
    """Load one whitespace-separated C-MAPSS sensor file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing C-MAPSS file: {path}")

    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != len(CMAPSS_COLUMNS):
        raise ValueError(
            f"Invalid C-MAPSS schema in {path}: expected "
            f"{len(CMAPSS_COLUMNS)} columns, found {frame.shape[1]}"
        )
    frame.columns = CMAPSS_COLUMNS
    if frame.isna().any().any():
        raise ValueError(f"Missing values found in {path}")

    frame[["unit_id", "cycle"]] = frame[["unit_id", "cycle"]].astype(int)
    return frame


def add_train_rul(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate RUL for every run-to-failure training observation."""
    result = frame.copy()
    maximum_cycle = result.groupby("unit_id")["cycle"].transform("max")
    result["rul"] = maximum_cycle - result["cycle"]
    if result["rul"].lt(0).any():
        raise ValueError("Training RUL cannot be negative")
    return result


def last_cycle_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the final observed row for each engine in unit order."""
    indices = frame.groupby("unit_id")["cycle"].idxmax()
    return frame.loc[indices].sort_values("unit_id").reset_index(drop=True)


def attach_test_rul(frame: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Attach official labels to final-cycle test observations."""
    result = last_cycle_rows(frame)
    clean_labels = pd.Series(labels).reset_index(drop=True).astype(float)
    if len(result) != len(clean_labels):
        raise ValueError("Expected one RUL value per test engine")
    result["rul"] = clean_labels.to_numpy()
    return result


def split_by_engine(
    frame: pd.DataFrame,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split complete engines into train and validation partitions."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        random_state=random_state,
    )
    train_index, valid_index = next(
        splitter.split(frame, groups=frame["unit_id"])
    )
    train = frame.iloc[train_index].copy()
    valid = frame.iloc[valid_index].copy()
    if not set(train["unit_id"]).isdisjoint(set(valid["unit_id"])):
        raise RuntimeError("Engine leakage detected")
    return train, valid


def make_validation_subset(
    frame: pd.DataFrame,
    min_rul: int = 10,
    max_rul: int = 100,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Truncate run-to-failure engines to realistic pre-failure snapshots."""
    if "rul" not in frame:
        raise ValueError("Validation frame must contain calculated RUL values")

    rng = np.random.default_rng(random_state)
    histories = []
    for _, engine in frame.groupby("unit_id", sort=True):
        lifetime = int(engine["cycle"].max())
        upper = min(max_rul, lifetime - 1)
        lower = min(min_rul, upper)
        sampled_rul = int(rng.integers(lower, upper + 1))
        cutoff_cycle = lifetime - sampled_rul
        histories.append(engine.loc[engine["cycle"].le(cutoff_cycle)])

    history = pd.concat(histories).sort_index().copy()
    snapshots = last_cycle_rows(history)
    if len(snapshots) != frame["unit_id"].nunique():
        raise RuntimeError("Validation truncation lost one or more engines")
    return history, snapshots


def load_fd001(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load FD001 train/test trajectories and official test labels."""
    data_dir = Path(data_dir)
    train = add_train_rul(load_sensor_file(data_dir / "train_FD001.txt"))
    test = load_sensor_file(data_dir / "test_FD001.txt")

    rul_path = data_dir / "RUL_FD001.txt"
    if not rul_path.is_file():
        raise FileNotFoundError(f"Missing C-MAPSS file: {rul_path}")
    labels = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0]
    test_last = attach_test_rul(test, labels)
    return train, test, test_last

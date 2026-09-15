"""Train FD001 Model 4 (CatBoost) and save reusable artifacts."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostError, CatBoostRegressor

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cmapss_rul.catboost_modeling import (  # noqa: E402
    catboost_baseline_parameters,
    catboost_improved_parameters,
    resolve_catboost_device,
)
from cmapss_rul.evaluation import regression_metrics  # noqa: E402
from cmapss_rul.training_utils import (  # noqa: E402
    RUL_CAP,
    baseline_data,
    cache_is_complete,
    engineered_final_data,
    engineered_validation_data,
    prediction_frame,
    prepare_experiment_data,
    save_training_plot,
)

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cmapss" / "CMaps"
ARTIFACTS = {
    "model": MODEL_DIR / "catboost_fd001.cbm",
    "metrics": MODEL_DIR / "catboost_fd001_metrics.csv",
    "predictions": MODEL_DIR / "catboost_fd001_predictions.csv",
}
SUMMARY_FIGURE = MODEL_DIR / "catboost_fd001_run_summary.png"


def _validate_saved_model(path: Path) -> None:
    model = CatBoostRegressor()
    model.load_model(path)


def train(force: bool = False) -> pd.DataFrame:
    """Train CatBoost unless a complete cache exists."""
    if cache_is_complete(
        ARTIFACTS,
        model_validator=_validate_saved_model,
        expected_model_prefix="CatBoost",
    ) and not force:
        print("CatBoost is already trained. Add --force to retrain it.")
        cached_metrics = pd.read_csv(ARTIFACTS["metrics"])
        if not SUMMARY_FIGURE.is_file():
            cached_predictions = pd.read_csv(ARTIFACTS["predictions"])
            save_training_plot(
                cached_metrics,
                cached_predictions,
                "CatBoost",
                "#6A4C93",
                MODEL_DIR,
            )
        return cached_metrics

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (
        train_df,
        test_df,
        official_test,
        train_split,
        validation_history,
        validation_snapshots,
    ) = prepare_experiment_data(DATA_DIR)
    device = resolve_catboost_device(prefer_gpu=True)
    print(f"Training CatBoost on {device}...")

    baseline_train, baseline_valid = baseline_data(
        train_split,
        validation_snapshots,
    )
    baseline_model = CatBoostRegressor(**catboost_baseline_parameters(device))
    start = time.perf_counter()
    try:
        baseline_model.fit(baseline_train, train_split["rul"])
    except CatBoostError:
        if device != "GPU":
            raise
        print("CatBoost GPU training failed; retrying on CPU.")
        device = "CPU"
        baseline_model = CatBoostRegressor(**catboost_baseline_parameters(device))
        baseline_model.fit(baseline_train, train_split["rul"])
    baseline_train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    baseline_predictions = np.clip(baseline_model.predict(baseline_valid), 0, None)
    baseline_predict_seconds = time.perf_counter() - start
    baseline_metrics = regression_metrics(
        validation_snapshots["rul"],
        baseline_predictions,
    )

    sensors, train_features, validation_features, validation_last = (
        engineered_validation_data(train_split, validation_history)
    )
    print(f"Selected sensors: {sensors}")
    improved_model = CatBoostRegressor(**catboost_improved_parameters(device))
    start = time.perf_counter()
    improved_model.fit(
        train_features,
        train_split["rul"].clip(upper=RUL_CAP),
        eval_set=(
            validation_features,
            validation_history["rul"].clip(upper=RUL_CAP),
        ),
    )
    improved_train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    improved_predictions = np.clip(improved_model.predict(validation_last), 0, None)
    improved_predict_seconds = time.perf_counter() - start
    improved_metrics = regression_metrics(
        validation_snapshots["rul"],
        improved_predictions,
    )

    _, full_train_features, test_last = engineered_final_data(train_df, test_df)
    final_parameters = catboost_improved_parameters(device)
    for key in ("use_best_model", "od_type", "od_wait"):
        final_parameters.pop(key)
    final_parameters["iterations"] = max(1, improved_model.get_best_iteration() + 1)
    final_model = CatBoostRegressor(**final_parameters)
    start = time.perf_counter()
    final_model.fit(
        full_train_features,
        train_df["rul"].clip(upper=RUL_CAP),
    )
    final_train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    test_predictions = np.clip(final_model.predict(test_last), 0, None)
    test_predict_seconds = time.perf_counter() - start
    official_metrics = regression_metrics(official_test["rul"], test_predictions)

    metrics = pd.DataFrame(
        [
            {
                "model": "CatBoost baseline",
                "split": "validation",
                "device": device,
                "train_seconds": baseline_train_seconds,
                "predict_seconds": baseline_predict_seconds,
                **baseline_metrics,
            },
            {
                "model": "CatBoost improved",
                "split": "validation",
                "device": device,
                "train_seconds": improved_train_seconds,
                "predict_seconds": improved_predict_seconds,
                **improved_metrics,
            },
            {
                "model": "CatBoost improved",
                "split": "official_test",
                "device": device,
                "train_seconds": final_train_seconds,
                "predict_seconds": test_predict_seconds,
                **official_metrics,
            },
        ]
    )
    predictions = prediction_frame(official_test, test_predictions)
    metrics.to_csv(ARTIFACTS["metrics"], index=False)
    predictions.to_csv(ARTIFACTS["predictions"], index=False)
    final_model.save_model(ARTIFACTS["model"])
    save_training_plot(
        metrics,
        predictions,
        "CatBoost",
        "#6A4C93",
        MODEL_DIR,
    )
    print(f"CatBoost complete. Official RMSE: {official_metrics['rmse']:.3f}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even when valid saved artifacts already exist.",
    )
    args = parser.parse_args(argv)
    train(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"

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

DISPLAY_NAMES = {
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "random_forest": "Random Forest",
    "ridge": "Ridge Regression",
}


def _algorithm_name(metrics_path: Path) -> str:
    key = metrics_path.name.removesuffix("_fd001_metrics.csv")
    return DISPLAY_NAMES.get(key, key.replace("_", " ").title())


def load_cached_comparison(model_dir: Path = MODEL_DIR) -> pd.DataFrame:
    """Build one ranked table from every compatible FD001 metrics file."""
    records: list[dict[str, object]] = []
    metric_files = sorted(Path(model_dir).glob("*_fd001_metrics.csv"))
    if not metric_files:
        raise FileNotFoundError(
            "No cached FD001 metrics were found. Run models/train_xgboost.py, "
            "models/train_catboost.py, or another model training script first."
        )

    for path in metric_files:
        metrics = pd.read_csv(path)
        missing = REQUIRED_METRIC_COLUMNS.difference(metrics.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

        validation = metrics.loc[metrics["split"].eq("validation")].sort_values(
            ["rmse", "nasa_score", "mae"]
        )
        if validation.empty:
            continue
        best_validation = validation.iloc[0]
        official = metrics.loc[
            metrics["split"].eq("official_test")
            & metrics["model"].eq(best_validation["model"])
        ].sort_values(["rmse", "nasa_score", "mae"])

        record: dict[str, object] = {
            "algorithm": _algorithm_name(path),
            "model": best_validation["model"],
            "device": best_validation["device"],
            "validation_rmse": best_validation["rmse"],
            "validation_mae": best_validation["mae"],
            "validation_r2": best_validation["r2"],
            "validation_nasa_score": best_validation["nasa_score"],
            "train_seconds": best_validation["train_seconds"],
            "predict_seconds": best_validation["predict_seconds"],
            "test_rmse": np.nan,
            "test_mae": np.nan,
            "test_r2": np.nan,
            "test_nasa_score": np.nan,
        }
        if not official.empty:
            best_official = official.iloc[0]
            record.update(
                {
                    "test_rmse": best_official["rmse"],
                    "test_mae": best_official["mae"],
                    "test_r2": best_official["r2"],
                    "test_nasa_score": best_official["nasa_score"],
                }
            )
        records.append(record)

    if not records:
        raise ValueError("Metrics files exist, but none contains a validation result.")

    comparison = pd.DataFrame(records).sort_values(
        ["validation_rmse", "validation_nasa_score", "validation_mae"],
        ignore_index=True,
    )
    comparison.insert(0, "rank", np.arange(1, len(comparison) + 1))
    return comparison


def save_comparison_plot(comparison: pd.DataFrame) -> Path:
    """Save validation and official-test comparison charts."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    metric_specs = [
        ("validation_rmse", "Validation RMSE"),
        ("validation_nasa_score", "Validation NASA score"),
    ]
    if comparison["test_rmse"].notna().any():
        metric_specs.extend(
            [
                ("test_rmse", "Official-test RMSE"),
                ("test_nasa_score", "Official-test NASA score"),
            ]
        )

    figure, axes = plt.subplots(
        1,
        len(metric_specs),
        figsize=(5 * len(metric_specs), 5),
    )
    axes_array = np.atleast_1d(axes)
    palette = sns.color_palette("viridis", n_colors=len(comparison))
    for axis, (column, title) in zip(axes_array, metric_specs):
        sns.barplot(
            data=comparison,
            x="algorithm",
            y=column,
            hue="algorithm",
            palette=palette,
            legend=False,
            ax=axis,
        )
        axis.set(title=title, xlabel="", ylabel="Lower is better")
        axis.tick_params(axis="x", rotation=15)
    figure.suptitle("FD001 cached model comparison", fontweight="bold")
    figure.tight_layout()
    path = MODEL_DIR / "fd001_model_comparison.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def compare_cached_results(show_plot: bool = True) -> pd.DataFrame:
    """Rank saved results and show their chart without fitting a model."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    comparison = load_cached_comparison()
    output_path = MODEL_DIR / "final_model_comparison.csv"
    comparison.to_csv(output_path, index=False)
    figure_path = save_comparison_plot(comparison)

    columns = [
        "rank",
        "algorithm",
        "model",
        "validation_rmse",
        "validation_mae",
        "validation_nasa_score",
        "test_rmse",
        "test_mae",
        "test_nasa_score",
    ]
    print("\nCached FD001 model comparison (ranked by validation RMSE):")
    print(comparison[columns].round(3).to_string(index=False))
    print(f"\nSelection winner: {comparison.iloc[0]['model']}")
    if comparison["test_rmse"].notna().any():
        official_winner = comparison.loc[comparison["test_rmse"].idxmin(), "model"]
        print(f"Lowest official-test RMSE: {official_winner}")
    print(f"Saved table: {output_path}")
    print(f"Saved chart: {figure_path}")

    if show_plot:
        image = plt.imread(figure_path)
        figure, axis = plt.subplots(figsize=(14, 5))
        axis.imshow(image)
        axis.axis("off")
        figure.tight_layout()
        plt.show()
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare saved FD001 results; this file never trains models."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the Matplotlib chart without opening it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    compare_cached_results(show_plot=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

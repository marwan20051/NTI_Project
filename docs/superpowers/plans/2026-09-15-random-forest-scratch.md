# Scratch Random Forest Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the submitted hand-written Random Forest into a reproducible FD001 Model 2 trainer that saves cached artifacts consumed by the local Streamlit dashboard.

**Architecture:** Put the serializable NumPy estimator in the importable `cmapss_rul` package and keep experiment orchestration in an independent script. Reuse the project's shared leakage-safe split, feature engineering, evaluation, prediction, plotting, and cache contracts so Model 2 is directly comparable with Models 3 and 4.

**Tech Stack:** Python 3.12, NumPy, pandas, Joblib, Matplotlib/Seaborn, Streamlit, existing `cmapss_rul` utilities

---

### Task 1: Create the reusable scratch estimator

**Files:**
- Create: `src/cmapss_rul/random_forest_scratch.py`
- Source to remove later: `random forest.py`

- [ ] **Step 1: Record the estimator contract as an executable smoke check**

Use this command after the module exists; before implementation it must fail with
`ModuleNotFoundError`:

```powershell
@'
import numpy as np
from cmapss_rul.random_forest_scratch import RandomForestScratch

X = np.arange(120, dtype=float).reshape(40, 3)
y = 0.5 * X[:, 0] - X[:, 1]
model_a = RandomForestScratch(n_estimators=4, max_depth=4, random_state=7)
model_b = RandomForestScratch(n_estimators=4, max_depth=4, random_state=7)
pred_a = model_a.fit(X, y).predict(X[:5])
pred_b = model_b.fit(X, y).predict(X[:5])
assert pred_a.shape == (5,)
assert np.isfinite(pred_a).all()
assert np.allclose(pred_a, pred_b)
assert model_a.feature_importances_.shape == (3,)
assert np.isclose(model_a.feature_importances_.sum(), 1.0)
print("scratch estimator contract passed")
'@ | python -
```

- [ ] **Step 2: Implement an importable, deterministic estimator**

Create an estimator with this public interface:

```python
class RandomForestScratch:
    def __init__(
        self,
        n_estimators: int = 60,
        max_depth: int = 12,
        min_samples_leaf: int = 5,
        max_features: float = 0.5,
        max_samples: float = 0.8,
        max_thresholds: int = 64,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None: ...

    def fit(self, X, y) -> "RandomForestScratch": ...
    def predict(self, X) -> np.ndarray: ...
```

Validate all hyperparameters and require finite 2-D `X` and finite 1-D `y`.
Reset `np.random.default_rng(self.random_state)` inside every `fit` call. For
each tree, sample `ceil(max_samples * n_rows)` bootstrap rows. At every node,
sample `ceil(max_features * n_features)` feature indices. Limit continuous
split candidates to at most `max_thresholds` evenly spaced valid positions,
calculate SSE reduction, and stop on depth, leaf-size, constant-target, or
non-positive-gain conditions. Store normalized split-count importance in
`feature_importances_` and reject prediction before fitting or with the wrong
feature count.

- [ ] **Step 3: Run estimator checks**

Run the smoke command from Step 1 and:

```powershell
python -m py_compile src\cmapss_rul\random_forest_scratch.py
```

Expected: `scratch estimator contract passed` and exit code 0.

- [ ] **Step 4: Commit the estimator**

```powershell
git add src/cmapss_rul/random_forest_scratch.py
git commit -m "feat: add reusable scratch random forest"
```

### Task 2: Build the cached FD001 trainer

**Files:**
- Create: `scripts/train_random_forest.py`

- [ ] **Step 1: Define paths and artifact contract**

Use these exact paths and serializer metadata:

```python
ARTIFACTS = {
    "model": MODEL_DIR / "random_forest_fd001.joblib",
    "metrics": METRIC_DIR / "random_forest_fd001_metrics.csv",
    "predictions": PREDICTION_DIR / "random_forest_fd001_predictions.csv",
    "metadata": METADATA_DIR / "random_forest_fd001.json",
}
SUMMARY_FIGURE = FIGURE_DIR / "random_forest_fd001_run_summary.png"
```

The metadata JSON must contain:

```python
{
    "id": "random_forest",
    "display_name": "Random Forest",
    "dataset": "FD001",
    "serializer": "joblib",
    "model_file": "models/random_forest_fd001.joblib",
    "metrics_file": "results/metrics/random_forest_fd001_metrics.csv",
    "predictions_file": "results/predictions/random_forest_fd001_predictions.csv",
    "summary_figure": "results/figures/random_forest_fd001_run_summary.png",
    "selected_sensors": selected_sensors,
    "feature_columns": feature_columns,
    "windows": [5, 15],
    "rul_cap": RUL_CAP,
}
```

- [ ] **Step 2: Implement fair baseline, improved, and official runs**

Import `prepare_experiment_data`, `baseline_data`,
`engineered_validation_data`, `engineered_final_data`, `prediction_frame`,
`cache_is_complete`, and `save_training_plot`. Use these configurations:

```python
BASELINE_PARAMETERS = {
    "n_estimators": 30,
    "max_depth": 10,
    "min_samples_leaf": 8,
    "max_features": 0.5,
    "max_samples": 0.7,
    "max_thresholds": 48,
    "random_state": 42,
}
IMPROVED_PARAMETERS = {
    "n_estimators": 60,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "max_features": 0.5,
    "max_samples": 0.8,
    "max_thresholds": 64,
    "random_state": 42,
}
```

Fit the baseline on raw nonconstant inputs and uncapped validation targets,
then fit the improved model on the shared engineered inputs with capped targets.
Fit the final improved configuration on all FD001 training engines and evaluate
the final-cycle official test inputs. Clip predictions to `[0, RUL_CAP]`. Time
every fit and prediction call with `time.perf_counter()` and create three metric
rows named `Random Forest baseline` and `Random Forest improved` with splits
`validation`, `validation`, and `official_test`.

- [ ] **Step 3: Save and validate the complete package**

Use `joblib.dump(final_model, ARTIFACTS["model"])`, write the metrics and
predictions CSVs, save metadata, and call:

```python
save_training_plot(
    metrics,
    predictions,
    "Random Forest",
    "#34D399",
    FIGURE_DIR,
)
```

The model validator must load the Joblib file and require a
`RandomForestScratch` instance with at least one fitted tree. Call
`cache_is_complete(..., expected_model_prefix="Random Forest")` before any data
loading, and expose `--force` using the same CLI shape as the other trainers.

- [ ] **Step 4: Compile the trainer**

```powershell
python -m py_compile scripts\train_random_forest.py
```

Expected: exit code 0.

- [ ] **Step 5: Commit the trainer**

```powershell
git add scripts/train_random_forest.py
git commit -m "feat: train cached scratch random forest"
```

### Task 3: Remove the root submission and document Model 2

**Files:**
- Delete: `random forest.py`
- Modify: `README.md`

- [ ] **Step 1: Remove the incompatible root script after confirming the replacement exists**

Verify `scripts/train_random_forest.py` and
`src/cmapss_rul/random_forest_scratch.py` exist, then delete only the root
`random forest.py`.

- [ ] **Step 2: Update usage documentation**

Add this command beside the other independent trainers:

```powershell
python scripts/train_random_forest.py
```

Document that Model 2 intentionally uses an educational NumPy implementation,
runs on CPU, shares the same validation/features/metrics contract, caches its
model with Joblib, and may be slower than library implementations.

- [ ] **Step 3: Commit relocation and documentation**

```powershell
git add README.md "random forest.py"
git commit -m "docs: add scratch random forest workflow"
```

### Task 4: Execute and verify end to end

**Files generated:**
- `models/random_forest_fd001.joblib`
- `models/metadata/random_forest_fd001.json`
- `results/metrics/random_forest_fd001_metrics.csv`
- `results/predictions/random_forest_fd001_predictions.csv`
- `results/figures/random_forest_fd001_run_summary.png`

- [ ] **Step 1: Train Model 2 once**

```powershell
conda activate nti-cmapss
python -B scripts\train_random_forest.py
```

Expected: baseline, improved, and official-test results print; all five artifacts
exist and are nonempty.

- [ ] **Step 2: Confirm cache reuse**

Run the same command again. Expected output:

```text
Random Forest is already trained. Add --force to retrain it.
```

- [ ] **Step 3: Validate artifact schemas and model loading**

```powershell
@'
import json
import joblib
import pandas as pd
from pathlib import Path

root = Path.cwd()
metadata = json.loads((root / "models/metadata/random_forest_fd001.json").read_text())
model = joblib.load(root / metadata["model_file"])
metrics = pd.read_csv(root / metadata["metrics_file"])
predictions = pd.read_csv(root / metadata["predictions_file"])
assert len(model.trees_) > 0
assert {"validation", "official_test"}.issubset(metrics["split"])
assert {"unit_id", "true_rul", "predicted_rul", "residual"}.issubset(predictions)
assert len(predictions) == 100
print("Random Forest artifact package passed")
'@ | python -
```

- [ ] **Step 4: Verify Streamlit integration and inference-only boundary**

```powershell
@'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("main.py", default_timeout=60).run(timeout=60)
assert not at.exception
assert any("3/4 model packages ready" in item.value for item in at.caption)
print("Streamlit recognizes Model 2")
'@ | python -

rg -n "\.fit\(|train_random_forest|cmapss_rul\.data" main.py
```

Expected: Streamlit check passes and `rg` finds no training call or trainer/data
import in `main.py`.

- [ ] **Step 5: Inspect final changes and commit generated artifacts**

```powershell
git diff --check
git status --short
git add README.md models/random_forest_fd001.joblib models/metadata/random_forest_fd001.json results/metrics/random_forest_fd001_metrics.csv results/predictions/random_forest_fd001_predictions.csv results/figures/random_forest_fd001_run_summary.png
git commit -m "results: add FD001 scratch random forest"
```

- [ ] **Step 6: Keep the completed branch local until explicitly asked to push**

Report the validation and official-test metrics, runtime, generated files, and
branch status. Do not push, merge, or deploy without an explicit request.

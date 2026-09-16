# Dashboard Winner Cards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the local Streamlit dashboard clearly identify the validation-selected model, official-test leader, and speed champion without inventing a misleading combined score.

**Architecture:** Keep the feature in `main.py`. Add pure dataframe helpers for deterministic winner selection and award labels, then render one reusable Decision center in the Overview and Compare models tabs. The feature consumes the existing comparison dataframe and never loads or retrains a model.

**Tech Stack:** Python 3.12, pandas, NumPy, Streamlit, existing Plotly dashboard, Streamlit AppTest

---

### Task 1: Add deterministic winner selection

**Files:**
- Modify: `main.py` after `build_comparison`

- [ ] **Step 1: Run the failing import check**

```powershell
conda run --no-capture-output -n nti-cmapss python -B -c "from main import select_dashboard_winners"
```

Expected: FAIL with `ImportError` because the helper does not exist.

- [ ] **Step 2: Add pure selection and award helpers**

```python
def _lowest_finite_row(comparison: pd.DataFrame, column: str) -> pd.Series | None:
    if comparison.empty or column not in comparison:
        return None
    values = pd.to_numeric(comparison[column], errors="coerce")
    candidates = comparison.loc[np.isfinite(values)].copy()
    if candidates.empty:
        return None
    candidates[column] = pd.to_numeric(candidates[column])
    return candidates.sort_values([column, "model"], kind="stable").iloc[0]


def select_dashboard_winners(
    comparison: pd.DataFrame,
) -> dict[str, pd.Series | None]:
    return {
        "validation": _lowest_finite_row(comparison, "validation_rmse"),
        "official": _lowest_finite_row(comparison, "test_rmse"),
        "speed": _lowest_finite_row(comparison, "train_seconds"),
        "prediction": _lowest_finite_row(comparison, "predict_seconds"),
    }


def award_labels(comparison: pd.DataFrame) -> dict[str, str]:
    awards: dict[str, list[str]] = {
        str(model): [] for model in comparison.get("model", pd.Series(dtype=str))
    }
    winners = select_dashboard_winners(comparison)
    for key, label in (
        ("validation", "🏆 Recommended"),
        ("official", "🎯 Test leader"),
        ("speed", "⚡ Fastest"),
    ):
        winner = winners[key]
        if winner is not None:
            awards.setdefault(str(winner["model"]), []).append(label)
    return {model: " · ".join(labels) for model, labels in awards.items()}
```

- [ ] **Step 3: Run selection smoke tests**

```powershell
@'
import numpy as np
import pandas as pd
from main import award_labels, build_comparison, discover_models, select_dashboard_winners

comparison = build_comparison(discover_models())
winners = select_dashboard_winners(comparison)
assert winners["validation"]["model"] == "Ridge Regression"
assert winners["official"]["model"] == "CatBoost"
assert winners["speed"]["model"] == "Ridge Regression"
assert winners["prediction"]["model"] == "Ridge Regression"
assert "🏆 Recommended" in award_labels(comparison)["Ridge Regression"]
assert select_dashboard_winners(comparison.assign(test_rmse=np.nan))["official"] is None
assert select_dashboard_winners(pd.DataFrame())["speed"] is None
invalid = pd.DataFrame({"model": ["A"], "validation_rmse": [np.inf]})
assert select_dashboard_winners(invalid)["validation"] is None
print("winner selection checks passed")
'@ | conda run --no-capture-output -n nti-cmapss python -B -
```

Expected: `winner selection checks passed`.

### Task 2: Render the Decision center

**Files:**
- Modify: `main.py` in `_inject_style` and before `render_overview`

- [ ] **Step 1: Add responsive Decision center CSS**

```css
.decision-grid {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin: .55rem 0 .9rem;}
.decision-card {padding: 1.2rem; border-radius: 18px; min-height: 175px; background: rgba(15,23,42,.88); border: 1px solid rgba(148,163,184,.18); box-shadow: 0 12px 28px rgba(0,0,0,.18);}
.decision-card.gold {border-top: 4px solid #FBBF24;}
.decision-card.purple {border-top: 4px solid #A78BFA;}
.decision-card.cyan {border-top: 4px solid #22D3EE;}
.decision-award {font-size: .76rem; letter-spacing: .1em; text-transform: uppercase; color: #CBD5E1; font-weight: 800;}
.decision-model {font-size: 1.45rem; color: #F8FAFC; font-weight: 800; margin: .45rem 0;}
.decision-value {font-size: 1.05rem; color: #7DD3FC; font-weight: 700;}
.decision-reason {font-size: .84rem; color: #94A3B8; margin-top: .45rem; line-height: 1.35;}
@media (max-width: 900px) {.decision-grid {grid-template-columns: 1fr;}}
```

- [ ] **Step 2: Add a safe card HTML helper**

```python
def _decision_card(css_class: str, award: str, model: str, value: str, reason: str) -> str:
    return f"""
    <div class="decision-card {css_class}">
      <div class="decision-award">{escape(award)}</div>
      <div class="decision-model">{escape(model)}</div>
      <div class="decision-value">{escape(value)}</div>
      <div class="decision-reason">{escape(reason)}</div>
    </div>
    """
```

- [ ] **Step 3: Add `render_decision_center`**

```python
def render_decision_center(comparison: pd.DataFrame) -> None:
    st.subheader("Decision center")
    if comparison.empty:
        st.info("Complete model packages are required before winners can be selected.")
        return

    winners = select_dashboard_winners(comparison)
    validation = winners["validation"]
    official = winners["official"]
    speed = winners["speed"]
    prediction = winners["prediction"]
    cards: list[str] = []

    if validation is not None:
        cards.append(_decision_card(
            "gold", "🏆 Recommended model", str(validation["model"]),
            f"Validation RMSE {validation['validation_rmse']:.2f} cycles",
            "Lowest validation error—the correct result to use for model selection.",
        ))
    if official is None:
        cards.append(_decision_card(
            "purple", "🎯 Official-test leader", "Not available",
            "No official-test metric",
            "Test performance will appear when a complete result is saved.",
        ))
    else:
        cards.append(_decision_card(
            "purple", "🎯 Official-test leader", str(official["model"]),
            f"Official-test RMSE {official['test_rmse']:.2f} cycles",
            "Lowest reported test error; shown separately to avoid test-driven selection.",
        ))
    if speed is not None:
        prediction_note = ""
        if prediction is not None and prediction["model"] != speed["model"]:
            prediction_note = (
                f" Prediction is fastest with {prediction['model']} "
                f"({prediction['predict_seconds']:.4f}s)."
            )
        cards.append(_decision_card(
            "cyan", "⚡ Speed champion", str(speed["model"]),
            f"Train {speed['train_seconds']:.3f}s · predict {speed['predict_seconds']:.4f}s",
            "Lowest saved training time." + prediction_note,
        ))

    st.markdown(
        '<div class="decision-grid">' + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Choose models using validation performance. Official-test results report "
        "generalization; speed describes computational cost."
    )
```

- [ ] **Step 4: Compile the renderer**

```powershell
conda run --no-capture-output -n nti-cmapss python -B -m py_compile main.py
```

Expected: exit code 0.

### Task 3: Integrate winners into both tabs

**Files:**
- Modify: `main.py` in `render_overview` and `render_comparison`

- [ ] **Step 1: Call `render_decision_center(comparison)` first in Overview**

Delete the old two-column validation/test message block so the conclusion is
not duplicated.

- [ ] **Step 2: Call `render_decision_center(comparison)` after the empty guard in Compare models**

Keep the existing note explaining that the table ranks by validation RMSE.

- [ ] **Step 3: Add the leaderboard award column**

```python
awards = award_labels(comparison)
leaderboard_source = comparison.copy()
leaderboard_source.insert(
    1,
    "award",
    leaderboard_source["model"].map(awards).fillna(""),
)
leaderboard = leaderboard_source[
    [
        "rank",
        "award",
        "model",
        "device",
        "validation_rmse",
        "validation_mae",
        "validation_r2",
        "validation_nasa",
        "test_rmse",
        "test_mae",
        "test_r2",
        "test_nasa",
    ]
].copy()
```

- [ ] **Step 4: Verify the inference-only boundary**

```powershell
rg -n "\.fit\(|train_ridge|train_random_forest|cmapss_rul\.data" main.py
```

Expected: no matches.

- [ ] **Step 5: Commit the feature**

```powershell
git add main.py
git commit -m "feat: highlight model winners in dashboard"
```

### Task 4: Verify the finished interface

**Files:**
- Verify: `main.py`
- Verify: `.streamlit/config.toml`

- [ ] **Step 1: Run Streamlit AppTest**

```powershell
@'
from streamlit.testing.v1 import AppTest

app = AppTest.from_file("main.py", default_timeout=120)
app.run()
assert not app.exception, app.exception
assert any("4/4 model packages ready" in str(item.value) for item in app.caption)
markdown = "\n".join(str(item.value) for item in app.markdown)
assert "Recommended model" in markdown and "Ridge Regression" in markdown
assert "Official-test leader" in markdown and "CatBoost" in markdown
assert "Speed champion" in markdown
print("winner dashboard AppTest passed")
'@ | conda run --no-capture-output -n nti-cmapss python -B -
```

Expected: `winner dashboard AppTest passed` and no exceptions.

- [ ] **Step 2: Run final checks**

```powershell
conda run --no-capture-output -n nti-cmapss python -B -m compileall -q main.py scripts src
git diff --check
git status --short --branch
```

Expected: compilation and diff check exit 0; only intentional plan progress may
remain.

- [ ] **Step 3: Inspect locally**

```powershell
conda activate nti-cmapss
python -m streamlit run main.py
```

Expected: Overview and Compare models show Ridge as Recommended and Fastest,
CatBoost as Official-test leader, and award badges in the leaderboard without
retraining any model.

- [ ] **Step 4: Commit completed plan state if checkboxes were updated**

```powershell
git add docs/superpowers/plans/2026-09-16-dashboard-winner-cards.md
git commit -m "docs: complete dashboard winner plan"
```

Do not push, merge, or deploy unless the user explicitly requests it.

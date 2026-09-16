# Dashboard Winner Cards Design

## Goal

Make the Streamlit dashboard immediately answer three different questions:

1. Which model should be selected using validation data?
2. Which model has the lowest official-test error?
3. Which model is fastest to train and use?

The interface must not hide these distinctions behind a single artificial
score. In particular, official-test performance remains descriptive and does
not replace validation performance as the model-selection rule.

## Decision Rules

The dashboard derives all winners from the existing comparison dataframe:

- **Recommended model:** smallest finite `validation_rmse`. This is the primary
  scientific selection because validation data is intended for model choice.
- **Official-test leader:** smallest finite `test_rmse`. This card is labelled
  as a reported result, not the selection criterion.
- **Speed champion:** smallest finite `train_seconds`. Its card also reports
  `predict_seconds`. If another model has the lowest prediction time, that fact
  is shown in a short note rather than hidden.

Ties are resolved deterministically by model name. A missing metric cannot win.
If no official-test metric is available, the official-test card displays an
unavailable state instead of failing the page.

With the current saved results, Ridge is the recommended model and speed
champion, while CatBoost is the official-test leader.

## Interface

Add a **Decision center** near the top of the Overview tab, before dataset
details. It contains three visually distinct cards:

- a gold trophy card for the recommended validation winner;
- a purple target card for the official-test leader;
- a cyan lightning card for the speed champion.

Each card contains the award, model name, winning value, and a one-sentence
reason. A short note below the cards explains that validation chooses the model
and the test set reports final generalization.

The Model Comparison tab repeats the same decision center so a viewer who opens
that tab directly receives the conclusion before the detailed charts. Its
leaderboard gains an `award` column. A model can hold more than one award, so
Ridge can display both the trophy and lightning badges.

The existing validation/test success messages in the Overview are removed to
avoid presenting the same conclusion twice.

## Code Structure

Keep the change inside `main.py` and split it into small helpers:

- `select_dashboard_winners(comparison)` performs deterministic, finite-value
  winner selection and returns the selected rows.
- `render_decision_center(comparison)` renders the three cards and explanatory
  note.
- `award_labels(comparison)` maps each model to one or more compact badges for
  the leaderboard.

The helpers depend only on the existing comparison dataframe. They do not load
models, retrain models, or modify saved artifacts.

## Verification

- Compile `main.py`.
- Unit-smoke the selection helper against the saved four-model comparison and
  assert Ridge wins validation and speed while CatBoost wins official test.
- Exercise empty, missing-test, and non-finite metric inputs without an
  exception.
- Run Streamlit AppTest and require zero exceptions, `4/4 model packages ready`,
  and visible Decision center winner text.
- Confirm `main.py` still contains no `.fit()` call or training-script import.

## Scope

This change improves comparison clarity only. It does not retrain models,
change metrics, invent a weighted overall score, merge the open pull request,
or deploy the dashboard.

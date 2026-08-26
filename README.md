# Linear Regression — Height Prediction from Weight

A learning project that walks through a full linear regression workflow with
[scikit-learn](https://scikit-learn.org/): load data, check assumptions, train an Ordinary
Least Squares model, predict, evaluate, deploy the trained model to disk (`.pkl`), and load
it back from an independent client — predicting a person's **height** from their **weight**.

## Overview

```mermaid
flowchart LR
    A[("resources/\nweight-height.csv")] --> B["linear_regression.main\n(training script)"]

    subgraph B["linear_regression.main — training"]
        direction TB
        B1["Load & inspect data"] --> B2["Prepare data\n(ft/lb → cm/kg)"]
        B2 --> B3["Check assumptions\n(normality, linearity)"]
        B3 --> B4["Train OLS\nLinearRegression"]
        B4 --> B5["Predict & evaluate\n(mean error)"]
    end

    B5 --> C[("resources/\nheight_prediction.pkl")]
    C --> D["height_predictor.main\n(client script)"]
    D --> E(["Predicted height (cm)\nfor a given weight (kg)"])
```

### Ordinary Least Squares — Linear Regression

The model fits a straight line `y = mx + c` (`height_cm = m · weight_kg + c`) through the
training data. Ordinary Least Squares (OLS) chooses the `m` (slope) and `c` (intercept) that
minimize the **sum of squared residuals** — the squared vertical distance between each actual
point and the line's prediction:

```
minimize  Σ (yᵢ - ŷᵢ)²   where  ŷᵢ = m·xᵢ + c
```

Squaring the residuals penalizes larger errors more heavily and keeps positive/negative
errors from cancelling out, giving a single line with a closed-form solution — no iterative
training loop needed. This is what `sklearn.linear_model.LinearRegression.fit()` computes in
step 6 of the workflow below, and it's the same fit whose residuals (`error = y - y_pred`)
are inspected in the evaluation step.

## Project structure

```
resources/
  weight-height.csv        # sample dataset (Gender, Height in inches, Weight in pounds)
  height_prediction.pkl    # trained model, deployed by linear_regression.main
src/
  linear_regression/
    __init__.py             # main() — trains, evaluates, and deploys the model to .pkl
  height_predictor/
    __init__.py             # main() — loads the .pkl model and predicts from new weights
  utils/
    __init__.py             # pretty_log / log_title console helpers
```

## Workflow (`linear_regression.main`)

1. **Load the data** — reads `resources/weight-height.csv` into a pandas DataFrame.
2. **Data understanding** — inspects shape, null counts, and dtypes.
3. **Data preparation**
   - Drops `Gender` (not numerical, not suited for linear regression).
   - Converts `Height` (ft) → `height_cm` and `Weight` (lb) → `weight_kg`.
4. **Assumption tests**
   - A1. Normality — histogram of weight, follows a bell curve. ✅
   - A2. Linearity — scatter plot of weight vs. height. ✅
   - A3. Multicollinearity — skipped (single feature, no other columns to correlate).
   - A4. Autoregression — skipped (single feature vs. single target).
   - Homoscedasticity and zero residual mean are checked during model evaluation instead.
5. **Model building** — `X = weight_kg`, `y = height_cm`.
6. **Model training** — `sklearn.linear_model.LinearRegression` fit via Ordinary Least Squares,
   producing the deliverables of `y = mx + c`: the slope/coefficient (`m`) and intercept (`c`).
7. **Prediction** — automatic prediction of `y_pred` for the full input set (a manual,
   point-by-point prediction path is sketched out but currently commented out).
8. **Model evaluation** — compares actual vs. predicted height and reports the mean error rate.
9. **Model deployment** — pickles the trained `LinearRegression` model to
   `resources/height_prediction.pkl`.

All steps print through `utils.pretty_log` / `utils.log_title` for readable, boxed console output.

## Serving predictions (`height_predictor.main`)

A separate, lightweight client loads the deployed `.pkl` model and predicts on new weights
without needing to retrain — the split between `linear_regression` (train + deploy) and
`height_predictor` (load + predict) mirrors a real train/serve boundary.

## Requirements

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/) for dependency management
- Dependencies (see `pyproject.toml`): `pandas`, `matplotlib`, `scikit-learn`

## Running it

```bash
uv sync
uv run linear-regression   # or: uv run hwp — trains, evaluates, and deploys the model
uv run hp                  # loads the deployed .pkl model and predicts a height
```

`linear-regression` / `hwp` and `hp` are registered console scripts that call
`linear_regression.main` and `height_predictor.main` respectively.

## Status

Complete — data prep, assumption checks, model training, prediction, evaluation, and
deployment to `.pkl` are all implemented, along with an independent client that loads the
deployed model and serves predictions.

---

### Troubleshooting:

> [!WARN]
> X doesn't have valid feature names:
>
> `sklearn/utils/validation.py:2827: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names`
>
#  Bike Sharing Demand Forecasting
## Full process documentation, from data cleaning to model training

**Models:** Ridge Regression and Random Forest Regressor
**Dataset:** UCI Bike Sharing (`hour.csv`, 17,389 hourly observations, 2011-2012)
**Target:** `cnt` (total number of bike rentals per hour)
**Goal:** predict the demand of a **future period** from the past, and measure how the error changes by hour of the day.

---

## 0. The big picture

```
hour.csv
   │
   ▼
1. Load + sort by date and hour
   │
   ▼
2. Inspect and explore (missing values, trends, hourly profile)
   │
   ▼
3. Build X and y (remove leaking columns, encode periodic variables)
   │
   ▼
4. Chronological split: first 80% = train, last 20% = test
   │
   ▼
5. Data pipeline (scaling + one-hot encoding) inside a model pipeline
   │
   ▼
6. Hyperparameter tuning with TimeSeriesSplit (train only)
   │
   ▼
7. Final training of the best models on the whole train set
   │
   ▼
8. One evaluation on test: MAE, RMSE, R2 + error by hour
   │
   ▼
9. Save the trained pipelines
```

The guiding rule of the whole project is: **never let information from the future reach the model during training or tuning.** Almost every step below exists to protect that rule.

---

## 1. Imports and settings

```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
RS = 42
```

**What it does:** loads the tools we need and sets a random seed `RS = 42`.

**Why:**
- Random Forest uses randomness (random subsets of rows and features). A fixed seed makes the results **reproducible**: running the notebook again gives the same numbers. The project explicitly asks for a reproducible notebook.
- `Pipeline` and `ColumnTransformer` let us chain preprocessing and the model into one object, which is what prevents leakage during cross-validation (see section 6).

---

## 2. Loading and sorting the data

```python
df = pd.read_csv("hour.csv", parse_dates=["dteday"])
df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
```

**What it does:**
- Reads the CSV and converts `dteday` into a real date type.
- Sorts the rows by date, then by hour, and resets the index to 0, 1, 2, ...

**Why:**
- The problem is a forecasting problem, so **row order = time order**. All later steps (the 80/20 split and `TimeSeriesSplit`) cut the data by position, so the order must be exactly chronological.
- `parse_dates` makes it possible to sort and resample by date properly.

### Variables of the dataset

| Variable | Meaning | Type |
|---|---|---|
| `instant` | Row identifier | ID (removed) |
| `dteday` | Date | Date (removed after sorting) |
| `season` | 1 spring, 2 summer, 3 fall, 4 winter | Categorical |
| `yr` | 0 = 2011, 1 = 2012 | Binary |
| `mnth` | Month, 1-12 | Periodic |
| `hr` | Hour, 0-23 | Periodic |
| `holiday` | Public holiday or not | Binary |
| `weekday` | Day of the week, 0-6 | Periodic |
| `workingday` | Working day (not weekend, not holiday) | Binary |
| `weathersit` | Weather type, 1 clear to 4 heavy rain/snow | Categorical |
| `temp`, `atemp` | Temperature and "feels like" temperature (normalized) | Numeric |
| `hum` | Humidity (normalized) | Numeric |
| `windspeed` | Wind speed (normalized) | Numeric |
| `casual`, `registered` | Rentals by casual / registered users | Leaking (removed) |
| `cnt` | Total rentals = `casual + registered` | **Target** |

---

## 3. Data inspection (cleaning checks)

```python
df.info()
df.describe()
df.isna().sum()
```

**What it does:** checks data types, value ranges and missing values.

**Why:** before modeling, we must know whether the data needs cleaning. For this dataset:
- There are **no missing values**, so no imputation is needed.
- The variables are already normalized or encoded by the dataset authors, so little cleaning is required. The main "cleaning" work is therefore choosing which columns are allowed as inputs (section 5) and encoding them correctly.
- One known property: **some hours are missing from the file** (hours with zero rentals do not appear as rows), which is why there are 17,389 rows instead of the 17,544 hours of two years. This should be mentioned in the report as a data limitation. It also means "the next row" is not always "the next hour", which is one more reason not to build lag features with simple row shifts.

---

## 4. Exploratory data analysis (EDA)

```python
# Weekly mean demand
df.set_index("dteday")["cnt"].resample("W").mean().plot()

# Average profile by hour, working day vs weekend
df.groupby(["workingday", "hr"])["cnt"].mean().unstack(0).plot()
```

**What it does:** draws (1) the evolution of demand over time and (2) the average demand for each hour of the day.

**What you should see and why it matters:**

| Observation | Consequence for modeling |
|---|---|
| Demand is higher in 2012 than in 2011 (growth trend) | The test period (late 2012) has higher demand than most of the train period. Tree models cannot predict above the values they saw in training, so expect some under-prediction. This is a key point for the discussion. |
| Strong yearly seasonality (more rentals in warm months) | `mnth`, `season` and `temp` are useful features. |
| Two peaks on working days (about 8h and 17-18h) and a single midday hump on weekends | The effect of `hr` depends on `workingday` (an **interaction**). Linear models cannot represent this easily, trees can. Peak hours are also where the largest errors are expected. |
| `temp` and `atemp` are almost identical | **Multicollinearity**, which affects the stability of linear models. Ridge handles it well. |

EDA is also the first of the required visualizations (time evolution and hourly profile).

---

## 5. Feature preparation

```python
y = df["cnt"]
X = df.drop(columns=["cnt", "casual", "registered", "instant", "dteday"])
```

### 5.1 Removing columns

| Column removed | Why |
|---|---|
| `cnt` | It is the target. Keeping it in X would give the answer to the model. |
| `casual`, `registered` | Their sum is exactly `cnt`. They would be a perfect leak: the model would just add them up and get a near-perfect score that is useless in practice. |
| `instant` | A row counter. It carries no real information and would simply act as a time index that the model could misuse. |
| `dteday` | The date was used for sorting. Its useful parts (year, month, weekday) are already in other columns. |

### 5.2 Cyclical encoding of periodic variables

```python
for col, period in [("hr", 24), ("mnth", 12), ("weekday", 7)]:
    X[col + "_sin"] = np.sin(2 * np.pi * X[col] / period)
    X[col + "_cos"] = np.cos(2 * np.pi * X[col] / period)

X = X.drop(columns=["hr", "mnth", "weekday"])
```

**What it does:** replaces each periodic variable by two columns, a sine and a cosine, which place the values on a circle.

**Why:**
- With the raw number, hour 23 and hour 0 look far apart (23 units), although they are neighbors in reality. Same for December (12) and January (1), and Sunday (6) and Monday (0).
- On a circle, 23h and 0h are close. This is especially important for a **linear model such as Ridge**, which otherwise treats the hour as a straight line from 0 to 23.
- Two columns (sin and cos) are needed because sine alone gives the same value for two different times (for example 6h and 12h).
- We drop the raw columns because the sin/cos pair already contains the same information.
- We keep the raw hour in a separate variable (`hr_test`) for the error analysis by hour.

### 5.3 Column groups

```python
cat_cols = ["season", "weathersit"]                      # categories without a numeric order
num_cols = ["temp", "atemp", "hum", "windspeed"]         # continuous
bin_cols = ["yr", "holiday", "workingday"]               # already 0/1
cyc_cols = [c for c in X.columns if c.endswith(("_sin", "_cos"))]
```

**Why groups:** each group needs a different treatment in the pipeline (section 7). `season` and `weathersit` are numbers 1-4 but they are **labels**, not quantities: "season 4" is not twice "season 2". So they get one-hot encoding.

---

## 6. Chronological train/test split

```python
n = int(len(X) * 0.8)
X_train, X_test = X.iloc[:n], X.iloc[n:]
y_train, y_test = y.iloc[:n], y.iloc[n:]
hr_test = df["hr"].iloc[n:]
```

**What it does:** the first 80% of rows (earliest dates) become the training set. The last 20% (latest dates, roughly from August to December 2012) become the test set. **No shuffling.**

**Why no shuffle:**
1. **Future leakage.** Shuffling would put October 2012 rows in the train set and June 2012 rows in the test set, so the model would "predict the past from the future".
2. **Autocorrelation.** Consecutive hours are very similar (same weather, same day). With shuffling, every test hour has its neighbors in the train set, and the model can interpolate instead of forecasting. Scores would look excellent but be unrealistic.
3. **It matches the real task.** The business question is to forecast a later period, so the test set must be a later period.

The test set is then **locked**: it is used one time, at the very end.

---

## 7. The data pipeline

```python
def build_preprocessor():
    return ColumnTransformer([
        ("num",  StandardScaler(), num_cols),
        ("cat",  OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("pass", "passthrough", bin_cols + cyc_cols),
    ])

pipe_ridge = Pipeline([("pre", build_preprocessor()), ("m", Ridge(random_state=RS))])
pipe_rf    = Pipeline([("pre", build_preprocessor()), ("m", RandomForestRegressor(random_state=RS, n_jobs=-1))])
```

### What each part does

| Step | Applied to | What it does | Why |
|---|---|---|---|
| `StandardScaler` | `temp`, `atemp`, `hum`, `windspeed` | Centers each column to mean 0 and standard deviation 1 | Ridge penalizes the size of the coefficients, so variables on different scales would be penalized unequally. Scaling makes the penalty fair. Random Forest does not need it, but it does no harm. |
| `OneHotEncoder` | `season`, `weathersit` | Creates one 0/1 column per category | Removes the false numeric order between categories. `handle_unknown="ignore"` avoids a crash if a category appears that was not seen in training. |
| `passthrough` | binary and cyclic columns | Left unchanged | They are already in a good range (0/1 or -1 to 1). |

### Why a Pipeline (and not transforming the data beforehand)

If we scaled the whole dataset before splitting, the scaler would compute its mean and standard deviation using **test data too**. That is a subtle form of **data leakage**.

With a `Pipeline`:
- During cross-validation, the scaler and encoder are **re-fitted on each training fold only**, then applied to the validation fold.
- At the end, preprocessing and model are saved together as one object, so there is no risk of applying the wrong preprocessing at prediction time.

---

## 8. Cross-validation with `TimeSeriesSplit`

### 8.1 Why we need cross-validation

We have hyperparameters to choose (`alpha`, `max_depth`, ...). To compare values we must evaluate each one on data the model has not trained on, but we **cannot use the test set** for this. So we create validation data **inside the training set**. One single validation split is unreliable (the result depends on which part you picked), so we repeat the process several times and average the scores. This is cross-validation.

### 8.2 Why not a normal K-fold

A standard K-fold lets some folds train on later data and validate on earlier data (training on the future to predict the past). It also shuffles or mixes neighbors. Both give over-optimistic scores for time series.

### 8.3 How `TimeSeriesSplit(n_splits=5)` works

The training set is cut into **6 consecutive blocks**. Each fold validates on the next block and trains on everything before it:

```
Fold 1: train [1]            -> validate [2]
Fold 2: train [1 2]          -> validate [3]
Fold 3: train [1 2 3]        -> validate [4]
Fold 4: train [1 2 3 4]      -> validate [5]
Fold 5: train [1 2 3 4 5]    -> validate [6]
```

- The validation block is **always after** the training block.
- The training window **grows** at each fold (expanding window).
- Early folds have less training data, so they usually score worse. This is normal.
- Each fold mimics the real situation: "I have data up to a date, I predict what comes next."
- The final CV score of one hyperparameter setting is the **average of the 5 fold scores**.

```python
tscv = TimeSeriesSplit(n_splits=5)
```

---

## 9. Hyperparameter tuning

```python
param_grids = {
    "Ridge":         {"m__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
    "Random Forest": {"m__n_estimators": [100, 300], "m__max_depth": [8, 12, 16, None]},
}

gs = GridSearchCV(pipe, param_grids[name], cv=tscv,
                  scoring="neg_root_mean_squared_error",
                  return_train_score=True, n_jobs=-1, refit=True)
gs.fit(X_train, y_train)
```

### The hyperparameters

| Model | Hyperparameter | What it controls | Risk if wrong |
|---|---|---|---|
| Ridge | `alpha` | Strength of the penalty on the coefficients. Large alpha means a simpler, smoother model. | Too small: unstable coefficients (overfitting). Too large: underfitting. |
| Random Forest | `n_estimators` | Number of trees. More trees give a more stable average. | Few trees: noisy. Many trees: slower, with diminishing returns. |
| Random Forest | `max_depth` | Maximum depth of each tree. | Too deep: trees memorize the training data (overfitting). Too shallow: underfitting. |

### Key parameters of `GridSearchCV`

| Parameter | Meaning |
|---|---|
| `cv=tscv` | Uses the time-ordered folds from section 8. |
| `scoring="neg_root_mean_squared_error"` | The metric to optimize. RMSE is used because it penalizes large errors strongly, which matters for peak-hour demand. It is negative because scikit-learn always **maximizes** scores, so the error is negated. |
| `return_train_score=True` | Also stores the training score of each fold, so we can plot train vs validation and detect overfitting. |
| `refit=True` | After choosing the best parameters, retrains that model on the **whole training set**. |
| `n_jobs=-1` | Uses all CPU cores. |

### What happens inside

For each parameter combination, `GridSearchCV` runs the 5 folds, averages the validation RMSE, and keeps the combination with the lowest value. For Random Forest that is 2 × 4 = 8 combinations × 5 folds = 40 trainings.

**Important:** all of this happens on the training set only. The test set has not been touched.

---

## 10. Training graphs: how to read them

The notebook draws:

1. **Ridge: RMSE vs `alpha`** (train and validation curves).
2. **Random Forest: RMSE vs `max_depth`** for each `n_estimators`.
3. **Validation RMSE per fold** for the best configuration.

**Why:** the graphs justify the hyperparameter choice and reveal overfitting.

| What you see | Meaning |
|---|---|
| Train error very low, validation error much higher | Overfitting (typical for deep Random Forest trees). |
| Both errors high and close | Underfitting (typical for a linear model on this non-linear problem). |
| Validation error lower in later folds | More training data helps, and fold 5 is the closest to the real situation. |
| Validation error flat across a range of values | The model is not sensitive to this hyperparameter there. |

---

## 11. Final training

Because of `refit=True`, the best model of each search is **automatically retrained on the full training set** (all 80%), using the best hyperparameters. These are the models that are evaluated on the test set and saved.

**Why retrain on all training data:** during CV, each model saw only part of the training set. The final model should use all the available past data, which usually gives a better forecast.

---

## 12. Final evaluation on the test set

```python
p = np.clip(model.predict(X_test), 0, None)

MAE  = mean_absolute_error(y_test, p)
RMSE = np.sqrt(mean_squared_error(y_test, p))
R2   = r2_score(y_test, p)
```

**Why a single evaluation:** if we tuned the model after looking at test scores, the test set would become part of the training process, and the final score would no longer be an honest estimate of future performance. So: predict once, report, do not retune.

**Why clip at 0:** the number of rentals cannot be negative, but a linear model such as Ridge can output negative values.

### The three metrics

| Metric | Formula idea | Unit | How to read it |
|---|---|---|---|
| **MAE** | Mean of the absolute errors | Rentals per hour | "On average the prediction is off by X bikes." Easy to interpret, not dominated by large errors. |
| **RMSE** | Square root of the mean squared error | Rentals per hour | Like MAE but punishes large errors more. If RMSE is much larger than MAE, there are some big misses (usually at peak hours). |
| **R²** | 1 − (model error / error of always predicting the mean) | No unit, up to 1 | Share of the variance explained. 1 = perfect, 0 = no better than predicting the average, negative = worse than the average. |

We report **train and test** metrics together: a large gap (good train, bad test) signals overfitting, or a distribution shift caused by the demand growth between 2011 and 2012.

---

## 13. Error analysis by hour

```python
err = pd.DataFrame({"hr": hr_test.values, "y": y_test.values})
for name, p in preds.items():
    err[name] = np.abs(err["y"] - p)
mae_by_hour = err.groupby("hr")[list(preds)].mean()
```

**Why:** a global score hides *where* the model fails. The project asks to analyze errors at peak hours. We compute the MAE for each hour of the day and for time bands (night, morning peak, midday, evening peak, evening).

**How to interpret:**
- Absolute errors are usually largest at **8h and 17-18h**, because demand is highest and most variable there.
- Night errors look small simply because demand is low. To compare fairly, divide the error by the average demand of that hour (relative error).
- This analysis shows whether a model that looks good globally still fails at the hours that matter most to the operator (bike availability at peak times).

Other plots: actual vs predicted over two weeks, residuals over time (to see drift or under-prediction due to growth), and predicted vs actual scatter.

---

## 14. Saving the models

```python
import joblib
joblib.dump(best_models["Ridge"],         "saved_models/ridge_pipeline.joblib")
joblib.dump(best_models["Random Forest"], "saved_models/random_forest_pipeline.joblib", compress=3)
joblib.dump(build_preprocessor().fit(X_train), "saved_models/data_preprocessing_pipeline.joblib")
```

**Why save the whole pipeline and not only the model:** the saved object contains the fitted scaler, encoder and model together. At prediction time you give it a feature table and it applies exactly the same transformations as in training.

**Why `compress=3`:** Random Forest files can become very large (hundreds of trees).

**Reusing a saved model:**

```python
model = joblib.load("saved_models/random_forest_pipeline.joblib")
X_new = make_features(new_raw_rows)     # same feature function as in training
model.predict(X_new)
```

The function `make_features` (column removal and sin/cos encoding) must be kept with the model, because the pipeline expects its output.

---

## 15. Limitations to discuss in the report

1. **Weather availability.** The model uses *observed* weather (`temp`, `hum`, `windspeed`, `weathersit`). In real forecasting, only weather *forecasts* are available, and they contain errors. Scores here are therefore optimistic. A useful experiment: retrain without weather features and measure how much performance drops.
2. **Distribution shift.** Demand grew from 2011 to 2012. Random Forest cannot predict values far above what it saw in training, so it tends to under-predict during high-demand periods in the test set.
3. **No lag features.** Past demand (for example the previous hour) is not used, because for a long forecasting horizon it would not be available. This keeps the task realistic but limits accuracy.
4. **Missing hours** in the file (hours with zero rentals are absent).
5. **Unmodeled events** such as special events, strikes or unusual holidays.
6. **Only two models** were trained here. A more complete comparison could add Linear Regression, Decision Tree and Gradient Boosting, as listed in the project brief.

---

## 16. Common mistakes avoided

| Mistake | Why it is wrong | How this project avoids it |
|---|---|---|
| Shuffling before splitting | Future leaks into training | Chronological split, no shuffle |
| Normal K-fold on time data | Trains on the future, validates on the past | `TimeSeriesSplit` |
| Scaling before splitting | Test statistics leak into the scaler | Preprocessing inside the `Pipeline` |
| Keeping `casual` / `registered` | They sum exactly to the target | Removed from X |
| Tuning on the test set | Test score becomes biased | Tuning only with CV on train, one test evaluation |
| Reporting only global scores | Hides the failure at peak hours | Error analysis by hour and time band |

---

## 17. Summary: the reason behind each step

| Step | Purpose |
|---|---|
| Sort by date and hour | Make row order equal time order |
| Remove `cnt`, `casual`, `registered`, `instant` | Prevent target leakage |
| sin/cos encoding | Make 23h close to 0h for the model |
| One-hot encoding | Remove false order in categories |
| Standard scaling | Fair penalty for Ridge |
| Chronological 80/20 split | Test on a real future period |
| Pipeline | Keep preprocessing inside CV and prevent leakage |
| `TimeSeriesSplit` | Validate only forward in time |
| `GridSearchCV` | Choose hyperparameters objectively |
| Single test evaluation | Honest estimate of future performance |
| MAE, RMSE, R² | Three complementary views of the error |
| Hourly error analysis | Find where the model fails |
| Save the full pipeline | Reuse the model without mistakes |
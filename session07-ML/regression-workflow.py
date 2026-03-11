# =============================================================================
# COMPLETE REGRESSION WORKFLOW — PRODUCTION TEMPLATE
# =============================================================================

import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_validate,
    GridSearchCV,
)
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# HELPER FUNCTIONS & SHARED COLOUR PALETTE
# =============================================================================
CLR_DARK = "#2c3e50"
CLR_BLUE = "#3498db"
CLR_GREEN = "#2ecc71"
CLR_RED = "#e74c3c"


def plot_regression_scatter(ax, y_true, y_pred, title):
    """Plots predicted vs actual values with R² and error metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Perfect prediction line
    min_val, max_val = (
        min(np.min(y_true), np.min(y_pred)),
        max(np.max(y_true), np.max(y_pred)),
    )
    ax.plot(
        [min_val, max_val], [min_val, max_val], "k--", lw=2, label="Perfect Prediction"
    )

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, color=CLR_BLUE)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)

    # Metrics text
    metrics_text = f"R²={r2:.3f}\nMAE={mae:.3f}\nRMSE={rmse:.3f}"
    ax.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_residuals(ax, y_true, y_pred, title):
    """Plots residuals vs predicted values."""
    residuals = y_true - y_pred
    mae = mean_absolute_error(y_true, y_pred)

    ax.scatter(y_pred, residuals, alpha=0.6, s=30, color=CLR_GREEN)
    ax.axhline(y=0, color="r", linestyle="-", lw=2)
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title(title)

    # Metrics
    ax.text(
        0.05,
        0.95,
        f"MAE={mae:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.grid(True, alpha=0.3)


print("✅ Shared colour palette and helper functions ready.")

# =============================================================================
# 1. LOAD DATA
# =============================================================================
# df = pd.read_csv("your_regression_dataset.csv")  # <-- UPDATE PATH
# For demonstration, creating dummy regression
np.random.seed(42)
n_samples = 1000
df = pd.DataFrame(
    {
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.rand(n_samples) * 100,
        "feature3": np.random.randint(0, 50, n_samples),
        "category1": np.random.choice(
            ["A", "B", "C", "Rare"], n_samples, p=[0.4, 0.4, 0.18, 0.02]
        ),
        "category2": np.random.choice(["X", "Y"], n_samples),
        "target_continuous": 50
        + 10 * np.random.randn(n_samples)
        + 0.5 * np.random.rand(n_samples) * 100,  # Realistic continuous target
    }
)

print("Dataset shape:", df.shape)
print("\nTarget statistics:")
print(df["target_continuous"].describe().round(2))

# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
numeric_cols = (
    df.select_dtypes(include=np.number).columns.drop("target_continuous").tolist()
)
cat_cols = df.select_dtypes(include="object").columns.tolist()

# EDA Visualization (Uncomment to display)
"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['target_continuous'].hist(bins=30, ax=axes[0])
axes[0].set_title('Target Distribution')
sns.heatmap(df[numeric_cols + ['target_continuous']].corr(), annot=True, cmap='coolwarm', ax=axes[1])
plt.tight_layout(); plt.show()
"""

# =============================================================================
# 3. DEFINE MODELS (MODEL ZOO)
# =============================================================================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(random_state=123),
    "Decision Tree": DecisionTreeRegressor(random_state=123),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=123, n_jobs=-1
    ),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "XGBoost": XGBRegressor(random_state=123),
    "LightGBM": LGBMRegressor(random_state=123, verbose=-1),
}
print(f"✅ {len(models)} baseline models ready!")

# =============================================================================
# 4. TRAIN/TEST SPLIT
# =============================================================================
target_col = "target_continuous"  # <-- UPDATE TARGET
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)
print(f"✅ Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")


# =============================================================================
# 5. PREPROCESSING PIPELINE
# =============================================================================
def collapse_rare_categories(df, threshold=0.03):
    """Collapse categories appearing less than threshold to 'Other'"""
    out = df.copy()
    for col in out.columns:
        counts = out[col].value_counts(normalize=True)
        rare = counts[counts < threshold].index
        out[col] = out[col].apply(lambda x: "Other" if x in rare else x)
    return out


rare_collapser = FunctionTransformer(
    lambda X: collapse_rare_categories(pd.DataFrame(X, columns=cat_cols)),
    validate=False,
)

num_pipe = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("zv", VarianceThreshold(threshold=0)),
        ("scale", StandardScaler()),
    ]
)

cat_pipe = Pipeline(
    [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("rare", rare_collapser),
        (
            "ohe",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, cat_cols),
    ]
)
print("✅ Preprocessing pipeline ready!")

# =============================================================================
# 6. 5-FOLD CROSS-VALIDATION (BASELINE MODELS)
# =============================================================================
cv5 = KFold(n_splits=5, shuffle=True, random_state=123)
scoring = {
    "r2": "r2",
    "neg_mae": "neg_mean_absolute_error",
    "neg_rmse": "neg_root_mean_squared_error",
}
cv_results = []

for name, reg in models.items():
    pipe = Pipeline([("pre", preprocessor), ("reg", reg)])
    cv_scores = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv5,
        scoring=scoring,
        n_jobs=-1,
    )
    cv_results.append(
        {
            "Model": name,
            "Mean R²": cv_scores["test_r2"].mean(),
            "Mean MAE": -cv_scores["test_neg_mae"].mean(),
            "Mean RMSE": np.sqrt(-cv_scores["test_neg_rmse"].mean()),
        }
    )

results_df = pd.DataFrame(cv_results).sort_values("Mean R²", ascending=False)
print("\nCV SUMMARY (BASELINE):")
print(results_df.round(3).to_string(index=False))

# =============================================================================
# 7. HYPERPARAMETER TUNING
# =============================================================================
tuning_configs = {
    "Ridge": {
        "model": Ridge(random_state=123),
        "params": {
            "reg__alpha": [0.1, 1.0, 10.0],
        },
    },
    "RandomForest": {
        "model": RandomForestRegressor(random_state=123, n_jobs=-1),
        "params": {
            "reg__n_estimators": [100, 200],
            "reg__max_depth": [5, 10, None],
            "reg__min_samples_split": [2, 5],
        },
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=123),
        "params": {
            "reg__n_estimators": [100, 200],
            "reg__learning_rate": [0.05, 0.1],
            "reg__max_depth": [3, 5],
        },
    },
}

tuning_results, grid_objects = [], {}

print("\n" + "=" * 50 + "\nSTARTING GRID SEARCH\n" + "=" * 50)
for name, config in tuning_configs.items():
    tuning_pipeline = Pipeline([("pre", preprocessor), ("reg", config["model"])])

    grid_search = GridSearchCV(
        tuning_pipeline,
        config["params"],
        cv=cv5,
        scoring="r2",
        n_jobs=-1,
        verbose=0,
    )

    start_t = time.time()
    grid_search.fit(X_train, y_train)
    duration = time.time() - start_t

    grid_objects[name] = grid_search
    tuning_results.append(
        {
            "Model": name,
            "Best R²": grid_search.best_score_,
            "Time(s)": duration,
            "Params": grid_search.best_params_,
        }
    )
    print(
        f"[✓] {name:.<20} Time: {duration:>5.1f}s | Best R²: {grid_search.best_score_:.4f}"
    )

best_model_name = sorted(tuning_results, key=lambda x: x["Best R²"], reverse=True)[0][
    "Model"
]
best_pipeline = grid_objects[best_model_name].best_estimator_

# =============================================================================
# 8. FINAL TEST EVALUATION
# =============================================================================
print("\n" + "=" * 50 + "\nTEST SET RESULTS (BEST TUNED MODEL)\n" + "=" * 50)

y_pred = best_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"Model Evaluated: {best_model_name}")
print(f"R²:        {r2:.3f}")
print(f"MAE:       {mae:.3f}")
print(f"RMSE:      {rmse:.3f}")
print(f"MAPE:      {mape:.1%}")


# =============================================================================
# 9. FEATURE IMPORTANCE EXTRACTION & PLOT
# =============================================================================
def plot_feature_importance(pipeline, top_n=15):
    """Extracts and plots feature importances from a fitted sklearn Pipeline."""
    pre = pipeline.named_steps["pre"]
    reg = pipeline.named_steps["reg"]

    try:
        feature_names = pre.get_feature_names_out()
    except AttributeError:
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

    if hasattr(reg, "feature_importances_"):
        importances = reg.feature_importances_
        metric_name = "Feature Importance"
    elif hasattr(reg, "coef_"):
        importances = np.abs(reg.coef_)
        metric_name = "Absolute Coefficient"
    else:
        print(f"⚠️ Feature importance not supported for {reg.__class__.__name__}")
        return

    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df_imp["Feature"] = df_imp["Feature"].str.split("__").str[-1]
    df_imp = df_imp.sort_values("Importance", ascending=True).tail(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(df_imp["Feature"], df_imp["Importance"], color=CLR_BLUE)
    plt.title(
        f"Top {top_n} {metric_name}s ({reg.__class__.__name__})", fontweight="bold"
    )
    plt.xlabel(metric_name)
    plt.tight_layout()
    plt.show()


plot_feature_importance(best_pipeline, top_n=10)

print("\n✅ COMPLETE REGRESSION PIPELINE EXECUTED SUCCESSFULLY!")

# =============================================================================
# 10. VISUALISE RESULTS FOR ALL MODELS
# =============================================================================
print("\n" + "=" * 50)
print("GENERATING PLOTS FOR ALL FITTED MODELS")
print("=" * 50)

num_models = len(models)
fig, axes = plt.subplots(nrows=num_models, ncols=2, figsize=(14, 5 * num_models))
if num_models == 1:
    axes = [axes]

for i, (name, reg) in enumerate(models.items()):
    pipe = Pipeline([("pre", preprocessor), ("reg", reg)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    # Left: Predicted vs Actual
    ax_scatter = axes[i][0]
    plot_regression_scatter(ax_scatter, y_test, y_pred, f"{name} - Predicted vs Actual")

    # Right: Residuals
    ax_residual = axes[i][1]
    plot_residuals(ax_residual, y_test, y_pred, f"{name} - Residuals")

plt.tight_layout(pad=3.0)
plt.show()

# =============================================================================
# 11. COMBINED PREDICTION vs ACTUAL FOR ALL MODELS
# =============================================================================
print("\n" + "=" * 50)
print("GENERATING COMBINED SCATTER PLOT FOR ALL MODELS")
print("=" * 50)

plt.figure(figsize=(12, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
min_val, max_val = (
    min(
        np.min(y_test),
        np.min(
            [
                pipe.predict(X_test)
                for pipe in [
                    Pipeline([("pre", preprocessor), ("reg", reg)]).fit(
                        X_train, y_train
                    )
                    for reg in models.values()
                ]
            ]
        ),
    ),
    max(
        np.max(y_test),
        np.max(
            [
                pipe.predict(X_test)
                for pipe in [
                    Pipeline([("pre", preprocessor), ("reg", reg)]).fit(
                        X_train, y_train
                    )
                    for reg in models.values()
                ]
            ]
        ),
    ),
)
plt.plot(
    [min_val, max_val], [min_val, max_val], "k--", lw=3, label="Perfect Prediction"
)

for (name, reg), color in zip(models.items(), colors):
    pipe = Pipeline([("pre", preprocessor), ("reg", reg)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    plt.scatter(
        y_test, y_pred, alpha=0.7, s=40, color=color, label=f"{name} (R²={r2:.3f})"
    )

plt.xlabel("Actual Values", fontsize=12)
plt.ylabel("Predicted Values", fontsize=12)
plt.title(
    "Model Comparison: Predicted vs Actual (Test Set)", fontweight="bold", fontsize=14
)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n🏆 REGRESSION WORKFLOW COMPLETE!")
print("💡 Update: target_col, CSV path, then RUN!")

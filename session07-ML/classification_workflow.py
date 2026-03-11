# =============================================================================
# COMPLETE CLASSIFICATION WORKFLOW — PRODUCTION TEMPLATE
# =============================================================================

import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
)
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# =============================================================================
# HELPER FUNCTIONS & SHARED COLOUR PALETTE
# =============================================================================
CLR_TP, CLR_TN, CLR_FP, CLR_FN = "#2ecc71", "#3498db", "#e74c3c", "#f39c12"
CLR_CURVE, CLR_DARK = "#8e44ad", "#2c3e50"


def get_outcomes(y_true, y_pred):
    """Return per-observation outcome labels: TP / TN / FP / FN."""
    return np.where(
        y_pred == y_true,
        np.where(y_pred == 1, "TP", "TN"),
        np.where(y_pred == 1, "FP", "FN"),
    )


def plot_confusion_matrix(ax, y_true, y_pred, title, y_proba=None):
    """Plots a customized, color-coded confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    cell_colours = [[CLR_TP, CLR_FN], [CLR_FP, CLR_TN]]
    cell_labels = [[f"TP\n{TP}", f"FN\n{FN}"], [f"FP\n{FP}", f"TN\n{TN}"]]

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    for r in range(2):
        for c in range(2):
            ax.add_patch(
                plt.Rectangle((c, 1 - r), 1, 1, color=cell_colours[r][c], alpha=0.78)
            )
            ax.text(
                c + 0.5,
                1 - r + 0.5,
                cell_labels[r][c],
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color="white",
            )

    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted: (1)", "Predicted: (0)"], fontsize=9)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Actual: (0)", "Actual: (1)"], fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / len(y_true)

    xlabel = f"Accuracy={accuracy:.2f}  Precision={precision:.2f}  Recall={recall:.2f}"
    if y_proba is not None:
        xlabel += f"  AUC={roc_auc_score(y_true, y_proba):.3f}"
    ax.set_xlabel(xlabel, fontsize=9, labelpad=8)


def plot_roc_curve(ax, y_true, y_proba, model_name, color=CLR_CURVE):
    """Plots a single ROC curve on existing axes."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{model_name} (AUC={auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.06, color=color)
    return auc


print("✅ Shared colour palette and helper functions ready.")

# =============================================================================
# 1. LOAD DATA
# =============================================================================
# df = pd.read_csv("your_dataset.csv")  # <-- UPDATE PATH
# For demonstration, creating dummy data:
np.random.seed(42)
df = pd.DataFrame(
    {
        "feature1": np.random.randn(1000),
        "feature2": np.random.rand(1000) * 100,
        "feature3": np.random.randint(0, 50, 1000),
        "category1": np.random.choice(
            ["A", "B", "C", "Rare"], 1000, p=[0.4, 0.4, 0.18, 0.02]
        ),
        "category2": np.random.choice(["X", "Y"], 1000),
        "target_column": np.random.choice([0, 1], 1000, p=[0.85, 0.15]),  # Imbalanced
    }
)

print("Dataset shape:", df.shape)
print(
    "\nTarget distribution:\n",
    df["target_column"].value_counts(normalize=True).round(3),
)

# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
numeric_cols = (
    df.select_dtypes(include=np.number).columns.drop("target_column").tolist()
)
cat_cols = df.select_dtypes(include="object").columns.tolist()

# EDA Visualization (Uncomment to display)
"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['target_column'].value_counts().plot(kind='bar', ax=axes[0], title='Target Distribution')
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=axes[1])
plt.tight_layout(); plt.show()
"""

# =============================================================================
# 3. DEFINE MODELS (MODEL ZOO)
# =============================================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=123),
    "Decision Tree": DecisionTreeClassifier(random_state=123),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=123, n_jobs=-1
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=123),
    "LightGBM": LGBMClassifier(random_state=123, verbose=-1),
}
print(f"✅ {len(models)} baseline models ready!")

# =============================================================================
# 4. TRAIN/TEST SPLIT
# =============================================================================
target_col = "target_column"  # <-- UPDATE TARGET
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

# Encode Target
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
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
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
cv_results = []

for name, clf in models.items():
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
    cv_scores = cross_validate(
        pipe,
        X_train,
        y_train_enc,
        cv=cv5,
        scoring=["roc_auc", "f1_weighted"],
        n_jobs=-1,
    )
    cv_results.append(
        {
            "Model": name,
            "Mean AUC": cv_scores["test_roc_auc"].mean(),
            "Mean F1": cv_scores["test_f1_weighted"].mean(),
        }
    )

results_df = pd.DataFrame(cv_results).sort_values("Mean AUC", ascending=False)
print("\nCV SUMMARY (BASELINE):")
print(results_df.round(3).to_string(index=False))

# =============================================================================
# 7. HYPERPARAMETER TUNING
# =============================================================================
# Succinct configuration for GridSearchCV
tuning_configs = {
    "LogisticRegression": {
        "model": LogisticRegression(random_state=123, class_weight="balanced"),
        "params": {
            "clf__C": [0.1, 1, 10],  # Regularization strength
            "clf__penalty": ["l2"],  # Penalty type
            "clf__solver": ["lbfgs"],  # Optimizer
        },
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=123, class_weight="balanced"),
        "params": {
            "clf__n_estimators": [100, 200],  # Number of trees
            "clf__max_depth": [5, 10, None],  # Depth control
            "clf__min_samples_split": [2, 5],  # Split regularization
        },
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=123, eval_metric="logloss"),
        "params": {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [3, 5],
        },
    },
}

tuning_results, grid_objects = [], {}

print("\n" + "=" * 50 + "\nSTARTING GRID SEARCH\n" + "=" * 50)
for name, config in tuning_configs.items():
    tuning_pipeline = Pipeline([("pre", preprocessor), ("clf", config["model"])])

    grid_search = GridSearchCV(
        tuning_pipeline,
        config["params"],
        cv=cv5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )

    start_t = time.time()
    # Note: Use y_train_enc to prevent encoding errors in trees
    grid_search.fit(X_train, y_train_enc)
    duration = time.time() - start_t

    grid_objects[name] = grid_search
    tuning_results.append(
        {
            "Model": name,
            "Best AUC": grid_search.best_score_,
            "Time(s)": duration,
            "Params": grid_search.best_params_,
        }
    )
    print(
        f"[✓] {name:.<20} Time: {duration:>5.1f}s | Best AUC: {grid_search.best_score_:.4f}"
    )

best_model_name = sorted(tuning_results, key=lambda x: x["Best AUC"], reverse=True)[0][
    "Model"
]
best_pipeline = grid_objects[best_model_name].best_estimator_

# =============================================================================
# 8. FINAL TEST EVALUATION
# =============================================================================
print("\n" + "=" * 50 + "\nTEST SET RESULTS (BEST TUNED MODEL)\n" + "=" * 50)

y_pred = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

print(f"Model Evaluated: {best_model_name}")
print(f"Accuracy: {accuracy_score(y_test_enc, y_pred):.3f}")
print(f"ROC-AUC:  {roc_auc_score(y_test_enc, y_proba):.3f}")
print("\nClassification Report:\n", classification_report(y_test_enc, y_pred))


# =============================================================================
# 9. FEATURE IMPORTANCE EXTRACTION & PLOT
# =============================================================================
def plot_feature_importance(pipeline, top_n=15):
    """Extracts and plots feature importances from a fitted sklearn Pipeline."""
    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]

    # 1. Extract feature names post-transformation
    try:
        # Works dynamically for OneHotEncoder, VarianceThreshold, etc.
        feature_names = pre.get_feature_names_out()
    except AttributeError:
        # Fallback for older sklearn versions
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

    # 2. Extract importances based on model type
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        metric_name = "Feature Importance"
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])  # Absolute coefficients for linear models
        metric_name = "Absolute Coefficient"
    else:
        print(f"⚠️ Feature importance not supported for {clf.__class__.__name__}")
        return

    # 3. Create DataFrame and plot
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})

    # Clean up feature names generated by ColumnTransformer (e.g., 'num__feature1' -> 'feature1')
    df_imp["Feature"] = df_imp["Feature"].str.split("__").str[-1]

    df_imp = df_imp.sort_values("Importance", ascending=True).tail(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(df_imp["Feature"], df_imp["Importance"], color=CLR_TN)
    plt.title(
        f"Top {top_n} {metric_name}s ({clf.__class__.__name__})", fontweight="bold"
    )
    plt.xlabel(metric_name)
    plt.tight_layout()
    plt.show()


# Execute Feature Importance Plot
plot_feature_importance(best_pipeline, top_n=10)

print("\n✅ COMPLETE CLASSIFICATION PIPELINE EXECUTED SUCCESSFULLY!")

# =============================================================================
# 10. VISUALISE RESULTS FOR ALL MODELS (CONFUSION MATRIX & ROC CURVES)
# =============================================================================
print("\n" + "=" * 50)
print("GENERATING PLOTS FOR ALL FITTED MODELS")
print("=" * 50)

# Create a figure with 2 columns (CM and ROC) and 1 row per model
num_models = len(models)
fig, axes = plt.subplots(nrows=num_models, ncols=2, figsize=(12, 5 * num_models))

# Ensure axes is 2D even if there's only 1 model
if num_models == 1:
    axes = [axes]

for i, (name, clf) in enumerate(models.items()):
    # 1. Build and fit the pipeline on the full training set
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train_enc)

    # 2. Generate predictions and probabilities on the unseen test set
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # -----------------------------------------------------------
    # LEFT COLUMN: Plot Confusion Matrix
    # -----------------------------------------------------------
    ax_cm = axes[i][0]
    plot_confusion_matrix(
        ax=ax_cm,
        y_true=y_test_enc,
        y_pred=y_pred,
        title=f"{name} - Confusion Matrix",
        y_proba=y_proba,
    )

    # -----------------------------------------------------------
    # RIGHT COLUMN: Plot ROC Curve
    # -----------------------------------------------------------
    ax_roc = axes[i][1]
    plot_roc_curve(ax=ax_roc, y_true=y_test_enc, y_proba=y_proba, model_name=name)

    # Format the ROC plot for readability
    ax_roc.plot(
        [0, 1], [0, 1], "k--", alpha=0.5, label="Random Chance"
    )  # Diagonal line
    ax_roc.set_title(f"{name} - ROC Curve", fontsize=12, fontweight="bold", pad=8)
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=9)
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)", fontsize=9)
    ax_roc.legend(loc="lower right", fontsize=9)

# Adjust layout to prevent overlapping text and display the plots
plt.tight_layout(pad=3.0)
plt.show()

# =============================================================================
# 11. COMBINED ROC CURVE (ALL MODELS ON ONE PLOT)
# =============================================================================
print("\n" + "=" * 50)
print("GENERATING COMBINED ROC PLOT FOR ALL MODELS")
print("=" * 50)

plt.figure(figsize=(10, 8))
ax = plt.gca()

# Generate distinct colours for each model (using a standard matplotlib colormap)
colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

# Loop through all models, fit, predict, and plot on the SAME axes
for (name, clf), color in zip(models.items(), colors):
    # Fit full pipeline to prevent data leakage
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train_enc)

    # Get probabilities for the positive class
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # Calculate False Positive Rate, True Positive Rate, and AUC
    fpr, tpr, _ = roc_curve(y_test_enc, y_proba)
    auc = roc_auc_score(y_test_enc, y_proba)

    # Plot curve (omitting fill_between so the overlapping plot remains clean)
    ax.plot(fpr, tpr, lw=2.5, color=color, label=f"{name} (AUC = {auc:.3f})")

# Plot the 50/50 random chance diagonal line
ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random Chance (AUC = 0.500)")

# Format the plot
ax.set_title(
    "Combined ROC Curve — Model Comparison", fontsize=14, fontweight="bold", pad=15
)
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=10)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, labelpad=10)
ax.set_xlim([-0.01, 1.0])
ax.set_ylim([0.0, 1.01])

# Order the legend by highest AUC (matplotlib puts them in the order they were plotted)
handles, labels = ax.get_legend_handles_labels()
# Sort handles and labels together based on the AUC text extracted from the label
hl = sorted(zip(handles, labels), key=lambda x: x[1].split("=")[-1], reverse=True)
handles2, labels2 = zip(*hl)
ax.legend(handles2, labels2, loc="lower right", fontsize=10, frameon=True, shadow=True)

ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

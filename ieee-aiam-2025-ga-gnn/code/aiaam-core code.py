import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import train_test_split
import xgboost as xgb

# -------------------------
# Paths and I/O
# -------------------------
desktop_path = r"C:\Users\1\Desktop"
data_path = os.path.join(desktop_path, "data.xlsx")

fig_dir = os.path.join(desktop_path, "figures")
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# -------------------------
# Load data
# -------------------------
df = pd.read_excel(data_path)

# All columns except last = features, last column = label
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Train XGBoost model
# -------------------------
model = xgb.XGBClassifier(
    n_estimators=100,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------
# Performance metrics (Fig.2)
# -------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 6))
plt.bar(metrics, values)
plt.title(
    "Model Performance Comparison",
    fontsize=14, fontweight="bold", family="Times New Roman"
)
plt.xlabel("Metrics", fontsize=12, fontweight="bold", family="Times New Roman")
plt.ylabel("Scores", fontsize=12, fontweight="bold", family="Times New Roman")
plt.ylim([0, 1])

for i, v in enumerate(values):
    plt.text(
        i, v + 0.01, str(round(v, 2)),
        ha="center", va="bottom",
        fontstyle="italic", fontweight="bold", fontname="Times New Roman"
    )

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "performance_comparison.png"), dpi=300)
plt.show()

# -------------------------
# SHAP summary plot (Fig.3)
# -------------------------
# Use a subset of training data as background for speed
background = shap.sample(X_train, 200, random_state=42)

explainer = shap.KernelExplainer(model.predict_proba, background)
shap_values = explainer.shap_values(X_test)  # list: [class0, class1]

plt.figure()
shap.summary_plot(shap_values[1], X_test, show=False)
plt.title(
    "SHAP Feature Importance",
    fontsize=14, fontweight="bold", family="Times New Roman"
)
plt.xlabel("SHAP Value", fontsize=12, fontweight="bold", family="Times New Roman")
plt.ylabel("Features", fontsize=12, fontweight="bold", family="Times New Roman")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "shap_summary_plot.png"), dpi=300)
plt.show()

# -------------------------
# Sensitivity analysis (Fig.4)
# -------------------------
perturbation_levels = np.linspace(0, 1, 10)
performance_scores = []

for level in perturbation_levels:
    X_test_perturbed = X_test.copy()
    X_test_perturbed[:, 0] += np.random.normal(
        loc=0, scale=level, size=X_test_perturbed[:, 0].shape
    )
    y_pred_perturbed = model.predict(X_test_perturbed)
    performance_scores.append(accuracy_score(y_test, y_pred_perturbed))

plt.figure(figsize=(8, 6))
plt.plot(
    perturbation_levels, performance_scores,
    marker="o", linestyle="-", linewidth=2
)
plt.title(
    "Sensitivity Analysis: Performance vs Perturbation",
    fontsize=14, fontweight="bold", family="Times New Roman"
)
plt.xlabel(
    "Perturbation Level",
    fontsize=12, fontweight="bold", family="Times New Roman"
)
plt.ylabel(
    "Accuracy",
    fontsize=12, fontweight="bold", family="Times New Roman"
)

for i, txt in enumerate(performance_scores):
    plt.text(
        perturbation_levels[i], performance_scores[i] + 0.01, str(round(txt, 2)),
        ha="center", va="bottom", fontsize=10,
        fontstyle="italic", fontweight="bold", fontname="Times New Roman"
    )

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "sensitivity_analysis.png"), dpi=300)
plt.show()

# -------------------------
# SHAP dependence plot (Fig.5)
# -------------------------
plt.figure()
shap.dependence_plot(0, shap_values[1], X_test, show=False)
plt.title(
    "SHAP Dependence Plot for Feature 0",
    fontsize=14, fontweight="bold", family="Times New Roman"
)
plt.xlabel(
    "Feature 0 Value",
    fontsize=12, fontweight="bold", family="Times New Roman"
)
plt.ylabel(
    "SHAP Value",
    fontsize=12, fontweight="bold", family="Times New Roman"
)
plt.tight_layout()
plt.savefig(
    os.path.join(fig_dir, "shap_dependence_plot_feature_0.png"),
    dpi=300
)
plt.show()

# -------------------------
# Confusion matrix (Fig.6)
# -------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", cbar=False,
    xticklabels=["Class 0", "Class 1"],
    yticklabels=["Class 0", "Class 1"]
)
plt.title(
    "Confusion Matrix",
    fontsize=14, fontweight="bold", family="Times New Roman"
)
plt.xlabel(
    "Predicted",
    fontsize=12, fontweight="bold", family="Times New Roman"
)
plt.ylabel(
    "True",
    fontsize=12, fontweight="bold", family="Times New Roman"
)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "confusion_matrix.png"), dpi=300)
plt.show()

# -------------------------
# ROC curve (Fig.7)
# -------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(
    fpr, tpr, lw=2,
    label="ROC curve (AUC = %0.2f)" % roc_auc
)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.title(
    "Receiver Operating Characteristic (ROC)",
    fontsize=14, fontweight="bold", family="Times New Roman"
)
plt.xlabel(
    "False Positive Rate",
    fontsize=12, fontweight="bold", family="Times New Roman"
)
plt.ylabel(
    "True Positive Rate",
    fontsize=12, fontweight="bold", family="Times New Roman"
)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "roc_curve.png"), dpi=300)
plt.show()

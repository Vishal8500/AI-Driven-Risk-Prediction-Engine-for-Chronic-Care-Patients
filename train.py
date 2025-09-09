# Full pipeline: Train XGBoost (with fallback), calibration, evaluation,
# classification report, SHAP explainability, and save model + plots.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve,
    classification_report
)
from sklearn.impute import SimpleImputer

# ----------------------------
# 1. Load CSV
# ----------------------------
csv_paths = [r"D:\Welldoc\patient_data.csv"]
data = None
for p in csv_paths:
    if os.path.exists(p):
        data = pd.read_csv(p)
        print("Loaded data from:", p)
        break
if data is None:
    raise FileNotFoundError("Could not find patient_data.csv. Put it at D:\\Welldoc or working dir or /mnt/data.")

# Features and target
if "label" not in data.columns:
    raise KeyError("Expected target column named 'label' in CSV.")
X = data.drop("label", axis=1)
y = data["label"].astype(int)

# Impute missing values
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=True, random_state=42, stratify=y
)

# ----------------------------
# 2. Train model (XGBoost with fallback)
# ----------------------------
HAS_XGB = True
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except Exception:
    HAS_XGB = False
    print("xgboost not available - using HistGradientBoosting fallback.")

pos = max(1, int(y_train.sum()))
neg = max(1, int(len(y_train) - pos))
scale_pos_weight = neg / pos
print(f"Train size: {len(y_train)}  Pos: {pos}  Neg: {neg}  scale_pos_weight={scale_pos_weight:.3f}")

xgb_params = {
    "learning_rate": 0.03,
    "n_estimators": 1000,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.5,
    "reg_lambda": 2.0,
    "reg_alpha": 0.1,
    "use_label_encoder": False,
    "random_state": 42,
    "verbosity": 0,
}

early_stopping_rounds = 50
eval_metric_for_es = "auc"
threshold = 0.5
final_model = None

if HAS_XGB:
    try:
        clf = XGBClassifier(**{k: v for k, v in xgb_params.items() if k != "n_estimators"})
        clf.set_params(n_estimators=xgb_params["n_estimators"], scale_pos_weight=scale_pos_weight)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric=eval_metric_for_es,
            early_stopping_rounds=early_stopping_rounds,
            verbose=50
        )
        final_model = clf
    except TypeError:
        print("early_stopping not supported in this version, fitting without it...")
        clf = XGBClassifier(**xgb_params)
        clf.set_params(scale_pos_weight=scale_pos_weight)
        clf.fit(X_train, y_train)
        final_model = clf

if not HAS_XGB or final_model is None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    print("Falling back to HistGradientBoostingClassifier...")
    final_model = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_depth=6,
        l2_regularization=1.0,
        random_state=42
    )
    final_model.fit(X_train, y_train)

# ----------------------------
# 3. Calibration
# ----------------------------
print("\nCalibrating model probabilities...")
try:
    calibrated = CalibratedClassifierCV(final_model, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    probs = calibrated.predict_proba(X_test)[:, 1]
except Exception:
    print("Calibration failed, using raw model outputs.")
    calibrated = final_model
    probs = final_model.predict_proba(X_test)[:, 1]

preds = (probs >= threshold).astype(int)

# ----------------------------
# 4. Evaluation
# ----------------------------
auroc = roc_auc_score(y_test, probs)
auprc = average_precision_score(y_test, probs)
brier = brier_score_loss(y_test, probs)

print("\n=== Evaluation ===")
print("AUROC:", round(auroc, 4))
print("AUPRC:", round(auprc, 4))
print("Brier score:", round(brier, 4))
print("\n--- Classification Report ---")
print(classification_report(y_test, preds, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (threshold={threshold})")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, probs)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Curve")
plt.show()

# Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, probs)
plt.figure(figsize=(6,4))
plt.plot(rec, prec, label=f"AUPRC={auprc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision-Recall Curve")
plt.show()

# ----------------------------
# 5. SHAP explainability
# ----------------------------
try:
    import shap
    explainer = shap.TreeExplainer(final_model)
    shap_vals = explainer(X_test)
    shap.summary_plot(shap_vals, X_test)
    shap.plots.waterfall(shap_vals[0])
except Exception as e:
    print("SHAP skipped:", e)

# ----------------------------
# 6. Save model + plots
# ----------------------------
outdir = r"D:\Welldoc\outputs"
os.makedirs(outdir, exist_ok=True)

# Save models
joblib.dump(final_model, os.path.join(outdir, "final_model.pkl"))
joblib.dump(calibrated, os.path.join(outdir, "calibrated_model.pkl"))

# Save plots
cm_fig = disp.figure_
cm_fig.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Curve")
plt.savefig(os.path.join(outdir, "roc_curve.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6,4))
plt.plot(rec, prec, label=f"AUPRC={auprc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision-Recall Curve")
plt.savefig(os.path.join(outdir, "precision_recall_curve.png"), dpi=300, bbox_inches="tight")
plt.close()

try:
    shap.summary_plot(shap_vals, X_test, show=False)
    plt.savefig(os.path.join(outdir, "shap_summary.png"), dpi=300, bbox_inches="tight")
    plt.close()
except Exception:
    pass

print("\nAll outputs saved in:", outdir)

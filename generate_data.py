# ============================
# Synthetic Dataset + XGBoost + Evaluation + SHAP
# Added features: prev_hosp_365, sleep_hours
# Full copy-paste runnable notebook code
# ============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    ConfusionMatrixDisplay, brier_score_loss, roc_curve, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb
import shap

# ----------------------------
# 1. Generate synthetic dataset
# ----------------------------
np.random.seed(42)
N = 2000

data = pd.DataFrame({
    "age": np.random.randint(30, 90, N),
    "sex": np.random.choice([0, 1], N),  # 0=female, 1=male
    "bmi": np.random.normal(28, 5, N),
    "systolic_bp": np.random.normal(130, 15, N),
    "diastolic_bp": np.random.normal(80, 10, N),
    "heart_rate": np.random.normal(75, 10, N),
    "glucose": np.random.normal(110, 25, N),
    "creatinine": np.random.normal(1.0, 0.3, N),
    "adherence": np.random.uniform(0.4, 1.0, N),  # med adherence %
    "steps": np.random.normal(6000, 2000, N),
    "prev_hosp_365": np.random.binomial(1, 0.12, N),  # new feature
    "sleep_hours": np.clip(np.random.normal(6.8, 1.2, N), 2, 12),  # new feature
})

# Synthetic target with some nonlinear relationships
logit = (
    0.03*(data["age"] - 60) +
    0.05*(data["bmi"] - 28) +
    0.04*(data["systolic_bp"] - 130) +
    0.05*(data["glucose"] - 110) -
    2*(data["adherence"] - 0.7) +
    0.6*data["prev_hosp_365"] -
    0.2*(data["sleep_hours"] - 7)
)
prob = 1 / (1 + np.exp(-logit))
data["label"] = np.random.binomial(1, prob)

output_filename = 'patient_data.csv'
print(f"7. Saving generated dataset to '{output_filename}'...")
data.to_csv(output_filename, index=False)
print("Script finished successfully.")
# app.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score
)
import xgboost as xgb

# ----------------------------
# Setup
# ----------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title="Patient Risk Dashboard", layout="wide", page_icon="🩺")

# ----------------------------
# Load Data & Model
# ----------------------------
data_path = r"D:\Welldoc\patient_data.csv"
shap_img_path = r"D:\Welldoc\outputs\shap_summary.png"
model_path = os.path.join(OUTPUT_DIR, "calibrated_model.pkl")
test_data_path = os.path.join(OUTPUT_DIR, "test_data.pkl")

data = pd.read_csv(data_path)
X = data.drop("label", axis=1)
y = data["label"]
feature_names = list(X.columns)

if os.path.exists(model_path) and os.path.exists(test_data_path):
    model = joblib.load(model_path)
    X_test, y_test = joblib.load(test_data_path)
else:
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=42
    )

    # Train model
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )
    clf.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(clf, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)

    joblib.dump(calibrated, model_path)
    joblib.dump((X_test, y_test), test_data_path)
    model = calibrated

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
                        ["Prediction", "Evaluation Dashboard", "Cohort View", "Patient Detail View"], 
                        key="page_nav")

# ----------------------------
# Prediction Page
# ----------------------------
if page == "Prediction":
    st.markdown("<h1 style='text-align:center;color:#4B0082;'>🩺 Patient Risk Prediction</h1>", unsafe_allow_html=True)
    st.markdown("### Enter patient parameters:")

    # Two-column layout for inputs
    cols = st.columns(2)
    user_input = {}
    for i, feature in enumerate(feature_names):
        col = cols[i % 2]
        val = col.number_input(f"{feature}", value=float(X[feature].mean()), format="%.3f", key=f"input_{feature}")
        user_input[feature] = val

    input_df = pd.DataFrame([user_input])
    pred_prob = model.predict_proba(input_df)[:, 1][0]
    pred_class = int(pred_prob >= 0.5)

    # Display in stylish cards
    col1, col2 = st.columns(2)
    col1.metric("Predicted Probability", f"{pred_prob:.3f}")
    col2.metric("Predicted Class", f"{pred_class}", delta_color="inverse")

    st.success("✅ Prediction Completed Successfully!")

# ----------------------------
# Evaluation Dashboard
# ----------------------------
elif page == "Evaluation Dashboard":
    st.markdown("<h1 style='text-align:center;color:#FF4500;'>📊 Model Evaluation Dashboard</h1>", unsafe_allow_html=True)

    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= 0.5).astype(int)

    # Metrics
    auroc = roc_auc_score(y_test, probs)
    auprc = average_precision_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.markdown("### 📈 Performance Metrics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("Precision", f"{precision:.3f}")
    col3.metric("Recall", f"{recall:.3f}")
    col4.metric("AUROC", f"{auroc:.3f}")
    col5.metric("AUPRC", f"{auprc:.3f}")
    col6.metric("Brier Score", f"{brier:.3f}")

    st.markdown("---")
    st.markdown("### 📊 Visualization Dashboard")

    # 2x2 grid
    col1, col2 = st.columns(2)

    # Confusion Matrix
    with col1:
        st.subheader("Confusion Matrix")
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
        disp.plot(cmap=plt.cm.Blues, ax=ax1)
        ax1.set_title("Confusion Matrix")
        st.pyplot(fig1)
        plt.close(fig1)

    # ROC Curve
    with col2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, probs)
        fig2, ax2 = plt.subplots(figsize=(6,5))
        ax2.plot(fpr, tpr, label=f"AUROC={auroc:.3f}", color="#FF6347", linewidth=2)
        ax2.plot([0,1],[0,1], linestyle="--", color="gray", alpha=0.7)
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)

    # Second row
    col3, col4 = st.columns(2)

    # Precision-Recall Curve
    with col3:
        st.subheader("Precision-Recall Curve")
        prec, rec, _ = precision_recall_curve(y_test, probs)
        fig3, ax3 = plt.subplots(figsize=(6,5))
        ax3.plot(rec, prec, label=f"AUPRC={auprc:.3f}", color="#4B0082", linewidth=2)
        ax3.set_xlabel("Recall")
        ax3.set_ylabel("Precision")
        ax3.set_title("Precision-Recall Curve")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        plt.close(fig3)

    # SHAP Summary (Static)
    with col4:
        st.subheader("SHAP Feature Importance")
        if os.path.exists(shap_img_path):
            st.image(shap_img_path, caption="SHAP Summary", use_container_width=True)
        else:
            st.error("SHAP summary image not found.")

# ----------------------------
# Cohort View
# ----------------------------
elif page == "Cohort View":
    st.markdown("<h1 style='text-align:center;color:#4B0082;'>🧑‍🤝‍🧑 Cohort Risk Overview</h1>", unsafe_allow_html=True)
    data['risk_score'] = model.predict_proba(X)[:,1]

    st.metric("Average Risk Score", f"{data['risk_score'].mean():.3f}")
    st.dataframe(data[['risk_score'] + feature_names])

    top10 = data.nlargest(10, 'risk_score')
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(top10.index.astype(str), top10['risk_score'], color="#FF6347")
    ax.set_xlabel("Patient Index")
    ax.set_ylabel("Risk Score")
    ax.set_title("Top 10 High-Risk Patients")
    st.pyplot(fig)
    plt.close(fig)

# ----------------------------
# Patient Detail View
# ----------------------------
elif page == "Patient Detail View":
    st.markdown("<h1 style='text-align:center;color:#FF4500;'>🩺 Patient Detail Dashboard</h1>", unsafe_allow_html=True)

    patient_index = st.selectbox("Select Patient Index", options=data.index, key="patient_select")
    patient_data = X.loc[patient_index:patient_index]
    patient_risk = model.predict_proba(patient_data)[:,1][0]
    st.metric("Patient Risk Score", f"{patient_risk:.3f}")

    st.markdown("### Key Patient Parameters")
    st.table(patient_data.T.rename(columns={patient_index:"Value"}))

    st.markdown("### Key Drivers of Risk")
    if os.path.exists(shap_img_path):
        st.image(shap_img_path, caption="SHAP Summary", use_container_width=True)
    else:
        st.warning("SHAP image not found.")

    st.markdown("### Recommended Next Actions")
    if patient_risk > 0.7:
        st.success("⚠️ High risk! Urgent intervention required.")
    elif patient_risk > 0.4:
        st.info("🔹 Moderate risk. Schedule follow-up.")
    else:
        st.success("✅ Low risk. Continue standard care.")

    st.markdown("### Risk Trend (Historical)")
    trend = [0.1, 0.3, 0.5, patient_risk]  # placeholder trend
    fig2, ax2 = plt.subplots(figsize=(8,3))
    ax2.plot(range(len(trend)), trend, marker='o', color="#4B0082")
    ax2.set_xlabel("Time Point")
    ax2.set_ylabel("Risk Score")
    ax2.set_title("Patient Risk Trend")
    st.pyplot(fig2)
    plt.close(fig2)

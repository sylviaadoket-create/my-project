import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import (RocCurveDisplay, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)

warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD ARTIFACTS
# ==========================================
@st.cache_resource
def load_artifacts():
    metrics = joblib.load('saved_models/metrics.joblib')
    X_test = joblib.load('saved_models/X_test.joblib')
    y_test = joblib.load('saved_models/y_test.joblib')
    features = joblib.load('saved_models/features.joblib')
    X_train_transformed = joblib.load('saved_models/X_train_transformed.joblib')

    model_names = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
    pipelines = {}
    for name in model_names:
        safe_name = name.replace(' ', '_')
        pipelines[name] = joblib.load(f'saved_models/{safe_name}_pipeline.joblib')

    return metrics, X_test, y_test, features, pipelines, X_train_transformed

metrics, X_test, y_test, features, pipelines, X_train_transformed = load_artifacts()

@st.cache_data
def load_raw_data():
    return pd.read_csv('htn_dat.csv')

df_raw = load_raw_data()

# ==========================================
# 2. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Clinical Event Prediction Dashboard", layout="wide", page_icon="🩺")

st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home / Overview",
    "📊 Exploratory Data Analysis",
    "📈 Model Performance & Metrics",
    "🔍 Feature Importance & XAI",
    "🩺 Patient Prediction"
])

st.sidebar.markdown("---")
selected_model_name = st.sidebar.selectbox("⚙️ Select Model", list(pipelines.keys()))
pipeline = pipelines[selected_model_name]
model = pipeline.named_steps['classifier']
preprocessor = pipeline.named_steps['preprocessor']

# ==========================================
# 3. PAGES
# ==========================================

# ---------- HOME ----------
if page == "🏠 Home / Overview":
    st.title("🏥 Clinical Event Prediction Dashboard")
    st.markdown("""
    Welcome to the **Hypertension & Clinical Event Prediction Dashboard**.

    This application uses machine learning to predict the probability of a clinical
    event based on patient demographics, vitals, and medical history.

    **Features used:**
    `DBP`, `SBP`, `BMI`, `age`, `married`, `male.gender`, `hgb_centered`,
    `adv_HIV`, `arv_naive`, `urban.clinic`, `log_creat_centered`, `SBP_ge120`

    ### How to use:
    1. Select a model from the sidebar.
    2. Explore the data in **EDA**.
    3. Review model metrics and ROC curves in **Model Performance**.
    4. Understand feature contributions in **Feature Importance & XAI**.
    5. Predict risk for a new patient in **Patient Prediction**.
    """)

    st.metric("Total Patients", len(df_raw))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Events (1)", int(df_raw['event'].sum()))
    with col2:
        st.metric("Non-Events (0)", int((df_raw['event'] == 0).sum()))
    with col3:
        st.metric("Event Rate", f"{df_raw['event'].mean()*100:.1f}%")

# ---------- EDA ----------
elif page == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df_raw.head(10))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='event', data=df_raw, ax=ax, palette='Set2')
        ax.set_xlabel("Event (0 = No, 1 = Yes)")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Age Distribution by Event")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df_raw, x='age', hue='event', kde=True, ax=ax, palette='Set2')
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Correlation Heatmap (Numeric Features)")
    num_cols = ['DBP', 'SBP', 'BMI', 'age', 'hgb_centered', 'log_creat_centered', 'event']
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_raw[num_cols].corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# ---------- MODEL PERFORMANCE ----------
elif page == "📈 Model Performance & Metrics":
    st.title(f"📈 Model Performance: {selected_model_name}")

    # Compute predictions on the fly from the loaded pipeline
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    current_metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }

    st.subheader("Evaluation Metrics")
    metrics_df = pd.DataFrame(list(current_metrics.items()), columns=['Metric', 'Score'])
    st.dataframe(metrics_df.style.format({'Score': '{:.4f}'}), use_container_width=True)

    st.subheader("ROC Curve")
    fig, ax = plt.subplots(figsize=(8, 6))
    # FIX: Do NOT pass color here. Set it after creation.
    roc_display = RocCurveDisplay.from_predictions(
        y_test, y_prob, ax=ax, name=selected_model_name
    )
    roc_display.line_.set_color("darkorange")
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_title(f"ROC Curve - {selected_model_name}")
    st.pyplot(fig)
    plt.close(fig)

    # Compare all models
    st.subheader("All Models Comparison")
    all_metrics = pd.DataFrame(metrics).T
    st.dataframe(all_metrics.style.format("{:.4f}"), use_container_width=True)

# ---------- FEATURE IMPORTANCE & XAI ----------
elif page == "🔍 Feature Importance & XAI":
    st.title(f"🔍 Feature Importance & XAI: {selected_model_name}")

    # Transform test data using the fitted preprocessor from the pipeline
    X_test_transformed = preprocessor.transform(X_test)

    st.subheader("Global Feature Importance")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])

    imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    imp_df = imp_df.sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=imp_df, ax=ax, palette='viridis')
    ax.set_title(f"Feature Importance - {selected_model_name}")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("SHAP Summary Plot (Global Explainability)")
    try:
        if hasattr(model, 'tree_'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_train_transformed)

        shap_values = explainer.shap_values(X_test_transformed)

        fig, ax = plt.subplots(figsize=(10, 6))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_test_transformed,
                              feature_names=features, show=False)
        else:
            shap.summary_plot(shap_values, X_test_transformed,
                              feature_names=features, show=False)
        st.pyplot(bbox_inches='tight')
        plt.close('all')
    except Exception as e:
        st.warning(f"SHAP summary plot could not be generated: {e}")

# ---------- PATIENT PREDICTION ----------
elif page == "🩺 Patient Prediction":
    st.title("🩺 Patient Prediction & Recommendations")
    st.markdown("Enter the patient's clinical parameters to predict event risk.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Vitals & Demographics")
        age = st.number_input("Age", min_value=10.0, max_value=100.0, value=35.0, step=1.0)
        SBP = st.number_input("Systolic BP (SBP)", min_value=50.0, max_value=250.0, value=120.0, step=1.0)
        DBP = st.number_input("Diastolic BP (DBP)", min_value=30.0, max_value=150.0, value=70.0, step=1.0)
        BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0, step=0.1)
        hgb_centered = st.number_input("Hemoglobin (Centered)", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
        log_creat_centered = st.number_input("Log Creatinine (Centered)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)

    with col2:
        st.markdown("#### Categorical / Binary Features")
        married = st.selectbox("Married", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        male_gender = st.selectbox("Male Gender", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        adv_HIV = st.selectbox("Advanced HIV", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        arv_naive = st.selectbox("ARV Naive", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        urban_clinic = st.selectbox("Urban Clinic", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        SBP_ge120 = st.selectbox("SBP >= 120", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    if st.button("🚀 Predict Risk", type="primary"):
        input_data = pd.DataFrame([{
            'DBP': float(DBP),
            'SBP': float(SBP),
            'BMI': float(BMI),
            'age': float(age),
            'married': int(married),
            'male.gender': int(male_gender),
            'hgb_centered': float(hgb_centered),
            'adv_HIV': int(adv_HIV),
            'arv_naive': int(arv_naive),
            'urban.clinic': int(urban_clinic),
            'log_creat_centered': float(log_creat_centered),
            'SBP_ge120': int(SBP_ge120)
        }])

        # Pipeline handles preprocessing automatically
        prob = pipeline.predict_proba(input_data)[0, 1]
        prob_pct = prob * 100

        st.markdown("---")
        st.subheader("Prediction Result")

        # Gauge-style display
        col_a, col_b, col_c = st.columns(3)
        with col_b:
            st.metric("Predicted Event Probability", f"{prob_pct:.2f}%")

        st.subheader("📋 Clinical Recommendation")
        if prob <= 0.30:
            st.success(f"🟢 **Low Risk ({prob_pct:.2f}%)** — Routine monitoring and standard care.")
        elif prob <= 0.50:
            st.warning(f"🟡 **Moderate Risk ({prob_pct:.2f}%)** — Increased surveillance, lifestyle interventions.")
        elif prob <= 0.70:
            st.error(f"🟠 **High Risk ({prob_pct:.2f}%)** — Medical review, potential pharmacological intervention.")
        else:
            st.error(f"🔴 **Critical Risk ({prob_pct:.2f}%)** — Immediate clinical intervention required.")

        st.markdown("---")
        st.subheader("🧠 Explainable AI — Local SHAP Waterfall")
        st.markdown("How each feature pushed the prediction for **this specific patient**:")

        try:
            input_transformed = preprocessor.transform(input_data)

            if hasattr(model, 'tree_'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.LinearExplainer(model, X_train_transformed)

            shap_values = explainer.shap_values(input_transformed)

            if isinstance(shap_values, list):
                vals = shap_values[1][0]
                base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            else:
                vals = shap_values[0]
                base_val = explainer.expected_value

            explanation = shap.Explanation(
                values=vals,
                base_value=base_val,
                feature_names=features,
                data=input_transformed[0]
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(explanation, show=False, max_display=12)
            st.pyplot(fig, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            st.warning(f"SHAP waterfall plot could not be generated: {e}")
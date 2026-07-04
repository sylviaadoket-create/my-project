import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import RocCurveDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD ARTIFACTS (from root directory)
# ==========================================
@st.cache_resource
def load_artifacts():
    """Load all pre-trained models and data artifacts from root directory."""
    # Load metrics and test data
    metrics = joblib.load('metrics.joblib')
    X_test = joblib.load('X_test.joblib')
    y_test = joblib.load('y_test.joblib')
    features = joblib.load('features.joblib')
    X_train_transformed = joblib.load('X_train_transformed.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    
    # Load all three model pipelines
    pipelines = {
        'Logistic Regression': joblib.load('Logistic_Regression_pipeline.joblib'),
        'Random Forest': joblib.load('Random_Forest_pipeline.joblib'),
        'Gradient Boosting': joblib.load('Gradient_Boosting_pipeline.joblib')
    }
    
    return metrics, X_test, y_test, features, pipelines, X_train_transformed, preprocessor

# Load everything once and cache it
metrics, X_test, y_test, features, pipelines, X_train_transformed, preprocessor = load_artifacts()

# Load raw dataset for EDA
@st.cache_data
def load_raw_data():
    return pd.read_csv('htn_dat.csv')

df_raw = load_raw_data()

# ==========================================
# 2. STREAMLIT UI LAYOUT
# ==========================================
st.set_page_config(
    page_title="HTN Clinical Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🩺 HTN Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to",
    [
        " Home / Overview",
        "📊 Exploratory Data Analysis",
        "📈 Model Performance & Metrics",
        " Feature Importance & XAI",
        " Patient Prediction"
    ]
)

st.sidebar.markdown("---")
selected_model_name = st.sidebar.selectbox(
    "⚙️ Select Model",
    list(pipelines.keys())
)

pipeline = pipelines[selected_model_name]
model = pipeline.named_steps['classifier']

# ==========================================
# 3. PAGES
# ==========================================

# ---------- HOME ----------
if page == "🏠 Home / Overview":
    st.markdown('<p class="main-header">🏥 HTN Clinical Event Prediction Dashboard</p>', unsafe_allow_html=True)
    st.markdown("""
    Welcome to the **Hypertension & Clinical Event Prediction Dashboard**.
    
    This application leverages machine learning to predict the probability of a clinical event
    based on patient demographics, vitals, and medical history from the HTN dataset.
    
    ### 📋 Features Used:
    `DBP`, `SBP`, `BMI`, `age`, `married`, `male.gender`, `hgb_centered`,
    `adv_HIV`, `arv_naive`, `urban.clinic`, `log_creat_centered`, `SBP_ge120`
    
    ###  How to Use:
    1. **Select a model** from the sidebar (Logistic Regression, Random Forest, or Gradient Boosting)
    2. **Explore the data** in the Exploratory Data Analysis tab
    3. **Review model metrics** and ROC curves in Model Performance
    4. **Understand feature contributions** in Feature Importance & XAI
    5. **Predict risk** for a new patient in the Patient Prediction tab
    """)
    
    # Dataset stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", len(df_raw))
    with col2:
        st.metric("Events (1)", int(df_raw['event'].sum()))
    with col3:
        st.metric("Non-Events (0)", int((df_raw['event'] == 0).sum()))
    with col4:
        event_rate = df_raw['event'].mean() * 100
        st.metric("Event Rate", f"{event_rate:.1f}%")
    
    st.markdown("---")
    st.subheader("📊 Dataset Preview")
    st.dataframe(df_raw.head(10), use_container_width=True)

# ---------- EDA ----------
elif page == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Shape:**", df_raw.shape)
        st.write("**Columns:**", list(df_raw.columns))
    with col2:
        st.write("**Missing Values:**")
        missing = df_raw.isnull().sum()
        missing_df = pd.DataFrame({'Column': missing.index, 'Missing': missing.values})
        st.dataframe(missing_df[missing_df['Missing'] > 0], use_container_width=True)
    
    st.markdown("---")
    st.subheader("Target Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='event', data=df_raw, ax=ax, palette='Set2')
        ax.set_title("Event Distribution")
        ax.set_xlabel("Event (0 = No, 1 = Yes)")
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        event_counts = df_raw['event'].value_counts()
        ax.pie(event_counts, labels=['No Event', 'Event'], autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
        ax.set_title("Event Percentage")
        st.pyplot(fig)
        plt.close(fig)
    
    st.markdown("---")
    st.subheader("Numeric Feature Distributions")
    numeric_cols = ['DBP', 'SBP', 'BMI', 'age', 'hgb_centered', 'log_creat_centered']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.histplot(data=df_raw, x=col, hue='event', kde=True, ax=axes[i], palette='Set2')
        axes[i].set_title(f"{col} by Event")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("---")
    st.subheader("Correlation Heatmap")
    corr_cols = ['DBP', 'SBP', 'BMI', 'age', 'hgb_centered', 'log_creat_centered', 'event']
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_raw[corr_cols].corr(), annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
    ax.set_title("Feature Correlation Matrix")
    st.pyplot(fig)
    plt.close(fig)

# ---------- MODEL PERFORMANCE ----------
elif page == "📈 Model Performance & Metrics":
    st.title(f"📈 Model Performance: {selected_model_name}")
    
    # Compute predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)
    
    # Metrics display
    st.subheader("Evaluation Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accuracy", f"{acc:.4f}")
    with col2:
        st.metric("Precision", f"{prec:.4f}")
    with col3:
        st.metric("Recall", f"{rec:.4f}")
    with col4:
        st.metric("F1-Score", f"{f1:.4f}")
    with col5:
        st.metric("ROC-AUC", f"{roc:.4f}")
    
    st.markdown("---")
    
    # ROC Curve
    st.subheader("ROC Curve")
    fig, ax = plt.subplots(figsize=(8, 6))
    roc_display = RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name=selected_model_name)
    # Fix: Set color after creation instead of using color parameter
    roc_display.line_.set_color("darkorange")
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_title(f"ROC Curve - {selected_model_name}")
    ax.legend(loc='lower right')
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("---")
    
    # Compare all models
    st.subheader("All Models Comparison")
    all_metrics_data = []
    for model_name, model_pipeline in pipelines.items():
        y_p = model_pipeline.predict(X_test)
        y_pr = model_pipeline.predict_proba(X_test)[:, 1]
        all_metrics_data.append({
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_p),
            'Precision': precision_score(y_test, y_p, zero_division=0),
            'Recall': recall_score(y_test, y_p, zero_division=0),
            'F1-Score': f1_score(y_test, y_p, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_pr)
        })
    
    comparison_df = pd.DataFrame(all_metrics_data)
    st.dataframe(comparison_df.style.format({
        'Accuracy': '{:.4f}',
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'F1-Score': '{:.4f}',
        'ROC-AUC': '{:.4f}'
    }), use_container_width=True)

# ---------- FEATURE IMPORTANCE & XAI ----------
elif page == "🔍 Feature Importance & XAI":
    st.title(f"🔍 Feature Importance & XAI: {selected_model_name}")
    
    # Global Feature Importance
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
    
    st.markdown("---")
    
    # SHAP Summary Plot
    st.subheader("SHAP Summary Plot (Global Explainability)")
    st.write("This plot shows how each feature impacts the model's predictions across all samples.")
    
    try:
        if hasattr(model, 'tree_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:
            explainer = shap.LinearExplainer(model, X_train_transformed)
            shap_values = explainer.shap_values(X_test)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_test, feature_names=features, show=False)
        else:
            shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"SHAP summary plot could not be generated: {e}")
    
    st.markdown("---")
    
    # SHAP Beeswarm Plot
    st.subheader("SHAP Beeswarm Plot")
    st.write("Shows the distribution of SHAP values for each feature.")
    
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        if isinstance(shap_values, list):
            shap.plots.beeswarm(shap.Explanation(values=shap_values[1], data=X_test, feature_names=features), show=False)
        else:
            shap.plots.beeswarm(shap.Explanation(values=shap_values, data=X_test, feature_names=features), show=False)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"SHAP beeswarm plot could not be generated: {e}")

# ---------- PATIENT PREDICTION ----------
elif page == "🩺 Patient Prediction":
    st.title("🩺 Patient Prediction & Recommendations")
    st.markdown("Enter the patient's clinical parameters to predict the risk of a clinical event.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Vitals & Demographics")
        age = st.number_input("Age", min_value=10.0, max_value=100.0, value=35.0, step=1.0)
        SBP = st.number_input("Systolic BP (SBP)", min_value=50.0, max_value=250.0, value=120.0, step=1.0)
        DBP = st.number_input("Diastolic BP (DBP)", min_value=30.0, max_value=150.0, value=70.0, step=1.0)
        BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0, step=0.1)
        hgb_centered = st.number_input("Hemoglobin (Centered)", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
        log_creat_centered = st.number_input("Log Creatinine (Centered)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    
    with col2:
        st.markdown("###  Categorical / Binary Features")
        married = st.selectbox("Married", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        male_gender = st.selectbox("Male Gender", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        adv_HIV = st.selectbox("Advanced HIV", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        arv_naive = st.selectbox("ARV Naive", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        urban_clinic = st.selectbox("Urban Clinic", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        SBP_ge120 = st.selectbox("SBP >= 120", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    st.markdown("---")
    
    if st.button("🚀 Predict Risk", type="primary", use_container_width=True):
        # Create input DataFrame
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
        
        # Predict
        prob = pipeline.predict_proba(input_data)[0, 1]
        prob_pct = prob * 100
        prediction = int(prob >= 0.5)
        
        # Display prediction
        st.subheader(" Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Probability", f"{prob_pct:.2f}%")
        with col2:
            st.metric("Predicted Class", "Event (1)" if prediction == 1 else "No Event (0)")
        
        st.markdown("---")
        
        # Risk-based recommendation
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
        
        # SHAP Waterfall Plot for this prediction
        st.subheader("🧠 Explainable AI — Local SHAP Waterfall")
        st.write("This plot shows how each feature contributed to this specific prediction.")
        
        try:
            input_transformed = preprocessor.transform(input_data)
            
            if hasattr(model, 'tree_'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_transformed)
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
            
            fig, ax = plt.subplots(figsize=(12, 6))
            shap.plots.waterfall(explanation, show=False, max_display=12)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"SHAP waterfall plot could not be generated: {e}")
            st.write("Error details:", str(e))

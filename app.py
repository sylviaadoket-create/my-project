import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, RocCurveDisplay)

warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD DATA & TRAIN MODELS
# ==========================================
@st.cache_data
def load_data():
    return pd.read_csv('htn_dat.csv')

@st.cache_resource
def get_models_and_data():
    """Train models on first load, cache afterwards."""
    df = load_data()
    
    features = ['DBP', 'SBP', 'BMI', 'age', 'married', 'male.gender', 'hgb_centered', 
                'adv_HIV', 'arv_naive', 'urban.clinic', 'log_creat_centered', 'SBP_ge120']
    target = 'event'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    preprocessor = SimpleImputer(strategy='median')
    
    models_dict = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    pipelines = {}
    metrics = {}
    
    for name, model in models_dict.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        pipelines[name] = pipeline
        
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        metrics[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
    
    return pipelines, metrics, X_test, y_test, features

pipelines, metrics, X_test, y_test, features = get_models_and_data()

# ==========================================
# 2. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Clinical Prediction Dashboard", layout="wide", page_icon="🩺")

st.sidebar.title(" Navigation")
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

# ==========================================
# 3. PAGES
# ==========================================

if page == " Home / Overview":
    st.title("🏥 Clinical Event Prediction Dashboard")
    st.markdown("""
    Welcome to the **Hypertension & Clinical Event Prediction Dashboard**.
    
    This app predicts the probability of a clinical event based on patient data.
    """)
    df = load_data()
    st.metric("Total Patients", len(df))

elif page == "📊 Exploratory Data Analysis":
    st.title(" Exploratory Data Analysis")
    df = load_data()
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.countplot(x='event', data=df, ax=ax, palette='Set2')
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.histplot(data=df, x='age', hue='event', kde=True, ax=ax, palette='Set2')
        st.pyplot(fig)

elif page == "📈 Model Performance & Metrics":
    st.title(f"📈 Model Performance: {selected_model_name}")
    
    st.subheader("Evaluation Metrics")
    model_metrics = metrics[selected_model_name]
    st.dataframe(pd.DataFrame([model_metrics]).T.rename(columns={0: 'Score'}))
    
    st.subheader("ROC Curve")
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    fig, ax = plt.subplots()
    roc_display = RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name=selected_model_name)
    roc_display.line_.set_color("darkorange")
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    st.pyplot(fig)

elif page == "🔍 Feature Importance & XAI":
    st.title(f"🔍 Feature Importance & XAI: {selected_model_name}")
    
    model = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    X_test_transformed = preprocessor.transform(X_test)
    
    st.subheader("Global Feature Importance")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])
    
    imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    imp_df = imp_df.sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=imp_df, ax=ax, palette='viridis')
    st.pyplot(fig)

elif page == " Patient Prediction":
    st.title("🩺 Patient Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=30)
        SBP = st.number_input("SBP", min_value=50, max_value=250, value=120)
        DBP = st.number_input("DBP", min_value=30, max_value=150, value=70)
        BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)
        hgb_centered = st.number_input("HGB Centered", value=0.0)
        log_creat_centered = st.number_input("Log Creatinine Centered", value=0.0)
    with col2:
        married = st.selectbox("Married", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        male_gender = st.selectbox("Male Gender", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        adv_HIV = st.selectbox("Advanced HIV", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        arv_naive = st.selectbox("ARV Naive", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        urban_clinic = st.selectbox("Urban Clinic", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        SBP_ge120 = st.selectbox("SBP >= 120", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    if st.button("🚀 Predict Risk"):
        input_data = pd.DataFrame([{
            'DBP': DBP, 'SBP': SBP, 'BMI': BMI, 'age': age,
            'married': married, 'male.gender': male_gender,
            'hgb_centered': hgb_centered, 'adv_HIV': adv_HIV,
            'arv_naive': arv_naive, 'urban.clinic': urban_clinic,
            'log_creat_centered': log_creat_centered, 'SBP_ge120': SBP_ge120
        }])
        
        prob = pipeline.predict_proba(input_data)[0, 1]
        prob_pct = prob * 100
        
        st.markdown(f"### Predicted Probability: **{prob_pct:.2f}%**")
        
        if prob <= 0.30:
            st.success(f"**Low Risk ({prob_pct:.2f}%)**: Routine monitoring.")
        elif prob <= 0.50:
            st.warning(f"**Moderate Risk ({prob_pct:.2f}%)**: Increased surveillance.")
        elif prob <= 0.70:
            st.error(f"**High Risk ({prob_pct:.2f}%)**: Medical review needed.")
        else:
            st.error(f"**Critical Risk ({prob_pct:.2f}%)**: Immediate intervention.")

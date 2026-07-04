import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay

warnings.filterwarnings('ignore')

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
@st.cache_data
def load_data():
    # Ensure your CSV is named 'htn_dat.csv' and has headers
    df = pd.read_csv('htn_dat.csv')
    return df

df = load_data()

# Define features and target based on your dataset structure
target = 'event'
features = ['DBP', 'SBP', 'BMI', 'age', 'married', 'male.gender', 'hgb_centered', 
            'adv_HIV', 'arv_naive', 'urban.clinic', 'log_creat_centered', 'SBP_ge120']

X = df[features]
y = df[target]

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# ==========================================
# 2. MODEL TRAINING (Cached for Performance)
# ==========================================
@st.cache_resource
def train_models():
    """Trains models on first load and caches them."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('classifier', model)
        ])
        # Fit on already imputed data for simplicity in this flow
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

models = train_models()

# ==========================================
# 3. STREAMLIT UI LAYOUT
# ==========================================
st.set_page_config(page_title="HTN Clinical Dashboard", layout="wide", page_icon="🩺")

st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio("Go to", [
    "Home / Overview",
    "Exploratory Data Analysis",
    "Model Performance & Metrics",
    "Feature Importance & XAI",
    "Patient Prediction"
])

st.sidebar.markdown("---")
selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
model = models[selected_model_name]

# ==========================================
# 4. PAGES
# ==========================================

# --- HOME ---
if page == "Home / Overview":
    st.title("🏥 HTN Clinical Event Prediction Dashboard")
    st.markdown("""
    Welcome to the **Hypertension & Clinical Event Prediction Dashboard**.
    This application uses machine learning to predict the probability of a clinical event (`event`)
    based on patient demographics, vitals, and medical history.
    
    **Features:** `DBP`, `SBP`, `BMI`, `age`, `married`, `male.gender`, `hgb_centered`, `adv_HIV`, `arv_naive`, `urban.clinic`, `log_creat_centered`, `SBP_ge120`.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Events", int(y.sum()))
    with col3:
        st.metric("Event Rate", f"{y.mean()*100:.1f}%")

# --- EDA ---
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='event', data=df, ax=ax, palette='Set2')
        st.pyplot(fig)
    with col2:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data=df, x='age', hue='event', kde=True, ax=ax, palette='Set2')
        st.pyplot(fig)

# --- PERFORMANCE ---
elif page == "Model Performance & Metrics":
    st.title(f"📈 Model Performance: {selected_model_name}")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    
    st.subheader("Evaluation Metrics")
    st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=['Score']))
    
    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name=selected_model_name)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    st.pyplot(fig)

# --- XAI ---
elif page == "Feature Importance & XAI":
    st.title(f"🔍 Feature Importance & XAI: {selected_model_name}")
    
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])
        
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances, y=features, ax=ax, palette='viridis')
    st.pyplot(fig)
    
    st.subheader("SHAP Summary Plot")
    
    # FIX: Extract the actual classifier from the model if it's wrapped
    # Check if model is the actual classifier or needs extraction
    model_to_explain = model
    
    try:
        # Create explainer based on model type
        if hasattr(model_to_explain, 'tree_'):
            explainer = shap.TreeExplainer(model_to_explain)
        else:
            explainer = shap.LinearExplainer(model_to_explain, X_train, feature_perturbation="interventional")
        
        shap_values = explainer.shap_values(X_test)
        
        fig, ax = plt.subplots()
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_test, show=False)
        else:
            shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error generating SHAP summary plot: {str(e)}")
        st.info("SHAP summary plot could not be generated for this model.")

# --- PREDICTION ---
elif page == "Patient Prediction":
    st.title("👤 Patient Prediction")
    st.markdown("Enter patient details to predict risk.")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=30)
        SBP = st.number_input("SBP", min_value=50, max_value=250, value=120)
        DBP = st.number_input("DBP", min_value=30, max_value=150, value=70)
        BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)
    with col2:
        married = st.selectbox("Married", [0, 1])
        male_gender = st.selectbox("Male Gender", [0, 1])
        adv_HIV = st.selectbox("Advanced HIV", [0, 1])
        arv_naive = st.selectbox("ARV Naive", [0, 1])
        urban_clinic = st.selectbox("Urban Clinic", [0, 1])
        SBP_ge120 = st.selectbox("SBP >= 120", [0, 1])
        hgb_centered = st.number_input("HGB Centered", value=0.0)
        log_creat_centered = st.number_input("Log Creatinine Centered", value=0.0)

    if st.button("Predict Risk"):
        input_data = pd.DataFrame([{
            'DBP': DBP, 'SBP': SBP, 'BMI': BMI, 'age': age,
            'married': married, 'male.gender': male_gender,
            'hgb_centered': hgb_centered, 'adv_HIV': adv_HIV,
            'arv_naive': arv_naive, 'urban.clinic': urban_clinic,
            'log_creat_centered': log_creat_centered, 'SBP_ge120': SBP_ge120
        }])
        
        prob = model.predict_proba(input_data)[0, 1]
        st.metric("Predicted Probability", f"{prob*100:.2f}%")
        
        # SHAP Waterfall for prediction
        st.subheader("SHAP Explanation")
        
        try:
            # Create explainer
            model_to_explain = model
            
            if hasattr(model_to_explain, 'tree_'):
                explainer = shap.TreeExplainer(model_to_explain)
            else:
                explainer = shap.LinearExplainer(model_to_explain, X_train, feature_perturbation="interventional")
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(input_data)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # For tree-based models with binary classification
                shap_vals = shap_values[1][0]  # Get SHAP values for positive class, first sample
            else:
                # For linear models or when shap_values is a 2D array
                if len(shap_values.shape) == 2:
                    shap_vals = shap_values[0]  # First sample
                else:
                    shap_vals = shap_values
            
            # Create waterfall plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_vals,
                    base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                    data=input_data.values[0],
                    feature_names=features
                ),
                show=False
            )
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error generating SHAP explanation: {str(e)}")
            st.info("SHAP explanation could not be generated for this prediction.")

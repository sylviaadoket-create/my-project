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
            # Create explainer based on model type
            if selected_model_name in ['Random Forest', 'Gradient Boosting']:
                explainer = shap.TreeExplainer(model)
            elif selected_model_name == 'Logistic Regression':
                explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
            else:
                try:
                    explainer = shap.TreeExplainer(model)
                except:
                    explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(input_data)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # For some models, SHAP returns a list
                shap_vals = shap_values[1][0]  # Get SHAP values for positive class, first sample
                base_value = explainer.expected_value[1]
            elif len(shap_values.shape) == 3:
                # For tree-based models with binary classification (n_samples, n_features, n_classes)
                shap_vals = shap_values[0, :, 1]  # First sample, all features, positive class
                base_value = explainer.expected_value[1]
            else:
                # For linear models or when shap_values is a 2D array
                if len(shap_values.shape) == 2:
                    shap_vals = shap_values[0]  # First sample
                else:
                    shap_vals = shap_values
                base_value = explainer.expected_value
            
            # Create waterfall plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_vals,
                    base_values=base_value,
                    data=input_data.values[0],
                    feature_names=features
                ),
                show=False
            )
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error generating SHAP explanation: {str(e)}")
            st.info("SHAP explanation could not be generated for this prediction.")

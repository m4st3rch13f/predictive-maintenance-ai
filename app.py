import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load models and scalers
scaler_anomaly = joblib.load("models/anomaly_scaler.pkl")
isolation_forest = joblib.load("models/isolation_forest_model.pkl")
dbscan_model = joblib.load("models/dbscan_model.pkl")
lstm_model = load_model("models/lstm_autoencoder_model.keras")

classifier = joblib.load("models/final_rf_classifier.pkl")
rf_scaler = joblib.load("models/rf_scaler.pkl")
label_encoder = joblib.load("models/rf_label_encoder.pkl")

code_map = {
    'P0133': 'Fuel System Issue - O2 sensor circuit slow response',
    'C0300': 'Ignition System Problem - Random misfire detected',
    'P0079P2004P3000': 'Valve Timing/Intake Control - Multiple system faults',
    'P0079C1004P3000': 'Valve Timing/Intake Control - Actuator circuit issues',
    'P0078U1004P3000': 'Valve Timing/Intake Control - Electrical malfunction',
    'P0078B0004P3000': 'Valve Timing/Intake Control - Performance issue',
    'P0079P1004P3000': 'Valve Timing/Intake Control - Mechanical fault',
    'P007EP2036P18D0': 'Valve Timing/Intake Control - Position sensor fault',
    'P007EP2036P18E0': 'Valve Timing/Intake Control - Calibration needed',
    'P007EP2036P18F0': 'Valve Timing/Intake Control - Stuck valve detected',
    'P007FP2036P18D0': 'Valve Timing/Intake Control - Sensor range issue',
    'P007FP2036P18E0': 'Valve Timing/Intake Control - Timing over-advanced',
    'P007FP2036P18F0': 'Valve Timing/Intake Control - Timing over-retarded',
}

def check_null_values(input_dict):
    """Check for any null/None values in the input"""
    errors = []
    for field, value in input_dict.items():
        if value is None or (isinstance(value, (float, int)) and np.isnan(value)):
            errors.append(f"❌ {field.replace('_', ' ').title()}: Value cannot be empty")
    return errors

st.set_page_config(page_title="Vehicle Health Analyzer", layout="centered")
st.title("🚗 AI-Powered Predictive Maintenance")
st.markdown("Provide vehicle sensor data to detect anomalies and classify potential faults.")

with st.form("input_form"):
    st.subheader("🔧 Vehicle Sensor Input")
    col1, col2 = st.columns(2)

    with col1:
        engine_coolant_temp = st.number_input("Engine Coolant Temp (°C)", value=85.0)
        engine_rpm = st.number_input("Engine RPM", value=2000.0)
        engine_load = st.slider("Engine Load (%)", 0.0, 100.0, 30.0)
        air_intake_temp = st.number_input("Air Intake Temp (°C)", value=30.0)
        intake_manifold_pressure = st.number_input("Intake Manifold Pressure", value=100.0)
        short_term_trim = st.slider("Short Term Fuel Trim Bank 1 (%)", -100.0, 100.0, 0.0)

    with col2:
        throttle_pos = st.slider("Throttle Position (%)", 0.0, 100.0, 20.0)
        car_year = st.number_input("Car Year", value=2015)
        minute = st.slider("Minute", 0, 59, 30)
        hour = st.slider("Hour", 0, 23, 14)
        day_of_week = st.selectbox("Day of Week", list(range(7)))
        month = st.selectbox("Month", list(range(1, 13)))
        year = st.selectbox("Year", [2016, 2017, 2018])

    submitted = st.form_submit_button("Run Analysis")

if submitted:
    # Create input dictionary
    input_dict = {
        'ENGINE_COOLANT_TEMP': engine_coolant_temp,
        'ENGINE_RPM': engine_rpm,
        'ENGINE_LOAD': engine_load,
        'AIR_INTAKE_TEMP': air_intake_temp,
        'INTAKE_MANIFOLD_PRESSURE': intake_manifold_pressure,
        'SHORT TERM FUEL TRIM BANK 1': short_term_trim,
        'THROTTLE_POS': throttle_pos,
        'CAR_YEAR': car_year,
        'MIN': minute,
        'HOURS': hour,
        'DAYS_OF_WEEK': day_of_week,
        'MONTHS': month,
        'YEAR': year
    }

    # Check for zero values in critical fields
    zero_fields = []
    critical_fields = [
        'ENGINE_COOLANT_TEMP', 
        'ENGINE_RPM', 
        'INTAKE_MANIFOLD_PRESSURE',
        'CAR_YEAR'
    ]
    
    for field in critical_fields:
        if input_dict[field] == 0:
            zero_fields.append(field)
    
    if zero_fields:
        st.error("### Error: Invalid Zero Values Detected")
        st.write("The following fields cannot be zero:")
        for field in zero_fields:
            st.markdown(f"- {field.replace('_', ' ').title()}")
        st.warning("Please provide valid values before submitting.")
        st.stop()  # Stop execution if zeros are found
    else:
        # Proceed with analysis
        input_data = pd.DataFrame([list(input_dict.values())], 
                                columns=list(input_dict.keys()))

        # Scale for anomaly detection
        X_scaled_anomaly = scaler_anomaly.transform(input_data)
        X_lstm = X_scaled_anomaly.reshape((X_scaled_anomaly.shape[0], 1, X_scaled_anomaly.shape[1]))

        # Run models
        iso_result = isolation_forest.predict(X_scaled_anomaly)[0]
        dbscan_result = 1 if dbscan_model.fit_predict(X_scaled_anomaly)[0] == -1 else 0
        lstm_recon = lstm_model.predict(X_lstm)
        lstm_err = np.mean(np.abs(X_lstm - lstm_recon), axis=(1, 2))[0]
        lstm_result = int(lstm_err > 0.05)

        # Ensemble
        anomaly_score = int((iso_result == -1)) + dbscan_result + lstm_result
        consensus = int(anomaly_score >= 2)

        # Classification
        X_scaled_class = rf_scaler.transform(input_data)
        pred_probs = classifier.predict_proba(X_scaled_class)[0]
        pred_label = classifier.predict(X_scaled_class)[0]
        problem = label_encoder.inverse_transform([pred_label])[0]
        confidence = round(np.max(pred_probs), 2)

        # Results - Anomaly Detection
        st.subheader("🔍 Analysis Results")
        
        with st.container():
            st.markdown("### 🚨 Anomaly Detection")
            anomaly_cols = st.columns(4)
            anomaly_cols[0].metric("Isolation Forest", 
                                 "Anomaly" if iso_result == -1 else "Normal", 
                                 "⚠️ Anomaly detected" if iso_result == -1 else "✅ No anomaly")
            anomaly_cols[1].metric("DBSCAN", 
                                 "Anomaly" if dbscan_result == 1 else "Normal", 
                                 "⚠️ Anomaly detected" if dbscan_result == 1 else "✅ No anomaly")
            anomaly_cols[2].metric("LSTM Autoencoder", 
                                 "Anomaly" if lstm_result == 1 else "Normal", 
                                 "⚠️ Anomaly detected" if lstm_result == 1 else "✅ No anomaly")
            anomaly_cols[3].metric("Consensus", 
                                 "Anomaly" if consensus == 1 else "Normal", 
                                 "❗Anomaly verified" if consensus == 1 else "✅ No anomaly")
            
        # Fault Classification
        with st.container():
            st.markdown("### 🔧 Fault Diagnosis")
            
            # Get the explanation
            explanation = code_map.get(problem, "No detailed diagnostic information available")
            
            # Create a nice alert box
            if consensus == 1:
                if confidence > 0.7:
                    st.error(f"""
                    **Diagnosed Issue:** {problem}  
                    **Severity:** {explanation.split(' - ')[0]}  
                    **Description:** {explanation.split(' - ')[1] if ' - ' in explanation else explanation}  
                    **Confidence:** {confidence*100:.0f}%
                    """)
                else:
                    st.warning(f"""
                    **Potential Issue:** {problem}  
                    **Possible Cause:** {explanation.split(' - ')[0]}  
                    **Description:** {explanation.split(' - ')[1] if ' - ' in explanation else explanation}  
                    **Confidence:** {confidence*100:.0f}%  
                    *Note: Low confidence - manual verification recommended*
                    """)
            else:
                st.success("""
                **Diagnosis:** No significant issues detected  
                **Status:** All systems normal  
                *Minor anomalies may not indicate actual problems*
                """)

        # Option to download results
        st.markdown("---")
        st.markdown("### 📊 Detailed Technical Data")
        with st.expander("View raw analysis data"):
            results_df = pd.DataFrame([{
                **input_data.iloc[0].to_dict(),
                'IF_Anomaly': int(iso_result == -1),
                'DBSCAN_Anomaly': dbscan_result,
                'LSTM_Anomaly': lstm_result,
                'Consensus_Anomaly': consensus,
                'Predicted_Problem': problem,
                'Prediction_Confidence': confidence,
                'Problem_Explanation': explanation
            }])
            st.dataframe(results_df)
            
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Full Results as CSV", csv, 
                             file_name="vehicle_analysis_results.csv", 
                             mime='text/csv')
import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Bug Predictor", page_icon="🐞", layout="centered")
st.title("🐞 Software Defect Prediction (v0.01)")
st.markdown("### Enter code metrics below")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Build paths to model and scaler
model_path = os.path.join(parent_dir, "models", "model.pkl")
scaler_path = os.path.join(parent_dir, "models", "scaler.pkl")

# Check if files exist
if not os.path.exists(model_path):
    st.error(f"Model not found at {model_path}")
    st.stop()
if not os.path.exists(scaler_path):
    st.error(f"Scaler not found at {scaler_path}")
    st.stop()

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Feature names (21 features from KC2 dataset)
feature_names = ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 'b', 't',
                 'lOCode', 'lOComment', 'lOBlank', 'lOCodeAndComment', 'uniq_Op', 'uniq_Opnd',
                 'total_Op', 'total_Opnd', 'branchCount']

with st.form("prediction_form"):
    st.subheader("📊 Code Metrics (normalized 0 to 1)")
    inputs = []
    col1, col2 = st.columns(2)
    for i, name in enumerate(feature_names):
        if i % 2 == 0:
            with col1:
                val = st.slider(name, 0.0, 1.0, 0.5, key=name)
        else:
            with col2:
                val = st.slider(name, 0.0, 1.0, 0.5, key=name)
        inputs.append(val)
    submitted = st.form_submit_button("🔍 Predict Defect", type="primary")

if submitted:
    input_array = np.array([inputs])
    input_scaled = scaler.transform(input_array)
    prob = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]
    if pred == 1:
        st.error(f"## ❌ **BUG LIKELY!**\n\nDefect Probability: {prob:.1%}")
        st.warning("⚠️ This module may contain bugs. Consider refactoring.")
    else:
        st.success(f"## ✅ **SAFE**\n\nNo Defect Probability: {1-prob:.1%}")
        st.info("👍 No bug predicted. Good quality code.")
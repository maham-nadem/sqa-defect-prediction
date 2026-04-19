import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Bug Predictor", page_icon="🐞", layout="centered")
st.title("🐞 Software Defect Prediction (v0.1)")
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

# Feature names (21 features) with readable labels and tooltips
feature_info = [
    ("Lines of Code", "Total number of lines of code. Higher value = more complex."),
    ("Cyclomatic Complexity", "Number of independent paths through code. Measures decision complexity."),
    ("Essential Complexity", "Structured programming complexity. Higher = harder to restructure."),
    ("Design Complexity", "How much the module's design is intertwined with others."),
    ("Total Operators + Operands", "Sum of all operators and operands in the module."),
    ("Halstead Volume", "Code volume (size * complexity). Higher = more information."),
    ("Program Level", "Efficiency of the code. Higher is better."),
    ("Difficulty", "How hard the code is to understand. Higher = more difficult."),
    ("Intelligence", "Algorithm quality. Higher = better algorithm."),
    ("Effort", "Estimated effort to write/understand the code."),
    ("Number of Branches", "Total conditional branches (if, switch, loops)."),
    ("Total Tokens", "Total number of tokens in the code."),
    ("Executable Lines of Code", "Lines that actually execute (excluding comments, blanks)."),
    ("Comment Lines", "Lines that are comments. Useful for documentation quality."),
    ("Blank Lines", "Empty lines. Usually for readability."),
    ("Lines with Code & Comment", "Lines that contain both code and a comment."),
    ("Unique Operators", "Distinct operator types used (e.g., +, -, =, if)."),
    ("Unique Operands", "Distinct variable/constant names used."),
    ("Total Operators", "Total occurrences of all operators."),
    ("Total Operands", "Total occurrences of all operands."),
    ("Branch Count", "Total number of branches in control flow.")
]

with st.form("prediction_form"):
    st.subheader("📊 Code Metrics (normalized 0 to 1)")
    inputs = []
    col1, col2 = st.columns(2)
    for i, (label, tooltip) in enumerate(feature_info):
        if i % 2 == 0:
            with col1:
                val = st.slider(label, 0.0, 1.0, 0.5, help=tooltip, key=label)
        else:
            with col2:
                val = st.slider(label, 0.0, 1.0, 0.5, help=tooltip, key=label)
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
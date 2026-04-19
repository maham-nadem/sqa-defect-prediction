import streamlit as st
import requests
import json

# Page config
st.set_page_config(page_title="Bug Predictor", page_icon="🐞", layout="centered")

# Title
st.title("🐞 Software Defect Prediction")
st.markdown("### Enter your code metrics below and find out if the module has a bug.")

# Input fields - simple and clean
with st.form("prediction_form"):
    st.subheader("📊 Code Metrics (normalized 0 to 1)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        loc = st.slider("Lines of Code (LOC)", 0.0, 1.0, 0.5, help="Higher value = more code lines")
        cyclo = st.slider("Cyclomatic Complexity", 0.0, 1.0, 0.5, help="Measures code complexity")
        length = st.slider("Length", 0.0, 1.0, 0.5)
        volume = st.slider("Volume", 0.0, 1.0, 0.5)
        difficulty = st.slider("Difficulty", 0.0, 1.0, 0.5)
    
    with col2:
        fan_in = st.slider("Fan-In", 0.0, 1.0, 0.5, help="Number of inputs")
        fan_out = st.slider("Fan-Out", 0.0, 1.0, 0.5, help="Number of outputs")
        num_ops = st.slider("Operators Count", 0.0, 1.0, 0.5)
        num_opnds = st.slider("Operands Count", 0.0, 1.0, 0.5)
        branch = st.slider("Branch Count", 0.0, 1.0, 0.5)
    
    submitted = st.form_submit_button("🔍 Predict Defect", type="primary")

# When user clicks predict
if submitted:
    payload = {
        "LOC": loc, "CYCLO": cyclo, "LENGTH": length,
        "VOLUME": volume, "DIFFICULTY": difficulty,
        "INT_FAN_IN": fan_in, "INT_FAN_OUT": fan_out,
        "NUM_OPERATORS": num_ops, "NUM_OPERANDS": num_opnds,
        "BRANCH_COUNT": branch
    }
    
    try:
        # Call API
        response = requests.post("http://127.0.0.1:8001/predict", json=payload, timeout=5)
        if response.status_code == 200:
            result = response.json()
            prob_defect = result["probability"]
            is_defect = result["defect"]
            
            # Show result with big emoji
            if is_defect == 1:
                st.error(f"## ❌ **BUG LIKELY!**")
                st.markdown(f"**Defect Probability:** {prob_defect:.1%}")
                st.warning("⚠️ This module may contain bugs. Consider refactoring.")
            else:
                st.success(f"## ✅ **SAFE**")
                st.markdown(f"**Defect Probability:** {1-prob_defect:.1%}")
                st.info("👍 No bug predicted. Good quality code.")
        
        
        else:
          st.error("API error")
          st.write("Status Code:", response.status_code)
          st.write("Response:", response.text)
    except Exception as e:
        st.error(f"Cannot connect to API. Start API first: `python api.py`")
        st.info("Keep the API terminal running and then refresh this app.")
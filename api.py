from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import numpy as np

app = FastAPI(title="Defect Prediction API")

# Load model
model_path = os.path.join("models", "model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model = joblib.load(model_path)

class DefectInput(BaseModel):
    LOC: float
    CYCLO: float
    LENGTH: float
    VOLUME: float
    DIFFICULTY: float
    INT_FAN_IN: float
    INT_FAN_OUT: float
    NUM_OPERATORS: float
    NUM_OPERANDS: float
    BRANCH_COUNT: float

@app.post("/predict")
def predict(data: DefectInput):
    try:
        # Use model_dump() instead of dict()
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]
        # Convert numpy types to Python native
        return {"defect": int(pred), "probability": float(round(prob, 3))}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def root():
    return {"message": "Defect Prediction API is running. Use /predict endpoint."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
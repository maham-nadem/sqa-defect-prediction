# ==================== CALIBRATED DEFECT PREDICTION MODEL ====================
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib

# Load dataset
data = pd.read_csv(r"C:\Users\Bajwa\Desktop\SQA\sqa-defect-prediction\data\data.csv")
data = data.dropna()

# Features and target
target = "DEFECT_LABEL"
X = data.drop(columns=[target])
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Base XGBoost model
base_model = XGBClassifier(
    scale_pos_weight=2.07,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42
)
base_model.fit(X_train, y_train)

# Calibrate probabilities
print("Calibrating probabilities...")
calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
calibrated_model.fit(X_train, y_train)

# Evaluate
y_pred = calibrated_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, target_names=['No Defect', 'Defect']))

# Save model
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "model.pkl")
joblib.dump(calibrated_model, model_path)
print(f"Model saved to {model_path}")

# Test zero input
zero_input = pd.DataFrame([[0.0]*10], columns=X.columns)
zero_prob = calibrated_model.predict_proba(zero_input)[0][1]
print(f"Zero input defect probability: {zero_prob:.3f}")
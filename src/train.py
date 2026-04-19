# ==================== TRAIN ON KC2 DATASET ====================
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
data = pd.read_csv(r"C:\Users\Bajwa\Desktop\SQA\sqa-defect-prediction\data\data.csv")

# Convert target 'problems' (no/yes) to binary: 0 = no defect, 1 = defect
data['defect'] = (data['problems'] == 'yes').astype(int)
data.drop('problems', axis=1, inplace=True)

# Select relevant features (all numeric columns except defect)
feature_cols = ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 'b', 't',
                'lOCode', 'lOComment', 'lOBlank', 'lOCodeAndComment', 'uniq_Op', 'uniq_Opnd',
                'total_Op', 'total_Opnd', 'branchCount']
X = data[feature_cols]
y = data['defect']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))

# Test zero input (all features 0) and high input (all features 1)
zero_input = np.zeros((1, len(feature_cols)))
high_input = np.ones((1, len(feature_cols)))
zero_prob = model.predict_proba(zero_input)[0][1]
high_prob = model.predict_proba(high_input)[0][1]
print(f"Zero input defect probability: {zero_prob:.3f}")
print(f"High input defect probability: {high_prob:.3f}")

# Save model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Model and scaler saved to models/")
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading dataset...")
df = pd.read_csv("../data/ai4i2020.csv")

print("Cleaning data...")
df = df.drop(columns=["UDI", "Product ID"])

target = "Machine failure"

df = pd.get_dummies(df, columns=["Type"], drop_first=True)

df = df.drop(columns=["TWF", "HDF", "PWF", "OSF", "RNF"])

X = df.drop(columns=[target])
y = df[target]

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "../models/scaler.pkl")

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)

print("Training XGBoost...")
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb.fit(X_train_scaled, y_train)

print("\nMODEL EVALUATION\n")

models = {"Random Forest": rf, "XGBoost": xgb}
best_model = None
best_score = 0

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    if acc > best_score:
        best_score = acc
        best_model = model

print("Saving best model...")
joblib.dump(best_model, "../models/failure_model.pkl")

joblib.dump(X.columns.tolist(), "../models/feature_names.pkl")

print("Training complete! Model saved in models folder.")

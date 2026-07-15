import pandas as pd
import joblib
import shap

print("📥 Loading model and data...")

# Load saved files
model = joblib.load("../models/failure_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
feature_names = joblib.load("../models/feature_names.pkl")

# Load dataset again
df = pd.read_csv("../data/ai4i2020.csv")
df = df.drop(columns=["UDI", "Product ID"])
df = pd.get_dummies(df, columns=["Type"], drop_first=True)
df = df.drop(columns=["TWF", "HDF", "PWF", "OSF", "RNF"])

X = df.drop(columns=["Machine failure"])

# Scale
X_scaled = scaler.transform(X)

print("🔍 Creating SHAP explainer...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)

print("📊 Generating global feature importance plot...")
shap.summary_plot(shap_values, X, plot_type="bar")

print("📈 Generating detailed SHAP plot...")
shap.summary_plot(shap_values, X)

print("✅ SHAP analysis complete.")

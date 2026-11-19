"""
predict_from_mlflow.py
Loads the Champion sentiment model from MLflow Model Registry via alias and generates predictions on unseen data.
Student: Sujal Chopra (1MS25CI125)
Department: CSE (AIML), Ramaiah Institute of Technology
"""

import os
import mlflow
import pandas as pd
from mlflow import MlflowClient

# MLflow tracking configuration
mlflow.set_tracking_uri("http://127.0.0.1:5000")
NAME = "sentiment-model"
client = MlflowClient()

# Verify unseen evaluation data
new_data_path = "data/new_data.csv"
if not os.path.exists(new_data_path):
    raise FileNotFoundError(f"Missing new data file: {new_data_path}")

new_df = pd.read_csv(new_data_path)

print("=" * 70)
print(f"MLFLOW MODEL INFERENCE — STUDENT: Sujal Chopra (1MS25CI125)")
print(f"Registered Model Name: {NAME}")
print("=" * 70)

# 1. Load Champion using the alias URI: models:/sentiment-model@champion
alias = "champion"
champion_mv = client.get_model_version_by_alias(NAME, alias)
champion_version = champion_mv.version
hf_model_id = champion_mv.tags.get("hf_model_id", "Unknown")
student_usn = champion_mv.tags.get("student_usn", "1MS25CI125")

print(f"\n[+] Loading Model using Alias URI: 'models:/{NAME}@{alias}'")
print(f"    - Model Version     : {champion_version}")
print(f"    - Hugging Face ID   : {hf_model_id}")
print(f"    - Tagged Student USN: {student_usn}")

# Load via MLflow pyfunc without direct Hugging Face imports
champion_model = mlflow.pyfunc.load_model(f"models:/{NAME}@{alias}")

# Generate predictions
print("\n[+] Generating predictions on 8 new unseen test examples...")
preds_df = champion_model.predict(new_df[["text"]])

# Save outputs
os.makedirs("outputs", exist_ok=True)
output_path = "outputs/new_data_predictions.csv"
preds_df.to_csv(output_path, index=False)
print(f"[+] Saved Champion predictions to: {output_path}")

# Display clean prediction table
print("\n" + "=" * 70)
print("PREDICTION RESULTS (CHAMPION MODEL)")
print("=" * 70)
display_df = preds_df[["text", "prediction", "sentiment"]].copy()
display_df.columns = ["Input Text", "Pred (0/1)", "Sentiment"]
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)
print(display_df.to_string(index=False))
print("=" * 70)

# Optional: verify Challenger alias can also be dynamically loaded
challenger_mv = client.get_model_version_by_alias(NAME, "challenger")
print(f"\n[Note] Challenger alias is also active: Version {challenger_mv.version} ({challenger_mv.tags.get('hf_model_id')})")
print("Demonstrated dynamic loading via alias URI without hard-coded version numbers.")


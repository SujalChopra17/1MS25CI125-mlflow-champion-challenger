"""
Lab 11 — Exercise 2: Predict using Model Registry Alias URI
Loads the active Champion model using 'models:/house-price-regressor@champion'
without knowing whether Version 1 or Version 2 was promoted.
Student: Sujal Chopra (1MS25CI125)
"""

import mlflow
from mlflow import MlflowClient
import pandas as pd

def main():
    TRACKING_URI = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(TRACKING_URI)
    MODEL_NAME = "house-price-regressor"
    client = MlflowClient()
    
    print("=" * 70)
    print("Lab 11 — Dynamic Inference via Model Registry Champion Alias")
    print("=" * 70)
    
    # Query registry metadata for the alias
    champ_mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    print(f"[*] Querying Model Registry for alias '@champion'...")
    print(f"    - Loaded Model Version : {champ_mv.version}")
    print(f"    - Underlying Run ID    : {champ_mv.run_id}")
    print(f"    - Model Type Tag       : {champ_mv.tags.get('model_type', 'N/A')}")
    print(f"    - Student USN Tag      : {champ_mv.tags.get('student_usn', 'N/A')}")
    
    # Load model dynamically via alias URI
    CHAMPION_URI = f"models:/{MODEL_NAME}@champion"
    print(f"\n[*] Loading model from alias URI: '{CHAMPION_URI}'")
    champion_model = mlflow.pyfunc.load_model(CHAMPION_URI)
    print("    Model successfully loaded into memory via alias.")
    
    # Predict on test inputs (using float64 to match inferred model schema)
    test_properties = pd.DataFrame({"sqft": [850.0, 1200.0, 1850.0, 2600.0]})
    preds = champion_model.predict(test_properties)
    
    res_df = test_properties.copy()
    res_df["predicted_price_k$"] = [round(float(p), 2) for p in preds]
    
    print("\nInference Results using Champion Model:")
    print(res_df.to_string(index=False))
    print("=" * 70)
    print("Confirmed: The script did not hard-code version numbers. It resolved the Champion alias dynamically.")

if __name__ == "__main__":
    main()

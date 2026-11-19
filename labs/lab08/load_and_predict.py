"""
Lab 08 — Exercise 2: Load Model via URI and Generate Predictions
Loads the packaged model artifact via runs:/ URI and performs inference.
Student: Sujal Chopra (1MS25CI125)
"""

import os
import mlflow
import pandas as pd

def main():
    TRACKING_URI = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(TRACKING_URI)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    uri_file = os.path.join(script_dir, "model_uri.txt")
    if not os.path.exists(uri_file):
        raise FileNotFoundError(f"model_uri.txt not found at {uri_file}. Run model_component.py first.")
        
    with open(uri_file) as f:
        model_uri = f.read().strip()
        
    print("=" * 65)
    print("Lab 08 — Load and Predict with MLflow Model")
    print("=" * 65)
    print(f"Loading Model from URI: {model_uri}")
    
    # Load packaged model using generic pyfunc flavor
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully into memory.")
    
    # New unseen test inputs (square footage of new houses)
    test_inputs = pd.DataFrame({"sqft": [600, 1100, 1600, 2200]})
    
    preds = loaded_model.predict(test_inputs)
    
    results_df = test_inputs.copy()
    results_df["predicted_price_k$"] = [round(p, 2) for p in preds]
    
    print("\nInference Results on New Unseen Properties:")
    print(results_df.to_string(index=False))
    print("=" * 65)

if __name__ == "__main__":
    main()

"""
Lab 09 — Exercise 2: Predict using Custom Hugging Face Model URI
Loads the custom pyfunc sentiment model from runs:/ URI and predicts on new sentences.
Student: Sujal Chopra (1MS25CI125)
"""

import os
import mlflow
import pandas as pd

def main():
    TRACKING_URI = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(TRACKING_URI)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    uri_file = os.path.join(script_dir, "custom_model_uri.txt")
    if not os.path.exists(uri_file):
        raise FileNotFoundError(f"custom_model_uri.txt not found at {uri_file}. Run log_custom_hf_model.py first.")
        
    with open(uri_file) as f:
        model_uri = f.read().strip()
        
    print("=" * 70)
    print("Lab 09 — Custom Hugging Face Model Inference via MLflow PyFunc")
    print("=" * 70)
    print(f"Loading Model from URI: {model_uri}")
    
    # Load custom PyFunc model without directly invoking Hugging Face pipeline
    custom_model = mlflow.pyfunc.load_model(model_uri)
    print("Custom model loaded successfully.")
    
    # Test sentences
    test_df = pd.DataFrame({"text": [
        "The battery life is fantastic and exceeded my expectations.",
        "The software update crashed my computer and erased files.",
        "Great quality for the price, thoroughly impressed with build.",
        "Terrible experience, would not recommend this service to anyone."
    ]})
    
    preds_df = custom_model.predict(test_df)
    
    print("\nInference Output Table:")
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 1000)
    print(preds_df[["text", "prediction", "sentiment", "confidence_score"]].to_string(index=False))
    print("=" * 70)

if __name__ == "__main__":
    main()

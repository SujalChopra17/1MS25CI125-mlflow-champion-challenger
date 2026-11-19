"""
Lab 08 — Exercise 1: Model Packaging with MLflow Models
Trains a linear model, packages it as an MLflow Model component with signature and input example,
and saves the Model URI for downstream serving.
Student: Sujal Chopra (1MS25CI125)
"""

import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def main():
    TRACKING_URI = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("Lab08_MLflow_Models")
    
    # Simple housing size (sqft) to price ($k) dataset
    X = pd.DataFrame({"sqft": [500, 750, 1000, 1250, 1500, 1750, 2000]})
    y = np.array([150, 210, 270, 325, 390, 440, 510])
    
    with mlflow.start_run(run_name="model_component_run") as run:
        run_id = run.info.run_id
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Log model with signature and input example
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X.head(2)
        )
        
        model_uri = f"runs:/{run_id}/model"
        
        mlflow.log_params({
            "model_type": "LinearRegression",
            "n_samples": len(X),
            "student_usn": "1MS25CI125"
        })
        
        mlflow.set_tags({
            "student_name": "Sujal Chopra",
            "model_uri": model_uri
        })
        
        # Save model URI to text file for load_and_predict.py to read
        with open("model_uri.txt", "w") as f:
            f.write(model_uri)
            
        print("=" * 65)
        print("Lab 08 — MLflow Model Component Logged Successfully")
        print("=" * 65)
        print(f"Run ID    : {run_id}")
        print(f"Model URI : {model_uri}")
        print("Model artifact contains MLmodel metadata, conda.yaml, and weights.")
        print("Saved URI to model_uri.txt for Exercise 2.")

if __name__ == "__main__":
    main()

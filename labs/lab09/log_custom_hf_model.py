"""
Lab 09 — Exercise 1: Custom Hugging Face Model Logging with MLflow PyFunc
Wraps Hugging Face Transformers pipeline in a custom mlflow.pyfunc.PythonModel
and logs the packaged model component to the central tracking server.
Student: Sujal Chopra (1MS25CI125)
"""

import os
import mlflow
import pandas as pd
import transformers
import torch

class CustomHFSentimentModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model_id):
        self.model_id = model_id

    def load_context(self, context):
        from transformers import pipeline
        self.pipeline = pipeline("sentiment-analysis", model=self.model_id)

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].astype(str).tolist()
        else:
            texts = [str(x) for x in model_input]
            
        outputs = self.pipeline(texts, truncation=True)
        preds = [1 if out["label"].upper().startswith("POS") else 0 for out in outputs]
        scores = [float(out["score"]) for out in outputs]
        
        return pd.DataFrame({
            "text": texts,
            "prediction": preds,
            "sentiment": ["positive" if p == 1 else "negative" for p in preds],
            "confidence_score": scores,
            "hf_model_id": self.model_id
        })

def main():
    TRACKING_URI = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("Lab09_Customized_HF_Models")
    
    HF_MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
    
    sample_input = pd.DataFrame({"text": [
        "The customer service was exceptional and responsive.",
        "The delivery was delayed and the parcel was damaged."
    ]})
    
    with mlflow.start_run(run_name="log_custom_hf_distilbert") as run:
        run_id = run.info.run_id
        
        # Log custom PythonModel
        model_info = mlflow.pyfunc.log_model(
            artifact_path="hf_custom_model",
            python_model=CustomHFSentimentModel(HF_MODEL_ID),
            input_example=sample_input,
            pip_requirements=["mlflow", "transformers", "torch", "pandas"]
        )
        
        model_uri = f"runs:/{run_id}/hf_custom_model"
        
        mlflow.log_params({
            "hf_model_id": HF_MODEL_ID,
            "framework": "transformers",
            "model_type": "pyfunc_custom",
            "student_usn": "1MS25CI125"
        })
        
        mlflow.set_tags({
            "student_name": "Sujal Chopra",
            "lab": "Lab09"
        })
        
        with open("custom_model_uri.txt", "w") as f:
            f.write(model_uri)
            
        print("=" * 65)
        print("Lab 09 — Custom Hugging Face Model Logged to MLflow")
        print("=" * 65)
        print(f"Run ID    : {run_id}")
        print(f"Model URI : {model_uri}")
        print(f"HF ID     : {HF_MODEL_ID}")
        print("Saved URI to custom_model_uri.txt for Exercise 2.")

if __name__ == "__main__":
    main()

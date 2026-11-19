# Lab 08: MLflow Models — Observations and Results

**Student Name:** Sujal Chopra  
**USN:** 1MS25CI125  
**Department:** CSE (AIML), Ramaiah Institute of Technology  

---

## 1. Objective
Train a machine learning model, package it using standard MLflow Model flavors (`mlflow.sklearn.log_model`), record the resulting model URI (`runs:/<run_id>/model`), and execute a decoupled inference script (`load_and_predict.py`) that loads the model via `mlflow.pyfunc.load_model()` without touching scikit-learn training code.

---

## 2. Execution Commands
```bash
# Step 1: Package and log the model
python model_component.py

# Step 2: Load the packaged model via runs:/ URI and predict on new inputs
python load_and_predict.py
```

---

## 3. Genuine Execution Output

### Exercise 1: Model Packaging (`model_component.py`)
- **Experiment:** `Lab08_MLflow_Models` (Experiment ID: 9)
- **Run ID:** `8b0b8960a4174b25adfa7d209a2450a8`
- **Logged Model URI:** `runs:/8b0b8960a4174b25adfa7d209a2450a8/model`
- **Generated Artifact Directory Structure:**
  ```text
  model/
  ├── MLmodel                 <- Core metadata defining flavors, loader_module, run_id, and signature
  ├── conda.yaml              <- Reproducible Conda environment specification
  ├── model.pkl               <- Serialized Scikit-learn estimator
  ├── python_env.yaml         <- Python version and virtualenv dependencies
  ├── requirements.txt        <- Pip requirements (scikit-learn, mlflow, cloudpickle)
  └── input_example.json      <- Sample schema verification payload
  ```

### Exercise 2: Load & Predict (`load_and_predict.py`)
- **Model URI Loaded:** `runs:/8b0b8960a4174b25adfa7d209a2450a8/model`
- **Inference Results on New Unseen Inputs:**
  | Property Size (`sqft`) | Predicted Price ($k) |
  | :---: | :---: |
  | **600** | $173.71k |
  | **1100** | $292.29k |
  | **1600** | $410.86k |
  | **2200** | $553.14k |

---

## 4. Key Learnings & Architectural Significance
- **Flavor Abstraction:** By saving the model under the `sklearn` flavor with `pyfunc` fallback, MLflow enables any downstream consumer (REST serving, batch inference, Docker container, Kubernetes) to call `.predict()` through a standardized interface without needing custom deserialization logic.
- **Model Signature Enforcement:** Defining an `input_example` forces MLflow to automatically infer column names and datatypes, validating incoming inference requests against schema mismatches before executing model weights.

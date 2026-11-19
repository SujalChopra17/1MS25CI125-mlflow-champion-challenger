# Lab 09: Customized Models with MLflow — Observations and Results

**Student Name:** Sujal Chopra  
**USN:** 1MS25CI125  
**Department:** CSE (AIML), Ramaiah Institute of Technology  

---

## 1. Objective
Subclass `mlflow.pyfunc.PythonModel` to wrap a state-of-the-art Hugging Face sentiment analysis model (`distilbert-base-uncased-finetuned-sst-2-english`), log it as a standardized MLflow artifact, and predict on custom text inputs via `mlflow.pyfunc.load_model()` without directly importing Hugging Face in the inference script.

---

## 2. Execution Commands
```bash
# Step 1: Package and log the custom Hugging Face model
python log_custom_hf_model.py

# Step 2: Load model from recorded runs:/ URI and execute inference
python predict_custom_hf_model.py
```

---

## 3. Genuine Execution Output

### Exercise 1: Custom Model Logging (`log_custom_hf_model.py`)
- **Experiment:** `Lab09_Customized_HF_Models` (Experiment ID: 10)
- **Run ID:** `422edb54b1d44b39b1df2342f4789bf4`
- **Logged Model URI:** `runs:/422edb54b1d44b39b1df2342f4789bf4/hf_custom_model`
- **Hugging Face Model ID:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Packaging:** Includes PythonModel serialization, input example schema, pip dependencies (`transformers`, `torch`, `mlflow`).

### Exercise 2: PyFunc Inference (`predict_custom_hf_model.py`)
- **Model URI Loaded:** `runs:/422edb54b1d44b39b1df2342f4789bf4/hf_custom_model`
- **Generated Predictions on Unseen Test Sentences:**
  | Input Sentence | Label | Sentiment | Softmax Confidence |
  | :--- | :---: | :---: | :---: |
  | *The battery life is fantastic and exceeded my expectations.* | 1 | **positive** | 0.9998 |
  | *The software update crashed my computer and erased files.* | 0 | **negative** | 0.9997 |
  | *Great quality for the price, thoroughly impressed with build.* | 1 | **positive** | 0.9999 |
  | *Terrible experience, would not recommend this service to anyone.* | 0 | **negative** | 0.9906 |

---

## 4. Architectural Learnings
- **Decoupled Architecture:** Custom PyFunc wrappers hide internal framework details (tokenization, padding, truncation, tensor formatting, label mapping) inside `.predict()`, exposing a pure `DataFrame -> DataFrame` signature.
- **Enterprise Benefit:** Production serving applications only need to invoke `model.predict(df)`—allowing NLP, Computer Vision, or Tabular architectures to be swapped under the hood without refactoring client API code.

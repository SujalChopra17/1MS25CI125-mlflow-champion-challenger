# Lab 04: MLflow Logging Functions — Observations and Results

**Student Name:** Sujal Chopra  
**USN:** 1MS25CI125  
**Department:** CSE (AIML), Ramaiah Institute of Technology  

---

## 1. Objective
Utilize MLflow logging functions (`log_param`, `log_params`, `log_metric`, `log_metrics`, `log_artifact`, `log_table`) to log clinical features, regression metrics, and an error table for a Hospital Length of Stay (LoS) prediction model.

---

## 2. Dataset Specification (Exact 20 Patients from Lab Book)
Twenty patient records relating symptoms before admission ($X$) to hospital stay in days ($Y$):
- $P001$ to $P020$ with split ratio: 75% Training (15 patients), 25% Testing (5 patients).

---

## 3. Execution Command
```bash
python hospital_stay_predictor.py
```

---

## 4. Genuine Execution Output
- **Experiment:** `Lab04_Hospital_Stay_Predictor` (ID: 4)
- **Run ID:** `27f2b54a1b2f42f68f6eb87a7aed4294`
- **Number of Training Observations:** 15
- **Number of Testing Observations:** 5
- **Fitted Regression Model:** $\text{Hospital Stay} = 0.8885 \times X + 0.9579$
- **Evaluation Metrics:**
  - **MAE:** `0.4221` days (~10.1 hours)
  - **MSE:** `0.3099`
  - **R² Score:** `0.9690` (indicates 96.9% of variance in hospital stay is explained by symptom duration)

### Test Set Actual vs. Predicted & Error Table
| Patient ID | Symptom Days ($X$) | Actual Hospital Days ($Y$) | Predicted Days | Absolute Error |
| :---: | :---: | :---: | :---: | :---: |
| **P001** | 1 | 2 | 1.85 | 0.15 |
| **P018** | 9 | 9 | 8.95 | 0.05 |
| **P016** | 5 | 6 | 5.40 | 0.60 |
| **P002** | 2 | 3 | 2.73 | 0.27 |
| **P009** | 9 | 10 | 8.95 | 1.05 |

---

## 5. Key Learnings
- `mlflow.log_metrics()` allows batch logging of multiple performance indicators simultaneously.
- Complex tabular artifacts can be logged both as downloadable CSVs (`mlflow.log_artifact`) and as interactive tables rendered natively in the MLflow UI (`mlflow.log_table`), enabling granular auditing of individual sample errors.

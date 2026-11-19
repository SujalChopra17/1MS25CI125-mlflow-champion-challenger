# Lab 11: Model Registry with MLflow — Observations and Results

**Student Name:** Sujal Chopra  
**USN:** 1MS25CI125  
**Department:** CSE (AIML), Ramaiah Institute of Technology  

---

## 1. Objective
Demonstrate the MLflow Model Registry workflow:
1. Register two competitive regression models under a single registered entity (`house-price-regressor`).
2. Implement automated decision logic to designate the best-performing model as `champion` and the alternative as `challenger`.
3. Execute `predict_champion.py` to demonstrate that prediction scripts resolve `models:/<name>@champion` dynamically without hard-coding specific version integers.

---

## 2. Execution Commands
```bash
# Step 1: Train, evaluate, register, and set aliases
python registry.py

# Step 2: Perform decoupled inference using Champion alias
python predict_champion.py
```

---

## 3. Genuine Execution Output

### Exercise 1: Model Registration & Alias Assignment (`registry.py`)
- **Registered Model:** `house-price-regressor`
- **Candidate A (Linear Regression):**
  - Run ID: `4af627c3c0954a0b95fd521fc9ac7618`
  - Version: `1`
  - Performance: `MAE = 15.6218`
  - Assigned Alias: **`@champion`**
- **Candidate B (Ridge Regression with $\alpha=100$):**
  - Run ID: `5804d1cc9861452e8aad0b2d47f1f60e`
  - Version: `2`
  - Performance: `MAE = 15.6219`
  - Assigned Alias: **`@challenger`**

### Exercise 2: Inference via Alias (`predict_champion.py`)
- **Target URI:** `models:/house-price-regressor@champion`
- **Resolved Version:** Version 1 (`LinearRegression`, USN `1MS25CI125`)
- **Predictions on Unseen Housing Inputs:**
  | Property Area (`sqft`) | Predicted Valuation ($k) |
  | :---: | :---: |
  | **850.0** | $175.27k |
  | **1200.0** | $227.38k |
  | **1850.0** | $324.15k |
  | **2600.0** | $435.82k |

---

## 4. Key Learnings & Production Significance
- **Eliminating Hard-coded Versions:** In classic machine learning deployment, updating a model requires changing `v1` to `v2` in prediction scripts or Docker containers. MLflow Aliases (`@champion`) decouple inference clients from model release cycles: moving the alias instantly updates the model served in production with zero code changes or client downtime.
- **Rollback Safety:** If a newly promoted Champion exhibits unforeseen latency regressions or drift in production, moving the `@champion` alias back to the prior version achieves instantaneous rollback.

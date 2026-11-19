# Lab 07: MLflow Tracking Server — Observations and Results

**Student Name:** Sujal Chopra  
**USN:** 1MS25CI125  
**Department:** CSE (AIML), Ramaiah Institute of Technology  

---

## 1. Objective
Demonstrate how multiple independent Python scripts and applications can log runs to a single, shared MLflow Tracking Server and experiment.

---

## 2. Execution Commands
```bash
# Terminal 1: Run Script 1 (Model with Intercept)
python linear_regression_run1.py

# Terminal 2: Run Script 2 (Model without Intercept)
python linear_regression_run2.py
```

---

## 3. Genuine Execution Output
- **Tracking Server:** `http://127.0.0.1:5000`
- **Shared Experiment Name:** `Lab07_Shared_Tracking_Server` (Experiment ID: 8)

| Metric / Parameter | Script 1 (`linear_regression_run1.py`) | Script 2 (`linear_regression_run2.py`) | Observation |
| :--- | :--- | :--- | :--- |
| **Run Name** | `script1_with_intercept` | `script2_without_intercept` | Distinct run names under same experiment. |
| **Run ID** | `3246ec20c0ee4a9da12497ff3e5160be` | `cf2c996acb594854a21f315adc8e81cb` | Recorded independently. |
| **`fit_intercept`** | `True` | `False` | Hyperparameter variation. |
| **Learned Slope** | `2.4370` | `2.8331` | Forcing through origin steepens slope. |
| **Learned Intercept** | `5.3353` | `0.0000` | Forced to 0 in Script 2. |
| **MAE** | **1.6983** | **2.5414** | Intercept improves fit by ~0.84 MAE. |
| **R² Score** | **0.9789** | **0.9387** | Model with intercept explains ~4% more variance. |

---

## 4. Viva Questions & Architectural Answers

### Q1: Do both scripts need separate tracking servers?
**Answer:** **No.** A single MLflow tracking server is designed to handle multiple concurrent clients and scripts across the entire engineering team or organization. Both scripts connect to the same URI (`http://127.0.0.1:5000`) and the server arbitrates concurrent writes to the underlying storage safely.

### Q2: Can both scripts log to the same experiment?
**Answer:** **Yes.** Both scripts targeted the exact same experiment name (`Lab07_Shared_Tracking_Server`). The server appends both runs under that shared experiment, allowing immediate side-by-side comparison in the MLflow UI.

### Q3: Why is a shared tracking server useful for a team?
**Answer:**
1. **Centralized Visibility:** Team members can review peers' experiments, avoiding duplicate training runs and wasted computational resources.
2. **Standardized Benchmarking:** Provides an immutable, shared ground truth where models can be sorted and evaluated on common leaderboards.
3. **Auditing & Governance:** Ensures compliance and lineage tracking by recording who trained which model, with what parameters, dataset version, and commit hash.
4. **Seamless Registry Handoff:** Enables downstream MLOps stages (staging, Champion–Challenger testing, deployment) to pull verified models directly without manual file transfers.

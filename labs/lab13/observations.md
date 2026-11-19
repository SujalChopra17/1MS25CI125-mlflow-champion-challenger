# Lab 13: MLflow Client — Observations and Results

**Student Name:** Sujal Chopra  
**USN:** 1MS25CI125  
**Department:** CSE (AIML), Ramaiah Institute of Technology  

---

## 1. Objective
Use the low-level `mlflow.tracking.MlflowClient` SDK to query tracking server metadata, search experiments, retrieve runs programmatically, and filter the **Top 3 runs with the lowest MAE** using SQL-like filtering and sorting syntax.

---

## 2. Execution Command
```bash
python client_demo.py
```

---

## 3. Genuine Execution Output
- **Tracking Server:** `http://127.0.0.1:5000`
- **Total Experiments Discovered:** 15 active experiments
- **Selected Experiment:** `Lab05_Test_Size_Comparison` (Experiment ID: 5)

### Initial Run Retrieval
```text
 run_id        run_name   status      mae       r2
8d03551e... test_size_40pct FINISHED 8.920565 0.985438
6124c570... test_size_30pct FINISHED 8.751831 0.985313
c8057194... test_size_20pct FINISHED 9.019685 0.983441
```

### Top 3 Runs with Lowest MAE (`order_by=["metrics.mae ASC"]`)
| Rank | Run ID | Run Name | MAE | R² Score | Test Split | Random Seed |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `6124c570b3a648ed8110d422dd2de1c5` | `test_size_30pct` | **8.7518** | 0.9853 | 0.3 | 42 |
| **2** | `8d03551eea124685bd01a8f75b69c666` | `test_size_40pct` | **8.9206** | 0.9854 | 0.4 | 42 |
| **3** | `c8057194e7654a91a4c8d84183d61f02` | `test_size_20pct` | **9.0197** | 0.9834 | 0.2 | 42 |

---

## 4. Key Learnings & API Capabilities
- **Programmatic Control:** While the web UI is ideal for visual inspection, `MlflowClient` enables automated CI/CD pipelines to query models, evaluate thresholds, and trigger automated deployment or alerting scripts.
- **Search Capabilities:** The `search_runs()` method accepts powerful filter expressions (`filter_string="metrics.mae > 0 and params.model_type = 'LinearRegression'"`), enabling complex queries across thousands of historical runs.

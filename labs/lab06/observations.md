# Lab 06: MLflow Autologging — Observations and Results

**Student Name:** Sujal Chopra  
**USN:** 1MS25CI125  
**Department:** CSE (AIML), Ramaiah Institute of Technology  

---

## 1. Objective
Compare automatic experiment tracking via `mlflow.sklearn.autolog()` with explicit manual tracking within the same experiment (`Lab06_Autolog_vs_Manual`).

---

## 2. Execution Command
```bash
python autolog_vs_manual.py
```

---

## 3. Genuine Execution Output
- **Experiment:** `Lab06_Autolog_vs_Manual` (ID: 7)
- **Run 1 (Autologged):** Run ID `558d290ed89f46d19a34d61b01e477e9` (`run_autologged`)
- **Run 2 (Manually Logged):** Run ID `f30ee56409ad45389ce3a44d89121081` (`run_manually_logged`)
- **Evaluation Metrics (Manual):** `test_mae: 3.8758`, `test_r2: 0.9943`, `slope: 4.1032`, `intercept: 7.9142`

---

## 4. In-Depth Comparison: Autologging vs. Manual Logging

| Dimension | `mlflow.sklearn.autolog()` | Explicit Manual Logging |
| :--- | :--- | :--- |
| **Setup Effort** | Single line (`autolog()`); zero boilerplate. | Requires multiple explicit `log_param`, `log_metric`, `log_model` calls. |
| **Parameters Captured** | Automatically captures all framework parameters (`copy_X`, `fit_intercept`, `n_jobs`, `positive`). | Captures precisely the parameters selected by the developer. |
| **Metrics Captured** | Captures default training metrics (`training_score`, `training_mae`, `training_r2`). | Allows custom validation metrics (e.g., holdout `test_mae`, latency). |
| **Model Packaging** | Automatically logs serialized model, conda environment, requirements, and signature. | Developer explicitly specifies signature, input examples, and artifact path. |
| **Control & Customization** | Low customization; may capture unnecessary internal parameters. | Full control over logged names, precision, tags, and structure. |

### Practical Recommendation
Use `autolog()` during early prototyping and fast experimentation to prevent accidental omission of hyperparameters. Shift to explicit manual logging (or hybrid logging) for production benchmarks and Champion–Challenger comparisons where strict metrics (such as latency guardrails and task-specific Macro-F1) are mandated.

# Lab 10: Model Evaluation — Observations and Results

**Student Name:** Sujal Chopra  
**USN:** 1MS25CI125  
**Department:** CSE (AIML), Ramaiah Institute of Technology  

---

## 1. Objective
Demonstrate experimental schema design by creating two separate MLflow experiments:
1. `Linear Regression Experiment` (evaluates baseline with single metric: `test_mae`)
2. `Linear Regression Metrics Experiment` (evaluates multi-dimensional regression metrics: `test_mae`, `test_mse`, `test_rmse`, `test_r2`)

---

## 2. Execution Command
```bash
python evaluate_experiments.py
```

---

## 3. Genuine Execution Output

### Experiment 1: Baseline Single-Metric Evaluation
- **Experiment Name:** `Linear Regression Experiment` (Experiment ID: 11)
- **Run ID:** `e7101deda142456d9a99eb34c7394891` (`baseline_mae_eval`)
- **Metric Captured:** `test_mae: 3.0010`

### Experiment 2: Comprehensive Multi-Metric Evaluation
- **Experiment Name:** `Linear Regression Metrics Experiment` (Experiment ID: 12)
- **Run ID:** `7472a52a2fd146f2be7765913235e701` (`comprehensive_metrics_eval`)
- **Metrics Captured:**
  - **MAE:** `3.0010` (Average magnitude of errors in the same unit)
  - **MSE:** `13.7582` (Penalizes larger outliers quadratically)
  - **RMSE:** `3.7092` (Square root of MSE, interpretable in original unit)
  - **R² Score:** `0.9906` (Proportion of explained variance)

---

## 4. Key Learnings & Comparison Analysis
- **Why Single Metric is Insufficient:** Relying exclusively on MAE hides the presence of severe outliers. Because MSE squares residuals, the discrepancy between MAE (3.0010) and RMSE (3.7092) immediately reveals whether a few extreme errors exist.
- **Experiment Segmentation:** Segmenting experiments based on metric maturity helps teams separate initial exploratory runs from official deployment candidate evaluation runs.

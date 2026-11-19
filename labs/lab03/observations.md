# Lab 03: MLflow Tracking — Observations and Results

**Student Name:** Sujal Chopra  
**USN:** 1MS25CI125  
**Department:** CSE (AIML), Ramaiah Institute of Technology  

---

## 1. Tracking Server Configuration
- **Server Tracking URI:** `http://127.0.0.1:5000`
- **Backend Store:** SQLite (`sqlite:///mlflow.db`)
- **Artifact Store:** `./mlartifacts`
- **Command to launch the server:**
  ```bash
  python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
  ```

---

## 2. Exercise 1: Server-Tracked Linear Regression
### Execution Command
```bash
python tracking_linear_regression.py
```
### Genuine Execution Output
- **Experiment:** `Lab03_Tracking_Linear_Regression` (ID: 2)
- **Run ID:** `a9a40b35e63544919d3eb2bdd711d327`
- **Run Name:** `central_tracked_lr_run`
- **Parameters Logged:** `fit_intercept: True`, `random_state: 42`, `server_uri: http://127.0.0.1:5000`, `student_usn: 1MS25CI125`, `test_size: 0.25`
- **Metrics Logged:** `learned_coef: 3.2082`, `learned_intercept: 14.5421`, `mae: 5.7882`, `r2_score: 0.9923`
- **Tags Logged:** `entity: LinearRegression`, `lab: Lab03`, `student_name: Sujal Chopra`

---

## 3. Detailed Explanation of MLflow UI Entities
When navigating the MLflow UI at `http://127.0.0.1:5000`, the core tracking abstractions include:

1. **Experiments:** Logical containers that group together related runs for a particular machine learning task or objective. Allows team members to separate exploratory runs from production benchmarks.
2. **Runs:** An individual execution of machine learning code. Each run records execution duration, git commit, source file, status (`FINISHED`, `FAILED`), and associated parameters/metrics.
3. **Parameters:** Immutable key-value inputs that define model configuration and hyperparameters (e.g., `learning_rate`, `test_size`, `fit_intercept`). Logged once per run.
4. **Metrics:** Numeric, scalar values recorded during or after training (e.g., `mae`, `loss`, `r2_score`). MLflow supports logging metric histories across steps or epochs to render interactive training curves.
5. **Artifacts:** Heavy output files produced by the run (e.g., serialized model weights, confusion matrix images, evaluation CSV files, tensorboard logs) stored in artifact repositories.
6. **Tags:** Flexible, queryable key-value strings used to label and annotate runs (e.g., `student_usn`, `model_role`, `git_branch`, `author`). Can be updated post-execution.

---

## 4. Exercise 2: Server-Tracked Delivery Time Predictor
### Execution Command
```bash
python tracking_delivery_predictor.py
```
### Genuine Execution Output
- **Experiment:** `Lab03_Delivery_Predictor_Server` (ID: 3)
- **Run ID:** `5d7fae1f061a443aa71d3d27a42b1ecb`
- **Run Name:** `delivery_server_run`
- **MAE:** `2.0000` min
- **Artifact Logged:** `outputs/delivery_server_preds.csv`
- **UI Verification:** Visible in MLflow UI under Experiment 3 with full parameter and metric tracking.

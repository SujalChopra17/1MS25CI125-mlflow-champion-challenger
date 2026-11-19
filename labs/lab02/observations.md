# Lab 02: Introduction to MLflow — Observations and Results

**Student Name:** Sujal Chopra  
**USN:** 1MS25CI125  
**Department:** CSE (AIML), Ramaiah Institute of Technology  

---

## 1. Exercise 1: First MLflow Run & `mlruns/` File Structure Analysis

### Objective
Initialize a local file-based MLflow run, log parameters, metrics, tags, and artifacts, and document the directory hierarchy created under `mlruns/`.

### Execution Command
```bash
python firstmlflow_run.py
```

### Genuine Execution Output
- **Experiment:** `Lab02_First_MLflow_Run`
- **Run ID:** `f4ea3d73b3084a3389ebda64995fd01a`
- **Logged Parameters:** `batch_size: 32`, `learning_rate: 0.01`, `optimizer: adam`, `student_usn: 1MS25CI125`
- **Logged Metrics:** `accuracy: 0.9450`, `loss: 0.1452`
- **Logged Tags:** `environment: local_development`, `student_name: Sujal Chopra`
- **Logged Artifact:** `run_summary.txt`

### Detailed Analysis of `mlruns/` Directory Contents
When tracking with the file store backend, MLflow organizes data systematically:
1. **`mlruns/<experiment_id>/meta.yaml`**: Contains experiment metadata including `experiment_id`, `name`, `artifact_location`, and `lifecycle_stage`.
2. **`mlruns/<experiment_id>/<run_id>/meta.yaml`**: Contains run-specific lifecycle information: `run_uuid`, `run_name`, `status` (`FINISHED`), `start_time`, `end_time`, and `user_id`.
3. **`params/` directory**: Stores each logged parameter as an individual text file named after the parameter key (e.g., `learning_rate`), with its value as file content.
4. **`metrics/` directory**: Stores each logged metric as a newline-delimited record containing `timestamp`, `value`, and `step` to enable training curves and progress plotting.
5. **`artifacts/` directory**: Stores all binary and file artifacts (e.g., `run_summary.txt`, model weights, figures, CSVs) attached to the run.
6. **`tags/` directory**: Stores user and system tags as key-value text files (including `mlflow.runName`, `mlflow.user`, `student_name`).

---

## 2. Exercise 2: Delivery Time Predictor with MLflow Tracking

### Objective
Track the Delivery Time Predictor linear regression experiment with MLflow parameters, metrics, and artifact logging.

### Execution Command
```bash
python delivery_time_predictor_mlflow.py
```

### Genuine Execution Output
- **Experiment:** `Lab02_Delivery_Time_Predictor`
- **Run ID:** `9fc66fb071ae487aaea1a06e333a9c5d`
- **Parameters Logged:** `model_family: LinearRegression`, `test_size: 0.2`, `random_state: 42`, `student_usn: 1MS25CI125`
- **Metrics Logged:** `mae: 2.0000`, `model_slope: 4.4000`, `model_intercept: 12.2000`, `n_train_samples: 4`, `n_test_samples: 1`
- **Artifact Logged:** `outputs/delivery_predictions.csv` containing:
  ```csv
  order,distance_km,actual_time_min,predicted_time_min,error_min
  2,2,19,21.0,2.0
  ```

### Key Learnings
- Integrating MLflow introduces zero overhead while replacing fragile manual logging with automated, structured persistence.
- Parameters and metrics can now be queried programmatically or compared via the MLflow UI.

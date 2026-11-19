# Lab 12: MLflow Projects — Observations and Results

**Student Name:** Sujal Chopra  
**USN:** 1MS25CI125  
**Department:** CSE (AIML), Ramaiah Institute of Technology  

---

## 1. Objective
Package a training workflow using the standard `MLproject` file format and execute parameterized CLI runs using `python -m mlflow run` with the local environment manager (`--env-manager local`).

---

## 2. Contents of `MLproject` File
```yaml
name: Lab12_MLproject

entry_points:
  main:
    parameters:
      test_size: {type: float, default: 0.25}
      random_state: {type: int, default: 7}
    command: "python train.py --test_size {test_size} --random_state {random_state}"
```

---

## 3. Execution Commands & Genuine Outputs

### Execution 1: Test Size = 0.25, Random State = 7
```bash
python -m mlflow run . --env-manager local -P test_size=0.25 -P random_state=7
```
**Terminal Output:**
```text
=== Running command 'python train.py --test_size 0.25 --random_state 7' in run with ID '0c742d943461488b9b9a82f89721a061' === 
============================================================
Lab 12 — MLproject Run Execution Complete
============================================================
Parameters : test_size = 0.25, random_state = 7
Samples    : Train = 60, Test = 20
Results    : MAE = 4.9518, R² = 0.9917
Equation   : y = 2.7825 * X + 8.8615
============================================================
=== Run (ID '0c742d943461488b9b9a82f89721a061') succeeded ===
```

### Execution 2: Test Size = 0.40, Random State = 7
```bash
python -m mlflow run . --env-manager local -P test_size=0.40 -P random_state=7
```
**Terminal Output:**
```text
=== Running command 'python train.py --test_size 0.40 --random_state 7' in run with ID '97e51f40e4164afbb904147cc4bdbf64' === 
============================================================
Lab 12 — MLproject Run Execution Complete
============================================================
Parameters : test_size = 0.4, random_state = 7
Samples    : Train = 48, Test = 32
Results    : MAE = 4.8824, R² = 0.9920
Equation   : y = 2.7957 * X + 8.9114
============================================================
=== Run (ID '97e51f40e4164afbb904147cc4bdbf64') succeeded ===
```

---

## 4. Key Learnings
- **Reproducible Execution Standard:** The `MLproject` file specifies unambiguous entry points, typed parameters with defaults, and command templates. Any team member or CI/CD automation can execute the project without knowing internal script arguments.
- **Automated Parameter Tracking:** `mlflow run` intercepts parameters passed through `-P` and logs them automatically to the active run, ensuring parameter immutability and preventing discrepancies between CLI input and logged values.
- **Environment Management:** The `--env-manager` flag supports `local`, `virtualenv`, `conda`, and `docker`, providing seamless portability from personal development to distributed cloud orchestrators (e.g., Kubernetes, Databricks).

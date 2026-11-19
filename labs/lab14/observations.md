# Lab 14: MLflow CLI — Observations and Results

**Student Name:** Sujal Chopra  
**USN:** 1MS25CI125  
**Department:** CSE (AIML), Ramaiah Institute of Technology  

---

## 1. Objective
Demonstrate operational mastery of the MLflow Command Line Interface (CLI) by executing and documenting:
1. Version interrogation (`python -m mlflow --version`)
2. CLI subcommand discovery (`python -m mlflow --help`)
3. Experiment search across the tracking backend (`python -m mlflow experiments search`)

---

## 2. Execution Commands & Terminal Outputs

### Exercise 1: Check MLflow Version
```bash
python -m mlflow --version
```
**Output:**
```text
mlflow, version 3.18.0
```

### Exercise 2: Explore Available Commands
```bash
python -m mlflow --help
```
**Output:**
The command returns the complete list of MLflow CLI modules, including:
- `experiments`: Create, list, delete, and search experiments.
- `runs`: Inspect and manage individual run records.
- `models`: Serve, predict, and package model artifacts locally.
- `server`: Launch the combined Tracking Server and Web UI.
- `run`: Execute parameterized MLproject packages.
- `artifacts`: Direct download/upload commands for remote storage.
- `db`: Database schema migrations and upgrades.

### Exercise 3: Search for Experiments
```bash
python -m mlflow experiments search
```
**Output Table:**
```text
Experiment Id    Name                                  Artifact Location   
---------------  ------------------------------------  --------------------
0                Default                               mlflow-artifacts:/0 
1                hf-sentiment-champion-challenger      mlflow-artifacts:/1 
10               Lab09_Customized_HF_Models            mlflow-artifacts:/10
11               Linear Regression Experiment          mlflow-artifacts:/11
12               Linear Regression Metrics Experiment  mlflow-artifacts:/12
13               Lab11_Model_Registry                  mlflow-artifacts:/13
14               Lab12_MLflow_Projects                 mlflow-artifacts:/14
2                Lab03_Tracking_Linear_Regression      mlflow-artifacts:/2 
3                Lab03_Delivery_Predictor_Server       mlflow-artifacts:/3 
4                Lab04_Hospital_Stay_Predictor         mlflow-artifacts:/4 
5                Lab05_Test_Size_Comparison            mlflow-artifacts:/5 
6                Lab05_Random_State_Sensitivity        mlflow-artifacts:/6 
7                Lab06_Autolog_vs_Manual               mlflow-artifacts:/7 
8                Lab07_Shared_Tracking_Server          mlflow-artifacts:/8 
9                Lab08_MLflow_Models                   mlflow-artifacts:/9 
```

---

## 3. Key Learnings
- **DevOps & CI/CD Automation:** The MLflow CLI allows automation scripts, shell hooks, and Jenkins/GitHub Actions workflows to interface with MLflow without writing Python wrappers.
- **Environment Configuration:** Setting `MLFLOW_TRACKING_URI` allows CLI commands to seamlessly query remote or centralized team tracking servers from any terminal window.

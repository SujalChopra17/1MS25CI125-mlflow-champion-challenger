# MLOps with MLflow — Comprehensive Laboratory Observation Report (Labs 01 – 14)

- **Student Name:** Sujal Chopra
- **Student USN:** 1MS25CI125
- **Institution:** Ramaiah Institute of Technology, Bengaluru
- **Department:** Department of Computer Science & Engineering (Artificial Intelligence & Machine Learning)
- **Course Title:** MLOps with MLflow (Skill Lab)
- **Submission Repository:** `1MS25CI125-mlflow-champion-challenger`

---

## Executive Summary
This document compiles the genuine execution records, output logs, metrics tables, and conceptual learnings for all 14 laboratory exercises specified in the official MLflow Lab Book. Every script has been implemented, executed in a reproducible local environment, and tracked using MLflow Tracking, MLflow Model Packaging, Model Registry, and MLproject standards.

---

## Lab Index
1. [Lab 01 — Introduction to MLOps](#lab-01--introduction-to-mlops)
2. [Lab 02 — Introduction to MLflow](#lab-02--introduction-to-mlflow)
3. [Lab 03 — MLflow Tracking](#lab-03--mlflow-tracking)
4. [Lab 04 — MLflow Logging Functions](#lab-04--mlflow-logging-functions)
5. [Lab 05 — Multiple Experiments & Runs](#lab-05--multiple-experiments--runs)
6. [Lab 06 — MLflow Autologging](#lab-06--mlflow-autologging)
7. [Lab 07 — MLflow Tracking Server](#lab-07--mlflow-tracking-server)
8. [Lab 08 — MLflow Models](#lab-08--mlflow-models)
9. [Lab 09 — Customized Models with MLflow](#lab-09--customized-models-with-mlflow)
10. [Lab 10 — Model Evaluation](#lab-10--model-evaluation)
11. [Lab 11 — Model Registry with MLflow](#lab-11--model-registry-with-mlflow)
12. [Lab 12 — MLflow Projects](#lab-12--mlflow-projects)
13. [Lab 13 — MLflow Client](#lab-13--mlflow-client)
14. [Lab 14 — MLflow CLI](#lab-14--mlflow-cli)

---

## Lab 01 — Introduction to MLOps
- **Objective:** Train baseline linear regression models across different test partition sizes and model a real-world delivery time prediction task.
- **Files:** `labs/lab01/baseline_model.py`, `labs/lab01/delivery_time_predictor.py`
- **Commands:**
  ```bash
  python baseline_model.py
  python delivery_time_predictor.py
  ```
- **Observations:**
  - In Exercise 1, varying test split from 0.20 to 0.40 scaled training samples from 40 down to 30, with MAE increasing from 3.6115 to 3.9307.
  - In Exercise 2, on the 5-sample delivery dataset (Distance: 1, 2, 3, 4, 5 km; Time: 16, 19, 27, 29, 34 min), the learned equation was $\text{Time} = 4.4000 \times \text{Distance} + 12.2000$ with test MAE of 2.0000 min.

---

## Lab 02 — Introduction to MLflow
- **Objective:** Initialize MLflow tracking, log parameters, metrics, tags, and artifacts, and inspect the internal file hierarchy of `mlruns/`.
- **Files:** `labs/lab02/firstmlflow_run.py`, `labs/lab02/delivery_time_predictor_mlflow.py`
- **Commands:**
  ```bash
  python firstmlflow_run.py
  python delivery_time_predictor_mlflow.py
  ```
- **Structure of `mlruns/`:**
  - `meta.yaml`: Run UUID, lifecycle stage, start/end timestamps.
  - `params/`: Individual key-value files for each logged parameter.
  - `metrics/`: Metric progression histories stored with timestamps and step counts.
  - `artifacts/`: Downloadable output files, CSV tables, plots, and models.
  - `tags/`: User, system, and run metadata tags.

---

## Lab 03 — MLflow Tracking
- **Objective:** Set a centralized Tracking URI (`http://127.0.0.1:5000`) and verify UI entities.
- **Files:** `labs/lab03/tracking_linear_regression.py`, `labs/lab03/tracking_delivery_predictor.py`
- **Commands:**
  ```bash
  python tracking_linear_regression.py
  python tracking_delivery_predictor.py
  ```
- **Key Entities Explained:**
  - *Experiments:* Containers grouping related ML runs.
  - *Runs:* Individual executions with execution status and duration.
  - *Parameters:* Immutable model configuration inputs.
  - *Metrics:* Scalar evaluation outputs logged over time.
  - *Artifacts:* Binary model files, figures, and datasets.
  - *Tags:* Searchable, queryable metadata labels.

---

## Lab 04 — MLflow Logging Functions
- **Objective:** Track a clinical Hospital Length of Stay (LoS) model on exact 20-patient data ($P001$ to $P020$) and log an error table.
- **Files:** `labs/lab04/hospital_stay_predictor.py`
- **Command:**
  ```bash
  python hospital_stay_predictor.py
  ```
- **Results:**
  - Training Samples: 15 | Testing Samples: 5
  - Learned Equation: $\text{Stay} = 0.8885 \times \text{Symptoms} + 0.9579$
  - Test MAE: `0.4221` days (~10.1 hours) | $R^2$: `0.9690`
  - Prediction & Error Table:
    | Patient | Symptom Days ($X$) | Actual Stay ($Y$) | Predicted | Error |
    | :---: | :---: | :---: | :---: | :---: |
    | P001 | 1 | 2 | 1.85 | 0.15 |
    | P018 | 9 | 9 | 8.95 | 0.05 |
    | P016 | 5 | 6 | 5.40 | 0.60 |
    | P002 | 2 | 3 | 2.73 | 0.27 |
    | P009 | 9 | 10 | 8.95 | 1.05 |

---

## Lab 05 — Multiple Experiments & Runs
- **Objective:** Organize multi-run hyperparameter and data size studies within targeted experiments.
- **Files:** `labs/lab05/multiple_runs.py`
- **Command:**
  ```bash
  python multiple_runs.py
  ```
- **Results:**
  - Fixed Seed (42) with Test Splits:
    - 20% split: $N_{train}=80$, MAE = 9.0197, $R^2 = 0.9834$
    - 30% split: $N_{train}=70$, MAE = 8.7518, $R^2 = 0.9853$
    - 40% split: $N_{train}=60$, MAE = 8.9206, $R^2 = 0.9854$
  - Fixed Test Split (0.25) across Seeds:
    - Seed 7: MAE = 10.6369, $R^2 = 0.9798$
    - Seed 42: MAE = 9.0632, $R^2 = 0.9829$
    - Seed 100: MAE = 12.3879, $R^2 = 0.9693$

---

## Lab 06 — MLflow Autologging
- **Objective:** Compare automated tracking via `mlflow.sklearn.autolog()` with explicit manual tracking.
- **Files:** `labs/lab06/autolog_vs_manual.py`
- **Command:**
  ```bash
  python autolog_vs_manual.py
  ```
- **Comparison:**
  - *Autologging (`run_autologged`):* Zero boilerplate; automatically captures model parameters, training metrics, serialized model, and signature.
  - *Manual Logging (`run_manually_logged`):* Full developer control over custom test-set MAE (`3.8758`) and $R^2$ (`0.9943`), eliminating unnecessary overhead.

---

## Lab 07 — MLflow Tracking Server
- **Objective:** Demonstrate that multiple independent Python scripts can send runs to a single shared tracking server and experiment.
- **Files:** `labs/lab07/linear_regression_run1.py`, `labs/lab07/linear_regression_run2.py`
- **Commands:**
  ```bash
  python linear_regression_run1.py
  python linear_regression_run2.py
  ```
- **Results:**
  - Script 1 (with intercept): Run ID `3246ec20c0ee4a9da12497ff3e5160be`, MAE: `1.6983`, $R^2$: `0.9789`
  - Script 2 (without intercept): Run ID `cf2c996acb594854a21f315adc8e81cb`, MAE: `2.5414`, $R^2$: `0.9387`
- **Viva Insights:** Scripts do not require separate tracking servers. A shared server enables centralized auditing, leaderboard benchmarking, and team collaboration.

---

## Lab 08 — MLflow Models
- **Objective:** Package a model using standard MLflow Model components and perform decoupled prediction via `runs:/` URI.
- **Files:** `labs/lab08/model_component.py`, `labs/lab08/load_and_predict.py`
- **Commands:**
  ```bash
  python model_component.py
  python load_and_predict.py
  ```
- **Results:**
  - Logged Model URI: `runs:/8b0b8960a4174b25adfa7d209a2450a8/model`
  - Loaded via `mlflow.pyfunc.load_model(model_uri)`
  - Predictions on new square-footages:
    - 600 sqft $\rightarrow$ $173.71k
    - 1100 sqft $\rightarrow$ $292.29k
    - 1600 sqft $\rightarrow$ $410.86k
    - 2200 sqft $\rightarrow$ $553.14k

---

## Lab 09 — Customized Models with MLflow
- **Objective:** Wrap a pretrained Hugging Face sentiment classification model in a custom `mlflow.pyfunc.PythonModel` and predict on new text sentences.
- **Files:** `labs/lab09/log_custom_hf_model.py`, `labs/lab09/predict_custom_hf_model.py`
- **Commands:**
  ```bash
  python log_custom_hf_model.py
  python predict_custom_hf_model.py
  ```
- **Results:**
  - Logged URI: `runs:/422edb54b1d44b39b1df2342f4789bf4/hf_custom_model`
  - Inference via generic `mlflow.pyfunc`:
    - *"The battery life is fantastic and exceeded my expectations."* $\rightarrow$ Positive (99.98% confidence)
    - *"The software update crashed my computer and erased files."* $\rightarrow$ Negative (99.97% confidence)
    - *"Great quality for the price, thoroughly impressed with build."* $\rightarrow$ Positive (99.99% confidence)
    - *"Terrible experience, would not recommend this service to anyone."* $\rightarrow$ Negative (99.06% confidence)

---

## Lab 10 — Model Evaluation
- **Objective:** Contrast single-metric evaluation with comprehensive multi-metric evaluation across distinct experiments.
- **Files:** `labs/lab10/evaluate_experiments.py`
- **Command:**
  ```bash
  python evaluate_experiments.py
  ```
- **Results:**
  - Experiment 1 (`Linear Regression Experiment`): Captured `test_mae: 3.0010`.
  - Experiment 2 (`Linear Regression Metrics Experiment`): Captured `test_mae: 3.0010`, `test_mse: 13.7582`, `test_rmse: 3.7092`, and `test_r2: 0.9906`.

---

## Lab 11 — Model Registry with MLflow
- **Objective:** Register candidate models into the MLflow Model Registry and assign `champion` and `challenger` aliases. Demonstrate that inference scripts load models via alias without hard-coded versions.
- **Files:** `labs/lab11/registry.py`, `labs/lab11/predict_champion.py`
- **Commands:**
  ```bash
  python registry.py
  python predict_champion.py
  ```
- **Results:**
  - Registered Model: `house-price-regressor`
  - Candidate A (LinearRegression, MAE 15.6218) $\rightarrow$ Assigned `@champion` (Version 1)
  - Candidate B (Ridge $\alpha=100$, MAE 15.6219) $\rightarrow$ Assigned `@challenger` (Version 2)
  - `predict_champion.py` loaded `models:/house-price-regressor@champion` dynamically, returning predicted prices ($175.27k, $227.38k, $324.15k, $435.82k) without hard-coding version 1.

---

## Lab 12 — MLflow Projects
- **Objective:** Define a parameterized `MLproject` and execute reproducible training runs with `--env-manager local`.
- **Files:** `labs/lab12/MLproject`, `labs/lab12/train.py`
- **Commands:**
  ```bash
  # Run 1: test_size = 0.25, random_state = 7
  python -m mlflow run . --env-manager local -P test_size=0.25 -P random_state=7

  # Run 2: test_size = 0.40, random_state = 7
  python -m mlflow run . --env-manager local -P test_size=0.40 -P random_state=7
  ```
- **Results:**
  - Run 1 (ID: `0c742d943461488b9b9a82f89721a061`): MAE = 4.9518, $R^2 = 0.9917$
  - Run 2 (ID: `97e51f40e4164afbb904147cc4bdbf64`): MAE = 4.8824, $R^2 = 0.9920$

---

## Lab 13 — MLflow Client
- **Objective:** Use `MlflowClient` SDK to query tracking server runs and filter the **Top 3 runs with lowest MAE**.
- **Files:** `labs/lab13/client_demo.py`
- **Command:**
  ```bash
  python client_demo.py
  ```
- **Results (Top 3 Runs in `Lab05_Test_Size_Comparison`):**
  1. Rank 1: Run `6124c570b3a648ed8110d422dd2de1c5` (`test_size_30pct`) — MAE: **8.7518**, $R^2$: 0.9853
  2. Rank 2: Run `8d03551eea124685bd01a8f75b69c666` (`test_size_40pct`) — MAE: **8.9206**, $R^2$: 0.9854
  3. Rank 3: Run `c8057194e7654a91a4c8d84183d61f02` (`test_size_20pct`) — MAE: **9.0197**, $R^2$: 0.9834

---

## Lab 14 — MLflow CLI
- **Objective:** Demonstrate MLflow CLI version interrogation, command discovery, and experiment search.
- **Files:** `labs/lab14/cli_outputs.txt`
- **Commands & Outputs:**
  - `python -m mlflow --version` $\rightarrow$ `mlflow, version 3.18.0`
  - `python -m mlflow --help` $\rightarrow$ Lists all 20+ top-level commands (`server`, `models`, `run`, `experiments`, etc.)
  - `python -m mlflow experiments search` $\rightarrow$ Lists all 15 active experiments on the tracking server.

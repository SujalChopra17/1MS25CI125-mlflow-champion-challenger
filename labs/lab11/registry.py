"""
Lab 11 — Exercise 1: MLflow Model Registry (Champion vs Challenger)
Trains two regression models with different regularizations, logs them, registers them
under 'house-price-regressor', and assigns 'champion' and 'challenger' aliases based on test MAE.
Student: Sujal Chopra (1MS25CI125)
"""

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    TRACKING_URI = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(TRACKING_URI)
    EXP_NAME = "Lab11_Model_Registry"
    MODEL_NAME = "house-price-regressor"
    mlflow.set_experiment(EXP_NAME)
    client = MlflowClient()
    
    np.random.seed(42)
    X = pd.DataFrame({"sqft": np.linspace(500, 3000, 80)})
    y = 50.0 + 0.15 * X["sqft"] + np.random.normal(0, 20, size=80)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # -------------------------------------------------------------
    # Candidate 1: Standard Linear Regression (Candidate A)
    # -------------------------------------------------------------
    with mlflow.start_run(run_name="candidate_linear_reg") as run_a:
        m_a = LinearRegression()
        m_a.fit(X_train, y_train)
        preds_a = m_a.predict(X_test)
        mae_a = mean_absolute_error(y_test, preds_a)
        r2_a = r2_score(y_test, preds_a)
        
        mlflow.log_params({"model_type": "LinearRegression", "student_usn": "1MS25CI125"})
        mlflow.log_metrics({"mae": float(mae_a), "r2": float(r2_a)})
        mlflow.sklearn.log_model(sk_model=m_a, artifact_path="model", input_example=X_train.head(2))
        run_id_a = run_a.info.run_id

    # -------------------------------------------------------------
    # Candidate 2: Ridge Regularized Regression (Candidate B)
    # -------------------------------------------------------------
    with mlflow.start_run(run_name="candidate_ridge_reg") as run_b:
        m_b = Ridge(alpha=100.0)
        m_b.fit(X_train, y_train)
        preds_b = m_b.predict(X_test)
        mae_b = mean_absolute_error(y_test, preds_b)
        r2_b = r2_score(y_test, preds_b)
        
        mlflow.log_params({"model_type": "Ridge", "alpha": 100.0, "student_usn": "1MS25CI125"})
        mlflow.log_metrics({"mae": float(mae_b), "r2": float(r2_b)})
        mlflow.sklearn.log_model(sk_model=m_b, artifact_path="model", input_example=X_train.head(2))
        run_id_b = run_b.info.run_id

    # -------------------------------------------------------------
    # Register both models in the MLflow Model Registry
    # -------------------------------------------------------------
    mv_a = mlflow.register_model(f"runs:/{run_id_a}/model", MODEL_NAME)
    mv_b = mlflow.register_model(f"runs:/{run_id_b}/model", MODEL_NAME)
    
    client.set_model_version_tag(MODEL_NAME, mv_a.version, "model_type", "LinearRegression")
    client.set_model_version_tag(MODEL_NAME, mv_a.version, "student_usn", "1MS25CI125")
    client.set_model_version_tag(MODEL_NAME, mv_b.version, "model_type", "Ridge_alpha_100")
    client.set_model_version_tag(MODEL_NAME, mv_b.version, "student_usn", "1MS25CI125")

    # Decision rule: Lower MAE is better
    if mae_a <= mae_b:
        champ_ver, chall_ver = mv_a.version, mv_b.version
        champ_name, chall_name = "LinearRegression (Candidate A)", "Ridge (Candidate B)"
    else:
        champ_ver, chall_ver = mv_b.version, mv_a.version
        champ_name, chall_name = "Ridge (Candidate B)", "LinearRegression (Candidate A)"
        
    client.set_registered_model_alias(MODEL_NAME, "champion", champ_ver)
    client.set_registered_model_alias(MODEL_NAME, "challenger", chall_ver)

    print("=" * 70)
    print("Lab 11 — Model Registry Aliases Configured")
    print("=" * 70)
    print(f"Registered Model Name: '{MODEL_NAME}'")
    print(f"Candidate A (Linear Regression) : MAE = {mae_a:.4f} -> Version {mv_a.version}")
    print(f"Candidate B (Ridge Regression)  : MAE = {mae_b:.4f} -> Version {mv_b.version}")
    print("-" * 70)
    print(f"[*] Alias '@champion'   assigned to Version {champ_ver} ({champ_name})")
    print(f"[*] Alias '@challenger' assigned to Version {chall_ver} ({chall_name})")
    print("=" * 70)

if __name__ == "__main__":
    main()

"""
Lab 12 — MLflow Projects: Parametrized Training Entrypoint
Accepts CLI parameters via MLproject and tracks execution.
Student: Sujal Chopra (1MS25CI125)
"""

import argparse
import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def parse_args():
    parser = argparse.ArgumentParser(description="MLflow Project Parametrized Trainer")
    parser.add_argument("--test_size", type=float, default=0.25, help="Test split ratio")
    parser.add_argument("--random_state", type=int, default=7, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    
    np.random.seed(args.random_state)
    X = np.linspace(10, 100, 80).reshape(-1, 1)
    y = 2.8 * X.squeeze() + 8.5 + np.random.normal(0, 6, size=80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    
    # In MLflow Projects, the run is already initialized by the runner
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # Note: MLflow Project runner automatically logs parameters passed via -P
    # Log metrics and tags
    mlflow.log_metrics({
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "mae": float(mae),
        "r2_score": float(r2),
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_)
    })
    
    mlflow.set_tags({
        "student_name": "Sujal Chopra",
        "student_usn": "1MS25CI125"
    })
    
    print("=" * 60)
    print("Lab 12 — MLproject Run Execution Complete")
    print("=" * 60)
    print(f"Parameters : test_size = {args.test_size}, random_state = {args.random_state}")
    print(f"Samples    : Train = {len(X_train)}, Test = {len(X_test)}")
    print(f"Results    : MAE = {mae:.4f}, R² = {r2:.4f}")
    print(f"Equation   : y = {model.coef_[0]:.4f} * X + {model.intercept_:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()

"""
Lab 05 — Multiple Experiments & Runs: Test Size Comparison & Random State Sensitivity
Demonstrating how MLflow tracks multiple runs within an experiment to study hyperparameters and sample sizes.
Student: Sujal Chopra (1MS25CI125)
"""

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def run_experiment(test_size, random_state, run_name):
    # Dataset: Housing price indicator (synthetic feature)
    np.random.seed(random_state)
    X = np.linspace(10, 100, 100).reshape(-1, 1)
    y = 50.0 + 3.5 * X.squeeze() + np.random.normal(0, 15, size=100)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    with mlflow.start_run(run_name=run_name):
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        mlflow.log_params({
            "test_size": test_size,
            "random_state": random_state,
            "model_type": "LinearRegression",
            "student_usn": "1MS25CI125"
        })
        
        mlflow.log_metrics({
            "n_training_samples": len(X_train),
            "n_testing_samples": len(X_test),
            "mae": float(mae),
            "r2_score": float(r2),
            "slope": float(model.coef_[0]),
            "intercept": float(model.intercept_)
        })
        
        mlflow.set_tag("student_name", "Sujal Chopra")
        
        return {
            "run_name": run_name,
            "test_size": test_size,
            "random_state": random_state,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "mae": round(mae, 4),
            "r2": round(r2, 4)
        }

def main():
    TRACKING_URI = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(TRACKING_URI)
    
    # Part 1: Comparing test sizes 0.2, 0.3, 0.4 with fixed random_state = 42
    mlflow.set_experiment("Lab05_Test_Size_Comparison")
    print("=" * 70)
    print("Lab 05 — Part 1: Comparing Test Sizes (0.2, 0.3, 0.4) with Fixed Seed (42)")
    print("=" * 70)
    
    res_part1 = []
    for ts in [0.2, 0.3, 0.4]:
        r = run_experiment(test_size=ts, random_state=42, run_name=f"test_size_{int(ts*100)}pct")
        res_part1.append(r)
    
    df1 = pd.DataFrame(res_part1)
    print(df1.to_string(index=False))
    
    # Part 2: Sensitivity across different random states with fixed test_size = 0.25
    mlflow.set_experiment("Lab05_Random_State_Sensitivity")
    print("\n" + "=" * 70)
    print("Lab 05 — Part 2: Sensitivity Across Random States (7, 42, 100)")
    print("=" * 70)
    
    res_part2 = []
    for seed in [7, 42, 100]:
        r = run_experiment(test_size=0.25, random_state=seed, run_name=f"seed_{seed}")
        res_part2.append(r)
        
    df2 = pd.DataFrame(res_part2)
    print(df2.to_string(index=False))
    print("=" * 70)

if __name__ == "__main__":
    main()

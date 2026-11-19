"""
Lab 01 — Exercise 1: Baseline Linear Regression Model
Demonstrates baseline training and evaluation with different test sizes, recording MAE and sample counts.
Student: Sujal Chopra (1MS25CI125)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def run_baseline(test_size=0.2, random_state=42):
    # Generate synthetic regression data
    np.random.seed(random_state)
    X = np.arange(1, 51).reshape(-1, 1)
    y = 2.5 * X.squeeze() + np.random.normal(0, 5, size=len(X))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    
    return {
        "test_size": test_size,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "mae": round(mae, 4),
        "coef": round(model.coef_[0], 4),
        "intercept": round(model.intercept_, 4)
    }

if __name__ == "__main__":
    print("=" * 60)
    print("Lab 01 - Baseline Model Evaluation across Different Test Sizes")
    print("=" * 60)
    
    test_sizes = [0.2, 0.25, 0.3, 0.4]
    results = [run_baseline(ts) for ts in test_sizes]
    
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    
    # Save observations to CSV
    df_res.to_csv("baseline_observations.csv", index=False)
    print("\nSaved observations to baseline_observations.csv")

"""
Lab 01 — Exercise 2: Delivery Time Predictor
Predicting actual delivery time from distance using linear regression.
Student: Sujal Chopra (1MS25CI125)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def main():
    # Exact dataset from Lab Book:
    # Distance X (km): 1, 2, 3, 4, 5
    # Actual time Y (min): 16, 19, 27, 29, 34
    data = {
        "order": [1, 2, 3, 4, 5],
        "distance_x_km": [1, 2, 3, 4, 5],
        "actual_time_y_min": [16, 19, 27, 29, 34]
    }
    df = pd.DataFrame(data)
    
    X = df[["distance_x_km"]]
    y = df["actual_time_y_min"]
    
    # Train/test split (e.g. 1 test observation for 5 samples, or leave-one-out)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    
    print("=" * 60)
    print("Lab 01 - Delivery Time Predictor Results")
    print("=" * 60)
    print(f"Number of training observations : {len(X_train)}")
    print(f"Number of test observations     : {len(X_test)}")
    print(f"Model Coefficient (slope)       : {model.coef_[0]:.4f} min/km")
    print(f"Model Intercept                 : {model.intercept_:.4f} min")
    print(f"Test Actual Time                : {y_test.values.tolist()} min")
    print(f"Test Predicted Time             : {[round(p, 2) for p in preds]} min")
    print(f"Mean Absolute Error (MAE)       : {mae:.4f} min")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": preds,
        "error": np.abs(y_test.values - preds),
        "mae": mae,
        "n_train": len(X_train),
        "n_test": len(X_test)
    })
    results_df.to_csv("delivery_time_results.csv", index=False)
    print("\nSaved delivery time prediction results to delivery_time_results.csv")

if __name__ == "__main__":
    main()

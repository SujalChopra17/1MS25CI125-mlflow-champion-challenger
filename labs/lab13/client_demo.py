"""
Lab 13 — MLflow Client API: Querying and Filtering Experiment Runs
Uses MlflowClient to query runs across experiments and filter the Top 3 best performing runs with lowest MAE.
Student: Sujal Chopra (1MS25CI125)
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

def main():
    TRACKING_URI = "http://127.0.0.1:5000"
    client = MlflowClient(tracking_uri=TRACKING_URI)
    
    print("=" * 75)
    print("Lab 13 — MLflow Client API Demonstration")
    print(f"Tracking Server: {TRACKING_URI}")
    print("=" * 75)
    
    # 1. List all active experiments
    experiments = client.search_experiments()
    print(f"\n[+] Total Experiments Found on Server: {len(experiments)}")
    for exp in experiments[:5]:
        print(f"    - ID: {exp.experiment_id:<3} | Name: {exp.name:<35} | Stage: {exp.lifecycle_stage}")
        
    # 2. Select experiment with multiple regression runs (e.g., Lab05_Test_Size_Comparison or search across all)
    target_exp_name = "Lab05_Test_Size_Comparison"
    exp = client.get_experiment_by_name(target_exp_name)
    if not exp:
        exp = experiments[0]
        
    print(f"\n[+] Inspecting Runs in Experiment '{exp.name}' (ID: {exp.experiment_id}):")
    all_runs = client.search_runs(experiment_ids=[exp.experiment_id])
    print(f"    Total runs retrieved: {len(all_runs)}")
    
    run_records = []
    for r in all_runs:
        run_records.append({
            "run_id": r.info.run_id[:8] + "...",
            "run_name": r.info.run_name,
            "status": r.info.status,
            "mae": r.data.metrics.get("mae", None),
            "r2": r.data.metrics.get("r2_score", None)
        })
    print(pd.DataFrame(run_records).to_string(index=False))
    
    # -------------------------------------------------------------
    # 3. Required Modification: Display ONLY the Top 3 Runs with Lowest MAE
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("TOP 3 RUNS WITH LOWEST MAE (ORDERED BY metrics.mae ASC)")
    print("=" * 75)
    
    # Search with order_by and max_results=3
    top3_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="metrics.mae > 0",
        order_by=["metrics.mae ASC"],
        max_results=3
    )
    
    top3_data = []
    for rank, r in enumerate(top3_runs, 1):
        top3_data.append({
            "Rank": rank,
            "Run ID": r.info.run_id,
            "Run Name": r.info.run_name,
            "MAE": round(r.data.metrics.get("mae", 0.0), 4),
            "R² Score": round(r.data.metrics.get("r2_score", 0.0), 4),
            "Test Size": r.data.params.get("test_size", "N/A"),
            "Random State": r.data.params.get("random_state", "N/A")
        })
        
    df_top3 = pd.DataFrame(top3_data)
    print(df_top3.to_string(index=False))
    print("=" * 75)

if __name__ == "__main__":
    main()

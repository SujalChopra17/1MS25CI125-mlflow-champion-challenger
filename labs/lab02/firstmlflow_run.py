"""
Lab 02 — Exercise 1: First MLflow Run
Demonstrating basic MLflow run initialization, logging params, metrics, and artifacts,
and inspecting the internal file structure of the `mlruns/` tracking directory.
Student: Sujal Chopra (1MS25CI125)
"""

import os
import mlflow

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

def main():
    # Use local file-based tracking store (mlruns)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Lab02_First_MLflow_Run")
    
    with mlflow.start_run(run_name="first_run_demo") as run:
        run_id = run.info.run_id
        print("=" * 60)
        print("Lab 02 - Executing First MLflow Run")
        print("=" * 60)
        print(f"Run ID: {run_id}")
        
        # 1. Log Parameters
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("student_usn", "1MS25CI125")
        
        # 2. Log Metrics
        mlflow.log_metric("loss", 0.1452)
        mlflow.log_metric("accuracy", 0.9450)
        
        # 3. Log Tags
        mlflow.set_tag("environment", "local_development")
        mlflow.set_tag("student_name", "Sujal Chopra")
        
        # 4. Create and Log an Artifact
        os.makedirs("sample_outputs", exist_ok=True)
        artifact_path = "sample_outputs/run_summary.txt"
        with open(artifact_path, "w") as f:
            f.write(f"Run completed successfully for USN 1MS25CI125\nRun ID: {run_id}\n")
        
        mlflow.log_artifact(artifact_path)
        print("Logged parameters, metrics, tags, and artifact successfully.")
        
    print("\nInspecting mlruns directory structure:")
    for root, dirs, files in os.walk("mlruns"):
        level = root.replace("mlruns", "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 4 * (level + 1)
        for f in files:
            print(f"{sub_indent}{f}")

if __name__ == "__main__":
    main()

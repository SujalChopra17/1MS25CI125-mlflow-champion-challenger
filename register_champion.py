"""
register_champion.py
Registers evaluated models into the MLflow Model Registry and sets Champion and Challenger aliases.
Student: Sujal Chopra (1MS25CI125)
Department: CSE (AIML), Ramaiah Institute of Technology
"""

import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXP = "hf-sentiment-champion-challenger"
NAME = "sentiment-model"
STUDENT_USN = "1MS25CI125"
STUDENT_NAME = "Sujal Chopra"

client = MlflowClient()
exp = client.get_experiment_by_name(EXP)
if not exp:
    raise ValueError(f"Experiment '{EXP}' not found. Please run evaluate.py first.")

runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"])
get = lambda p: next((r for r in runs if r.info.run_name.startswith(p)), None)

compare_run = get("compare_")
if not compare_run:
    raise ValueError("Comparison run not found.")

winner = compare_run.data.tags.get("winner", "A")
print(f"[*] Found comparison run: {compare_run.info.run_id}")
print(f"[*] Evaluated Winner from decision rule: Model {winner}")

vers = {}
for k in ["A", "B"]:
    r = get(f"{k}_")
    if not r:
        raise ValueError(f"Run for model {k} not found.")
    
    print(f"[*] Registering Model {k} (Run ID: {r.info.run_id}) under '{NAME}'...")
    mv = mlflow.register_model(f"runs:/{r.info.run_id}/model", NAME)
    
    hf_model_id = r.data.params.get("hf_model_id", "unknown")
    role = "champion" if k == winner else "challenger"
    
    # Tag model version with metadata as required by the rubric
    client.set_model_version_tag(NAME, mv.version, "hf_model_id", hf_model_id)
    client.set_model_version_tag(NAME, mv.version, "student_usn", STUDENT_USN)
    client.set_model_version_tag(NAME, mv.version, "student_name", STUDENT_NAME)
    client.set_model_version_tag(NAME, mv.version, "role", role)
    
    vers[k] = mv.version
    print(f"    -> Registered version {mv.version} for Model {k} ({hf_model_id})")

other = "B" if winner == "A" else "A"
client.set_registered_model_alias(NAME, "champion", vers[winner])
client.set_registered_model_alias(NAME, "challenger", vers[other])

print("\n" + "="*50)
print("MLflow Model Registry Aliases Configured Successfully:")
print(f"  champion   -> version {vers[winner]} (Model {winner})")
print(f"  challenger -> version {vers[other]} (Model {other})")
print(f"  Student USN: {STUDENT_USN} ({STUDENT_NAME})")
print("="*50)


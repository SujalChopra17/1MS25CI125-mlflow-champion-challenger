import os, time, json, mlflow, torch, transformers, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

USN = "1MS25CI125"          # <-- CHANGE THIS
TASK, SEED, N, BATCH = "sentiment", 42, 100, 8   # set N=10 for a quick test
EXP = "hf-sentiment-champion-challenger"
PRIMARY, MARGIN, LAT_X = "f1_macro", 0.01, 5
MODELS = {
    "A": ("distilbert-base-uncased-finetuned-sst-2-english", "distilbert-sst2", "champion"),
    "B": ("siebert/sentiment-roberta-large-english", "roberta-large", "challenger candidate"),
}
os.makedirs("data", exist_ok=True); os.makedirs("outputs", exist_ok=True)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXP)

path = "data/eval_inputs.csv"
if not os.path.exists(path):
    ds = load_dataset("nyu-mll/glue", "sst2", split="validation").shuffle(seed=SEED).select(range(N))
    pd.DataFrame({"text": ds["sentence"], "label": ds["label"]}).to_csv(path, index=False)
df = pd.read_csv(path)

class HFSentiment(mlflow.pyfunc.PythonModel):
    def __init__(self, hf_model_id):
        self.hf_model_id = hf_model_id
    def load_context(self, context):
        from transformers import pipeline
        self.pipe = pipeline("sentiment-analysis", model=self.hf_model_id)
    def predict(self, context, model_input, params=None):
        texts = model_input["text"].astype(str).tolist()
        out = self.pipe(texts, batch_size=8, truncation=True)
        preds = [1 if o["label"].upper().startswith("POS") else 0 for o in out]
        return pd.DataFrame({"text": texts, "prediction": preds,
                             "sentiment": ["positive" if p else "negative" for p in preds],
                             "hf_model_id": self.hf_model_id})

def evaluate(key):
    hf_id, short, role = MODELS[key]
    pipe = pipeline("sentiment-analysis", model=hf_id)
    t0 = time.time()
    out = pipe(df.text.tolist(), batch_size=BATCH, truncation=True)
    total = time.time() - t0
    preds = [1 if o["label"].upper().startswith("POS") else 0 for o in out]
    m = {"accuracy": accuracy_score(df.label, preds),
         "f1_macro": f1_score(df.label, preds, average="macro"),
         "avg_latency_ms": total / len(df) * 1000,
         "total_inference_s": total}
    res = df.copy(); res["prediction"] = preds; res["correct"] = res.label == res.prediction
    pf, mf = f"outputs/predictions_{short}.csv", f"outputs/metrics_summary_{short}.json"
    res.to_csv(pf, index=False); json.dump(m, open(mf, "w"), indent=2)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(df.label, preds)).plot(ax=ax)
    ax.set_title(f"Confusion matrix - {short}")
    fig.savefig(f"outputs/confusion_{short}.png")
    with mlflow.start_run(run_name=f"{key}_{TASK}_{short}"):
        mlflow.log_params({"hf_model_id": hf_id, "task": TASK, "role": role,
                           "batch_size": BATCH, "eval_samples": len(df),
                           "seed": SEED, "device": "cuda" if torch.cuda.is_available() else "cpu"})
        mlflow.log_metrics(m)
        mlflow.set_tags({"student_usn": USN, "dataset_name": "glue/sst2-validation",
                         "library_versions": f"transformers={transformers.__version__}, torch={torch.__version__}, mlflow={mlflow.__version__}"})
        for f in (pf, mf, path): mlflow.log_artifact(f)
        mlflow.log_figure(fig, "confusion_matrix.png")
        mlflow.pyfunc.log_model(artifact_path="model", python_model=HFSentiment(hf_id),
                                input_example=df[["text"]].head(2),
                                pip_requirements=["mlflow", "transformers", "torch"])
    plt.close(fig)
    return m, res

mA, rA = evaluate("A")
mB, rB = evaluate("B")

delta = mB[PRIMARY] - mA[PRIMARY]
if delta >= MARGIN:
    winner = "B" if mB["avg_latency_ms"] <= LAT_X * mA["avg_latency_ms"] else "A"
elif delta > -MARGIN:
    winner = "A" if mA["avg_latency_ms"] <= mB["avg_latency_ms"] else "B"
else:
    winner = "A"

names = list(mA.keys())
fig, ax = plt.subplots(figsize=(8, 4))
x = range(len(names))
ax.bar([i - 0.2 for i in x], [mA[n] for n in names], 0.4, label="A distilbert")
ax.bar([i + 0.2 for i in x], [mB[n] for n in names], 0.4, label="B roberta-large")
ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=20); ax.set_yscale("symlog"); ax.legend()
ax.set_title("Champion vs Challenger")
fig.savefig("outputs/compare_bar.png")
dis = pd.DataFrame({"text": df.text, "label": df.label, "pred_A": rA.prediction, "pred_B": rB.prediction})
dis = dis[dis.pred_A != dis.pred_B].head(5); dis["comment"] = ""
dis.to_csv("outputs/disagreements.csv", index=False)
json.dump({"A": mA, "B": mB, "delta_primary_metric": delta, "winner": winner},
          open("outputs/metrics_summary.json", "w"), indent=2)

with mlflow.start_run(run_name=f"compare_{TASK}"):
    mlflow.log_params({"primary_metric": PRIMARY, "margin": MARGIN, "latency_guardrail_x": LAT_X,
                       "model_A": MODELS["A"][0], "model_B": MODELS["B"][0]})
    mlflow.log_metric("delta_primary_metric", delta)
    mlflow.set_tags({"winner": winner, "student_usn": USN})
    mlflow.log_figure(fig, "compare_bar.png")
    for f in ["outputs/metrics_summary.json", "outputs/disagreements.csv", "README.md", "requirements.txt"]:
        if os.path.exists(f): mlflow.log_artifact(f)

print(pd.DataFrame({"A": mA, "B": mB}).assign(diff=lambda d: d.B - d.A))
print("delta:", delta, "| winner:", winner)

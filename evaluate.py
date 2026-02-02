import json
from pathlib import Path

metrics = json.load(open("ml/artifacts/metrics.json"))

best_auc = max(v["Mean AUC"] for v in metrics.values())

print("Best AUC:", best_auc)

if best_auc < 0.70:
    raise ValueError("Model performance below threshold")

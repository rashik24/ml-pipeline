import json

metrics = json.load(open("artifacts/metrics.json"))

# Pandas saved metrics column-wise:
# {"Mean AUC": {"RF":0.8, "SVM":0.7}, ...}

mean_auc_dict = metrics["Mean AUC"]
best_auc = max(mean_auc_dict.values())

print("Best AUC:", best_auc)

if best_auc < 0.70:
    raise ValueError("Model performance below threshold")

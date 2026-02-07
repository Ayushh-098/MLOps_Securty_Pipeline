import pandas as pd
import pickle
import yaml
import mlflow

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

data_path = params["data"]["clean"]
model_path = "models/model.pkl"

# Load data
df = pd.read_csv(data_path)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

preds = model.predict(X)

acc = accuracy_score(y, preds)
prec = precision_score(y, preds, zero_division=0)
rec = recall_score(y, preds, zero_division=0)
f1 = f1_score(y, preds, zero_division=0)

mlflow.set_experiment("Security Threat Detection")

with mlflow.start_run():
    mlflow.log_metric("eval_accuracy", acc)
    mlflow.log_metric("eval_precision", prec)
    mlflow.log_metric("eval_recall", rec)
    mlflow.log_metric("eval_f1", f1)

print("Evaluation complete")

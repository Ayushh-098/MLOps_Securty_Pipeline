import pandas as pd
import yaml
import os
import mlflow
import mlflow.sklearn
import pickle


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

data_path = params["data"]["clean"]
test_size = params["model"]["test_size"]
random_state = params["model"]["random_state"]

model_out = "models/model.pkl"
os.makedirs("models", exist_ok=True)

# Load data
df = pd.read_csv(data_path)

# LAST COLUMN = TARGET (standard cybersecurity datasets follow this)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

mlflow.set_experiment("Security Threat Detection")

with mlflow.start_run():
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Save model for DVC
    with open(model_out, "wb") as f:
        pickle.dump(model, f)

    # Log model to MLflow
    mlflow.sklearn.log_model(model, artifact_path="model")

    print("Training complete")


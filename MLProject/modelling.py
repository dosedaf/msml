import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train():
    data = pd.read_csv("heart_preprocessed.csv")
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    with mlflow.start_run():
        mlflow.log_metric("accuracy", acc)

        joblib.dump(model, "model.joblib")
        mlflow.log_artifact("model.joblib")

if __name__ == "__main__":
    train()

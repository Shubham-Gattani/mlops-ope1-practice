import pandas as pd
from google.cloud import bigquery
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import joblib
import os


def load_data_from_bigquery():
    client = bigquery.Client()
    QUERY = """
        SELECT rolling_avg_10, volume_sum_10, target
        FROM `mlops-iris-week1-graded.feast.stock_features_all`
    """

    df = client.query(QUERY).to_dataframe()
    df = df.dropna()

    return df


def train_and_log_model(X_train, y_train, X_test, y_test, params):
    model = LogisticRegression(**params, max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    signature = infer_signature(X_train[:5], model.predict(X_train[:5]))

    run_name = f"C={params['C']}_solver={params['solver']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(
            sk_model = model, 
            signature = signature,
            name = "model",
            registered_model_name="mlops-ope-practice"
            )

    return acc, model


def main():
    print("✅ Loading v1 training data...")
    df = load_data_from_bigquery()
    print("✅ Combined dataset shape:", df.shape)

    X = df[["rolling_avg_10", "volume_sum_10"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    mlflow.set_experiment("stock_prediction_v1")

    candidate_params = [
        {"C": 0.1, "solver": "liblinear"},
        {"C": 1.0, "solver": "lbfgs"},
    ]

    best_acc = -1
    best_model = None
    best_params = None

    for params in candidate_params:
        print(f"\n✅ Training model with params: {params}")
        acc, model = train_and_log_model(X_train, y_train, X_test, y_test, params)
        print(f"Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_params = params

    print("\n🎉 BEST model params:", best_params)
    print("🎉 BEST accuracy:", best_acc)

    os.makedirs("models", exist_ok=True)
    best_model_path = "models/best_logistic_model_v1.joblib"
    joblib.dump(best_model, best_model_path)

    print(f"\n✅ Saved best v1 model to {best_model_path}")


if __name__ == "__main__":
    main()

from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def train_direct_bq():
    print("\n=== ✅ Loading training data directly from BigQuery ===")

    client = bigquery.Client()

    query = """
        SELECT rolling_avg_10, volume_sum_10, target
        FROM `mlops-iris-week1-graded.feast.stock_features_all`
    """

    df = client.query(query).to_dataframe()

    print("✅ Loaded:", df.shape)

    X = df[["rolling_avg_10", "volume_sum_10"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # ✅ Extremely fast model
    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("\n✅ Accuracy:", acc)
    print("\n✅ Classification Report:\n", classification_report(y_test, preds))
    print("\n🎉 Training complete (FAST MODEL).\n")


if __name__ == "__main__":
    train_direct_bq()

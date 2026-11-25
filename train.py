"""Train a simple classifier on heart.csv and save it as model.joblib.

Run:  python train.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "heart.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
TARGET = "target"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # This dataset contains many duplicated rows. If we kept them, the same
    # patient could end up in both the train and test sets and the accuracy
    # would look better than it really is.
    rows_before = len(df)
    df = df.drop_duplicates()
    print(f"Rows: {rows_before} -> {len(df)} after dropping duplicates")

    X = df.drop(columns=TARGET)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling + logistic regression in one pipeline: the API can pass raw
    # feature values and the pipeline applies the same scaling as in training.
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH.name}")


if __name__ == "__main__":
    main()

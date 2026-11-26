"""Train several models on heart.csv, track them with MLflow and register the best one.

Run:  python train.py
Then: mlflow ui --backend-store-uri sqlite:///mlflow.db   (open http://127.0.0.1:5000)

What happens:
1. Every candidate model is trained in its own MLflow run. Each run logs
   its parameters, its metrics and the model itself (artifacts).
2. The best run (highest cross-validated F1 on the training data) is
   registered in the MLflow Model Registry as "heart-disease-model".
   Every registration creates a new version (1, 2, 3, ...) and the
   alias "champion" is moved to the newest best version.
3. The best model is also saved to model.joblib (+ model_info.json),
   which the FastAPI service loads, exactly like in Practice 6.
"""
import json
import os
import tempfile
import warnings
from pathlib import Path

# Hide MLflow's hint about its AI tracing features (not used here).
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "heart.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
MODEL_INFO_PATH = BASE_DIR / "model_info.json"
TARGET = "target"

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# MLflow settings. A SQLite database is used as the backend store because
# the Model Registry needs a database (plain files are not enough).
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}")
EXPERIMENT_NAME = "heart-disease-prediction"
REGISTERED_MODEL_NAME = "heart-disease-model"
CHAMPION_ALIAS = "champion"

# heart.csv has integer columns and no missing values, so MLflow's warning
# about integer columns that could contain missing values does not apply.
warnings.filterwarnings("ignore", message=".*Inferred schema contains integer column.*")

# The candidate models we compare. Each one becomes one MLflow run.
CANDIDATES = [
    {"model_type": "logistic_regression", "C": 0.1},
    {"model_type": "logistic_regression", "C": 1.0},
    {"model_type": "logistic_regression", "C": 10.0},
    {"model_type": "random_forest", "n_estimators": 200, "max_depth": 5},
]


def build_model(params: dict):
    """Create a scikit-learn pipeline from a candidate's parameters."""
    if params["model_type"] == "logistic_regression":
        classifier = LogisticRegression(C=params["C"], max_iter=1000, random_state=RANDOM_STATE)
    elif params["model_type"] == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model type: {params['model_type']}")
    # Scaling + classifier in one pipeline: the API can pass raw feature
    # values and the pipeline applies the same scaling as in training.
    return make_pipeline(StandardScaler(), classifier)


def run_name(params: dict) -> str:
    """Readable run name, e.g. 'logistic_regression_C=0.1'."""
    extra = "_".join(f"{k}={v}" for k, v in params.items() if k != "model_type")
    return f"{params['model_type']}_{extra}"


def load_data():
    df = pd.read_csv(DATA_PATH)

    # This dataset contains many duplicated rows. If we kept them, the same
    # patient could end up in both the train and test sets and the accuracy
    # would look better than it really is.
    rows_before = len(df)
    df = df.drop_duplicates()
    print(f"Rows: {rows_before} -> {len(df)} after dropping duplicates")

    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def train_one(params: dict, X_train, X_test, y_train, y_test) -> dict:
    """Train one candidate inside its own MLflow run and log everything."""
    with mlflow.start_run(run_name=run_name(params)) as run:
        model = build_model(params)

        # Cross-validation on the TRAINING data is used to choose the best
        # model, so the test set stays a fair final check.
        cv_f1 = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="f1").mean()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "cv_f1": cv_f1,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        # 1) Parameters (hyperparameters + how the data was split)
        mlflow.log_params(params)
        mlflow.log_params(
            {
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "cv_folds": CV_FOLDS,
                "n_train_rows": len(X_train),
                "n_test_rows": len(X_test),
            }
        )
        # 2) Metrics
        mlflow.log_metrics(metrics)

        # 3) Artifacts: the model in MLflow format (with its input signature),
        #    the plain model.joblib file and a text classification report.
        signature = infer_signature(X_train, model.predict(X_train))
        # MLflow saves sklearn models with skops, which only loads types it
        # trusts. Tree models need sklearn's Tree type; we trained the model
        # ourselves, so it is safe to trust it.
        trusted = ["sklearn.tree._tree.Tree"] if params["model_type"] == "random_forest" else None
        logged_model = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            skops_trusted_types=trusted,
        )
        with tempfile.TemporaryDirectory() as tmp:
            joblib_path = Path(tmp) / "model.joblib"
            joblib.dump(model, joblib_path)
            mlflow.log_artifact(str(joblib_path))
        mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")

        print(
            f"  {run_name(params):<40} cv_f1={metrics['cv_f1']:.3f}  "
            f"test_acc={metrics['accuracy']:.3f}  test_f1={metrics['f1']:.3f}"
        )
        return {
            "run_id": run.info.run_id,
            "model_uri": logged_model.model_uri,
            "params": params,
            "metrics": metrics,
            "model": model,
        }


def register_best(best: dict) -> str:
    """Register the best run's model and point the 'champion' alias at it."""
    version = mlflow.register_model(best["model_uri"], REGISTERED_MODEL_NAME).version

    client = MlflowClient()
    client.update_registered_model(
        REGISTERED_MODEL_NAME,
        description="Classifier for the Kaggle heart disease dataset (heart.csv). "
        "Educational project only, not for medical use.",
    )
    client.update_model_version(
        REGISTERED_MODEL_NAME,
        version,
        description=f"Best of {len(CANDIDATES)} candidates by cross-validated F1: {run_name(best['params'])}",
    )
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "model_type", best["params"]["model_type"])
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version)
    return version


def main() -> None:
    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)  # creates the experiment the first time
    print(f"MLflow experiment '{EXPERIMENT_NAME}' at {TRACKING_URI}")

    results = [train_one(p, X_train, X_test, y_train, y_test) for p in CANDIDATES]
    best = max(results, key=lambda r: r["metrics"]["cv_f1"])
    print(f"Best model: {run_name(best['params'])} (cv_f1={best['metrics']['cv_f1']:.3f})")

    version = register_best(best)
    print(f"Registered '{REGISTERED_MODEL_NAME}' version {version} (alias '{CHAMPION_ALIAS}')")

    # Save the best model for the FastAPI service (same as in Practice 6),
    # plus a small JSON file that says which registry version it is.
    joblib.dump(best["model"], MODEL_PATH)
    model_info = {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "version": int(version),
        "alias": CHAMPION_ALIAS,
        "run_id": best["run_id"],
        "experiment": EXPERIMENT_NAME,
        "params": best["params"],
        "metrics": {k: round(v, 4) for k, v in best["metrics"].items()},
    }
    MODEL_INFO_PATH.write_text(json.dumps(model_info, indent=2))
    print(f"Saved {MODEL_PATH.name} and {MODEL_INFO_PATH.name}")


if __name__ == "__main__":
    main()

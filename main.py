"""FastAPI service that serves predictions from the trained model.

Run:  uvicorn main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
MODEL_INFO_PATH = BASE_DIR / "model_info.json"

if not MODEL_PATH.exists():
    raise RuntimeError("model.joblib not found. Run `python train.py` first.")

# The model is an artifact created by train.py. Load it once at startup
# instead of reading the file on every request.
model = joblib.load(MODEL_PATH)

# train.py also writes model_info.json: which MLflow registry version the
# model.joblib file is, and its metrics. It is optional for the API to work.
model_info = json.loads(MODEL_INFO_PATH.read_text()) if MODEL_INFO_PATH.exists() else None

app = FastAPI(
    title="Heart Disease ML API",
    description="Serves the best model trained on heart.csv (tracked and registered with MLflow). "
    "For learning purposes only, not for medical use.",
    version="2.0.0",
)


class PatientFeatures(BaseModel):
    """The 13 input features, named exactly like the columns of heart.csv."""

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = yes)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (1 = yes)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of the peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia code (0-3)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 52,
                    "sex": 1,
                    "cp": 0,
                    "trestbps": 125,
                    "chol": 212,
                    "fbs": 0,
                    "restecg": 1,
                    "thalach": 168,
                    "exang": 0,
                    "oldpeak": 1.0,
                    "slope": 2,
                    "ca": 2,
                    "thal": 3,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class, as in the 'target' column (0 or 1)")
    probability: float = Field(..., description="Model's probability for the predicted class")


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.get("/model-info")
def get_model_info():
    """Which registered MLflow model version is being served, with its metrics."""
    if model_info is None:
        raise HTTPException(status_code=404, detail="model_info.json not found. Run `python train.py`.")
    return model_info


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures):
    # One-row DataFrame with the same column names/order the model was trained on.
    row = pd.DataFrame([features.model_dump()])[list(model.feature_names_in_)]

    predicted_class = int(model.predict(row)[0])
    class_index = list(model.classes_).index(predicted_class)
    probability = float(model.predict_proba(row)[0][class_index])

    return PredictionResponse(prediction=predicted_class, probability=round(probability, 4))

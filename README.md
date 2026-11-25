# Heart Disease Prediction API (FastAPI + Docker)

A small machine learning project: a classifier is trained on the Kaggle
[Heart Disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
(`heart.csv`), saved to `model.joblib`, and served as a REST API with FastAPI.
The API is packaged in a Docker container.

> Educational project only. Not for medical use.

## Project structure

```
ml-fastapi-docker/
├── main.py            # FastAPI app (GET /, POST /predict)
├── train.py           # trains the model and saves model.joblib
├── model.joblib       # saved model (artifact loaded by the API)
├── heart.csv          # dataset
├── requirements.txt   # Python dependencies
├── Dockerfile         # container instructions
└── README.md
```

## Model

- Dataset: `heart.csv`, 1025 rows, 13 features, binary `target` column.
- The file contains many duplicate rows (302 unique). `train.py` drops them
  before the train/test split so the same patient is not in both sets.
- Model: `StandardScaler` + `LogisticRegression` in one scikit-learn pipeline.
- Test accuracy: about 0.80 (80/20 stratified split, `random_state=42`).

## Run locally

```bash
# 1. (optional) create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. train the model (creates model.joblib)
python train.py

# 4. start the API
uvicorn main:app --reload
```

Open <http://127.0.0.1:8000/docs> for the interactive Swagger UI.

## Run with Docker

```bash
docker build -t heart-api .
docker run -p 8000:8000 heart-api
```

Then open <http://localhost:8000/docs>.

## API endpoints

### `GET /`

```json
{"message": "ML API is running"}
```

### `POST /predict`

Request body (all 13 fields are required):

```json
{
  "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
  "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
  "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
}
```

Response:

```json
{"prediction": 0, "probability": 0.7511}
```

`prediction` is the predicted class, using the same coding as the `target`
column of the dataset. `probability` is the model's probability for that class.
Invalid input (missing fields, values out of range) returns HTTP 422.

Example with curl (macOS / Linux / Git Bash):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,"restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":2,"thal":3}'
```

Example with PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,"restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":2,"thal":3}'
```

## Features

| Field | Meaning |
|---|---|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = yes) |
| restecg | Resting ECG result (0-2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment (0-2) |
| ca | Number of major vessels colored by fluoroscopy (0-4) |
| thal | Thalassemia code (0-3) |

"""Streamlit frontend for the Heart Disease ML API.

Run:  streamlit run frontend/app.py      (the API must be running)
Open: http://localhost:8501

The page does NOT load the model itself. It sends the form data to the
FastAPI service (POST /predict) and shows the answer:
    user -> Streamlit (this page) -> FastAPI -> model
"""
import os

import requests
import streamlit as st

# Where the FastAPI service is. Locally it is 127.0.0.1:8000; in Docker
# Compose the frontend reaches it by its service name (http://api:8000).
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_SECONDS = 10

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")


def api_get(path: str):
    """GET a JSON endpoint of the API. Returns (data, error_message)."""
    try:
        response = requests.get(f"{API_URL}{path}", timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as error:
        return None, str(error)


# ---------- Sidebar: API status and which model is being served ----------
with st.sidebar:
    st.header("Backend")
    st.caption(f"API: {API_URL}")
    _, api_error = api_get("/")
    if api_error:
        st.error("API is not reachable. Start it with `uvicorn main:app`.")
    else:
        st.success("API is running")

    info, _ = api_get("/model-info")
    if info:
        st.header("Served model")
        st.write(f"**{info['registered_model_name']}**, version {info['version']}")
        st.write(f"Alias: `{info['alias']}`  \nType: `{info['params']['model_type']}`")
        metrics = info["metrics"]
        col_a, col_b = st.columns(2)
        col_a.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        col_b.metric("F1", f"{metrics['f1']:.3f}")
        col_a.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        col_b.metric("CV F1", f"{metrics['cv_f1']:.3f}")
        st.caption(f"MLflow run: `{info['run_id'][:8]}`")

# ---------- Main page: patient form ----------
st.title("Heart Disease Prediction")
st.write(
    "Enter the patient's data and press **Predict**. The values are sent to the "
    "FastAPI service, which returns the model's prediction."
)
st.caption("Educational project only. Not for medical use.")

YES_NO = {0: "No", 1: "Yes"}

with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=52)
        sex = st.selectbox("Sex", [1, 0], format_func=lambda v: "Male" if v == 1 else "Female")
        cp = st.selectbox("Chest pain type (cp)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting blood pressure, mm Hg (trestbps)", 50, 250, 125)
        chol = st.number_input("Serum cholesterol, mg/dl (chol)", 50, 700, 212)
    with col2:
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", [0, 1], format_func=YES_NO.get)
        restecg = st.selectbox("Resting ECG result (restecg)", [0, 1, 2], index=1)
        thalach = st.number_input("Maximum heart rate achieved (thalach)", 50, 250, 168)
        exang = st.selectbox("Exercise-induced angina (exang)", [0, 1], format_func=YES_NO.get)
    with col3:
        oldpeak = st.number_input(
            "ST depression induced by exercise (oldpeak)", 0.0, 10.0, 1.0, step=0.1
        )
        slope = st.selectbox("Slope of peak exercise ST segment (slope)", [0, 1, 2], index=2)
        ca = st.selectbox("Major vessels colored by fluoroscopy (ca)", [0, 1, 2, 3, 4], index=2)
        thal = st.selectbox("Thalassemia code (thal)", [0, 1, 2, 3], index=3)

    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    # Field names must match the API's PatientFeatures model (= heart.csv columns).
    payload = {
        "age": int(age), "sex": sex, "cp": cp, "trestbps": int(trestbps), "chol": int(chol),
        "fbs": fbs, "restecg": restecg, "thalach": int(thalach), "exang": exang,
        "oldpeak": round(float(oldpeak), 1), "slope": slope, "ca": ca, "thal": thal,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as error:
        st.error(f"Could not reach the API at {API_URL}: {error}")
    else:
        if response.ok:
            result = response.json()
            st.subheader("Result")
            res1, res2 = st.columns(2)
            res1.metric("Predicted class (target)", result["prediction"])
            res2.metric("Probability of this class", f"{result['probability']:.1%}")
            st.progress(result["probability"])
            st.caption(
                "The class uses the coding of the `target` column in heart.csv "
                "(the Kaggle dataset description says 1 = heart disease, 0 = no disease)."
            )
        else:
            st.error(f"The API returned an error (HTTP {response.status_code}).")
            st.json(response.json())

    with st.expander("Request sent to the API"):
        st.json(payload)

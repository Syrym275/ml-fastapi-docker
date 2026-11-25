# Python base image
FROM python:3.13-slim

# Working directory inside the container
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install dependencies in several steps instead of one. Docker caches every
# step that finishes, so on a slow or unstable network a retry only repeats the
# step that failed instead of downloading everything again.
# (Versions match requirements.txt; numpy and scipy are scikit-learn dependencies.)
RUN pip install --no-cache-dir --timeout 120 --retries 10 numpy==2.5.5
RUN pip install --no-cache-dir --timeout 120 --retries 10 scipy==1.18.1
RUN pip install --no-cache-dir --timeout 120 --retries 10 pandas==3.0.5
RUN pip install --no-cache-dir --timeout 120 --retries 10 scikit-learn==1.9.1

# Remaining (small) dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt

# Copy the project files (main.py, model.joblib, ...)
COPY . .

# Port the API listens on
EXPOSE 8000

# Run the FastAPI app with Uvicorn (0.0.0.0 so it is reachable from outside the container)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

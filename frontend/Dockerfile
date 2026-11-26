# Streamlit frontend image
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Retries and a long timeout help on a slow or unstable network.
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt

COPY app.py .

EXPOSE 8501

# 0.0.0.0 so the page is reachable from outside the container;
# headless + no usage stats so Streamlit does not ask questions on start.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]

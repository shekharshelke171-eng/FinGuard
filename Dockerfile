FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && groupadd --gid 10001 finguard \
    && useradd --uid 10001 --gid finguard --no-create-home --shell /usr/sbin/nologin finguard \
    && mkdir -p /app/database /tmp/matplotlib \
    && chown -R 10001:10001 /app/database /tmp/matplotlib

COPY config/ ./config/
COPY dashboard/ ./dashboard/
COPY database/connection.py database/models.py ./database/
COPY src/ ./src/
COPY models/fraud_detection_model.joblib ./models/fraud_detection_model.joblib
COPY data/raw/financial_fraud_detection_dataset.csv ./data/raw/financial_fraud_detection_dataset.csv
COPY data/processed/processed_transactions.csv ./data/processed/processed_transactions.csv

USER 10001:10001
EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

# TODO (Task 4)
# Write a Dockerfile for the FastAPI service using python:3.12-slim as the base image.
# Documentation: https://docs.docker.com/reference/dockerfile/
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable buffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV MLFLOW_TRACKING_URI="http://host.docker.internal:5000"
ENV MLFLOW_EXPERIMENT_NAME="sentiment-eval"

ENV MODEL_NAME="sentiment"
ENV MODEL_STAGE="production"
ENV MODEL_SOURCE="mlflow"

ENV HF_MODEL_ID="baptle/FinBERT_market_based"
ENV HF_DATASET_ID="baptle/financial_headlines_market_based"


# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /prediction-api

COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy project files
COPY /app ./app
COPY /data ./data
COPY /scripts ./scripts/
COPY mlruns/ ./app/mlruns/

RUN python scripts/smoke_test.py && \
    python scripts/evaluate.py && \
    python scripts/promote.py

# Expose API port
EXPOSE 8000

# Run FastAPI app with Uvicorn
CMD ["python", "app/main.py"]

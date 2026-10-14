# TODO (Task 4)
# Write a Dockerfile for the FastAPI service using python:3.12-slim as the base image.
# Documentation: https://docs.docker.com/reference/dockerfile/
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable buffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /prediction-api

# Copy project files
COPY /app /app
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Expose API port
EXPOSE 8000

# Run FastAPI app with Uvicorn
CMD ["python", "app/main.py"]

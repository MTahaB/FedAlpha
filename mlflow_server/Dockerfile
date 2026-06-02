FROM python:3.11-slim

WORKDIR /app
COPY requirements-mlflow.txt .
RUN pip install --no-cache-dir -r requirements-mlflow.txt
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]

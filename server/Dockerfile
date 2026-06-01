FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY quant ./quant
COPY federated_learning ./federated_learning
CMD ["python", "-m", "federated_learning.server"]

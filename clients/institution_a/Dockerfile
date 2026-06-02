FROM python:3.11-slim

WORKDIR /app
COPY requirements-client.txt .
RUN pip install --no-cache-dir -r requirements-client.txt
COPY quant ./quant
COPY federated_learning ./federated_learning
COPY clients ./clients
CMD ["python", "-m", "clients.institution_a.train"]

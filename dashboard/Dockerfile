FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app
COPY requirements-dashboard.txt .
RUN pip install --no-cache-dir -r requirements-dashboard.txt
COPY blockchain ./blockchain
COPY clients ./clients
COPY data ./data
COPY dashboard ./dashboard
COPY federated_learning ./federated_learning
COPY oracle ./oracle
COPY quant ./quant
COPY tests ./tests
COPY verification ./verification
CMD ["streamlit", "run", "dashboard/app.py", "--server.address", "0.0.0.0"]

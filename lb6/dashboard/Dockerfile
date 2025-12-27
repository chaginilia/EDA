FROM python:3.11-slim

WORKDIR /app

COPY dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY eda.py .
COPY dashboard/generate_dashboard.py .
COPY dashboard/run_dashboard.py .

RUN mkdir -p /app/trained_model

CMD python generate_dashboard.py && python run_dashboard.py
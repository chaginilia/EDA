FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY eda.py .
COPY model.py .
COPY app.py .

RUN mkdir -p /app/trained_model

CMD python model.py && python app.py
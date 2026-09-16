FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python ml/train.py
RUN mkdir -p /app/ml/model_template \
    && cp /app/ml/saved_model/model.joblib /app/ml/model_template/model.joblib \
    && cp /app/ml/saved_model/target_names.joblib /app/ml/model_template/target_names.joblib \
    && cp /app/ml/saved_model/model_metadata.json /app/ml/model_template/model_metadata.json

EXPOSE 8000

CMD ["python", "scripts/start_api.py"]

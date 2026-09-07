FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --host 0.0.0.0 is required inside Docker because:
# 127.0.0.1 inside a container only listens on the container's own loopback.
# External requests (including docker run -p 8000:8000) arrive on the container's
# network interface, so 0.0.0.0 tells uvicorn to accept connections on all interfaces.

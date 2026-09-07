# Use a lightweight Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required by scikit-learn/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Run the app
# --host 0.0.0.0 is required inside Docker because:
# 127.0.0.1 inside a container only listens on the container's own loopback.
# External requests (including docker run -p 8000:8000) arrive on the container's
# network interface, so 0.0.0.0 tells uvicorn to accept connections on all interfaces.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.9-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install numpy first as it is required for building cyvlfeat
RUN pip install --no-cache-dir numpy>=1.21.0

# Copy requirements first to leverage cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Set entrypoint
ENTRYPOINT ["python", "src/run_detection.py"]

ARG RUNTIME=cpu
FROM python:3.11-slim AS base

WORKDIR /app

# System deps needed by opencv and numba
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -U pip setuptools wheel

# Install cpu or gpu extras based on build arg
RUN pip install --no-cache-dir -e ".[all,${RUNTIME}]"

ENTRYPOINT ["python", "scripts/main_runner.py"]
CMD ["--help"]
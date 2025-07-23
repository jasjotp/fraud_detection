# ---------- src/api/Dockerfile ----------
# Use the official slim Python image as a tiny, secure base
FROM python:3.11-slim

# ------------------------------------------------------------------
# 1. Environment hygiene (less noise, smaller layers)
# ------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ------------------------------------------------------------------
# 2. System packages — gcc & librdkafka-dev are needed
#    if you use aiokafka or confluent-kafka.
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc build-essential librdkafka-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
# 3. Copy *only* requirements first for better layer caching.
# ------------------------------------------------------------------
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# ------------------------------------------------------------------
# 4. Copy the actual FastAPI application code.
#    Directory structure must be:
#       ./app/main.py
#       ./app/routers/...
# ------------------------------------------------------------------
COPY app ./app

ENV PYTHONPATH=/app
# ------------------------------------------------------------------
# 5. Expose the default port Uvicorn will listen on.
# ------------------------------------------------------------------
EXPOSE 8000

# ------------------------------------------------------------------
# 6. Start Gunicorn with the Uvicorn worker class (production-grade).
#    • 4 workers by default (change via env if you like)
#    • binds to all IPs so Docker forwards to host.
# ------------------------------------------------------------------
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "-b", "0.0.0.0:8000", "--workers", "4"]

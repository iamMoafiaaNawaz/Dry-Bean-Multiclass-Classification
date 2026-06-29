# ==============================================================
# Dry Bean Classification — Dockerfile
# ==============================================================
#
# Two-stage build:
#   1. builder  — installs Python dependencies into a venv
#   2. runtime  — copies only the venv + source; no build tools
#
# This keeps the final image lean (~600 MB vs ~1.2 GB single-stage).
#
# Usage
# -----
#   # Build image
#   docker build -t dry-bean-classifier .
#
#   # Train the model (downloads dataset automatically)
#   docker run --rm dry-bean-classifier
#
#   # Run prediction with demo sample
#   docker run --rm dry-bean-classifier python src/predict.py
#
#   # Run pytest test suite
#   docker run --rm dry-bean-classifier pytest tests/ -v
#
#   # Persist trained model to host machine
#   docker run --rm -v "%cd%/models:/app/models" dry-bean-classifier
# ==============================================================


# ──────────────────────────────────────────────────────────────
# Stage 1 — Builder
# ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Prevent .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system build dependencies needed by numpy / scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip first — avoids legacy resolver issues
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ──────────────────────────────────────────────────────────────
# Stage 2 — Runtime
# ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Point to the venv from the builder stage
    PATH="/opt/venv/bin:$PATH" \
    # Log level can be overridden at runtime:
    #   docker run -e LOG_LEVEL=DEBUG dry-bean-classifier
    LOG_LEVEL=INFO

WORKDIR /app

# Copy virtual environment from builder (no gcc / build tools in final image)
COPY --from=builder /opt/venv /opt/venv

# Copy project source
COPY src/       ./src/
COPY tests/     ./tests/
COPY data/Dry_Bean_Dataset.csv      ./data/Dry_Bean_Dataset.csv

# Pre-create models/ directory so train.py can write artefacts
RUN mkdir -p models

# Smoke-test: verify the interpreter and key packages are importable
RUN python -c "import sklearn; import xgboost; import pandas; print('✅ All packages OK')"

# Default command — run the full training pipeline
# Override with:  docker run dry-bean-classifier python src/predict.py
CMD ["python", "src/train.py"]
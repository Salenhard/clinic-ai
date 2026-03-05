# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="clinical-graph-builder"
LABEL description="Clinical decision graph extractor from PDF guidelines (Claude API)"

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=appuser:appuser . .

# Create directories that are written to at runtime
RUN mkdir -p /app/pipeline_cache /data/input /data/output \
 && chown -R appuser:appuser /app /data

USER appuser

# /data/input  — mount your PDF files here
# /data/output — graph.json and metrics.json are written here
VOLUME ["/data/input", "/data/output"]

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "main.py"]

# Default flags — override via `docker run ... -- --section "..."`
CMD ["--help"]

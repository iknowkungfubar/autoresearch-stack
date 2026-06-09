# Dockerfile for Autonomous Research Stack
FROM python:3.14-slim

LABEL maintainer="turin@autoresearch.io"
LABEL description="Autonomous LLM training research stack"

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import config; print('OK')" || exit 1

# Default command - prepare data
CMD ["python", "autonomous_loop.py", "--prepare-only"]

# Dockerfile for Autonomous Research Stack
# Build: docker build -t autoresearch-stack .
# Run: docker run --rm -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY autoresearch-stack

# hadolint ignore=DL3007
FROM python:3.11-slim AS builder

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

# --- Production stage ---
FROM python:3.11-slim

# Create non-root user first so subsequent COPY --chown works
RUN useradd -m -u 1000 appuser

WORKDIR /app

# No runtime system packages needed beyond the slim base

# Copy installed packages from builder (ownership set to appuser)
COPY --from=builder --chown=appuser:appuser /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder --chown=appuser:appuser /usr/local/bin /usr/local/bin
COPY --from=builder --chown=appuser:appuser /app /app

# Ensure appuser owns everything
RUN chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import config; print('OK')" || exit 1

# Default command - prepare data
CMD ["python", "autonomous_loop.py", "--prepare-only"]

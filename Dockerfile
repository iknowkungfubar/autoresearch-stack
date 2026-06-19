# Dockerfile for Autonomous Research Stack
FROM python:3.13-slim

LABEL maintainer="turin@autoresearch.io"
LABEL description="Autonomous LLM training research stack"

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy install config first for caching
COPY pyproject.toml README.md ./
COPY src/ src/

# Install the package
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from autoresearch import __version__; print(f'OK v{__version__}')" || exit 1

# Default command
CMD ["python", "-m", "autoresearch", "--prepare-only"]

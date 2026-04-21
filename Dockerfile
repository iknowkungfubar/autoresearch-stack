# Dockerfile for Autonomous Research Stack
# Build: docker build -t autoresearch-stack .
# Run: docker run --rm -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY autoresearch-stack

FROM python:3.11-slim

LABEL maintainer="turin@autoresearch.io"
LABEL description="Autonomous LLM training research stack"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Default command - prepare data
CMD ["python", "autonomous_loop.py", "--prepare-only"]
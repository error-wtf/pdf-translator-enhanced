# PDF Translator Enhanced - Docker Image
# 
# Build: docker build -t pdf-translator .
# Run:   docker run -p 7860:7860 pdf-translator
#
# With GPU: docker run --gpus all -p 7860:7860 pdf-translator
#
# © 2025 Sven Kalinowski with small help of Lino Casu
# Licensed under the Anti-Capitalist Software License v1.4

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim

LABEL maintainer="mail@error.wtf"
LABEL description="PDF Translator Enhanced - Scientific PDF Translation"
LABEL version="2.0.0"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # PDF processing
    poppler-utils \
    # Fonts for various languages
    fonts-dejavu-core \
    fonts-liberation \
    fonts-noto-cjk \
    fonts-noto-core \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY *.py ./
COPY tests/ ./tests/

# Create directories
RUN mkdir -p /app/uploads /app/outputs /app/cache

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/')" || exit 1

# Default command - run Gradio UI
CMD ["python", "gradio_app.py"]

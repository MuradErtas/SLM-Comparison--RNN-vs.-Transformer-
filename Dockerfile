FROM python:3.11-slim

WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch CPU-only (much smaller)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch>=2.0.0

# Install other dependencies
RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    bpeasy>=0.1.0 \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    pydantic>=2.0.0

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the API server
CMD ["python", "api_server.py"]
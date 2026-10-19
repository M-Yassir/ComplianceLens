FROM python:3.13-slim

# Prevent Python from writing bytecode and buffer outputs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install minimal system build dependencies and curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary directly from Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency definition for Docker layer caching
COPY requirements.txt .

# Install Python dependencies directly into system environment using uv
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application source code and data assets
COPY src/ ./src
COPY data/ ./data

# Expose API port
EXPOSE 8000

# Healthcheck endpoint
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Launch FastAPI via Uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
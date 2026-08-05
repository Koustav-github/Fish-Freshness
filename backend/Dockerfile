FROM python:3.12-slim

WORKDIR /app

# System deps required by OpenCV and rembg/onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install all Python dependencies (frozen = exact versions from uv.lock)
RUN uv sync --frozen --no-cache

# Copy application code and model files
COPY main.py ./
COPY models/ ./models/

EXPOSE 8000

# Use uv run so it picks up the managed virtualenv automatically
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

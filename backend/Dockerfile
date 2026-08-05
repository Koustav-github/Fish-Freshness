FROM python:3.12-slim

WORKDIR /app

# --- CHANGED SECTION: image size ---
# libgl1 dropped — opencv-python-headless doesn't link against Mesa/OpenGL,
# only the full opencv-python GUI build needs it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# -----------------------

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install all Python dependencies (frozen = exact versions from uv.lock)
RUN uv sync --frozen --no-cache

# Copy application code
# --- CHANGED SECTION: AWS SageMaker integration ---
# No local model files to copy anymore — YOLO + classifier now run on the
# deployed SageMaker endpoint, called over the network from main.py.
COPY main.py ./
# -----------------------

EXPOSE 8000

# Use uv run so it picks up the managed virtualenv automatically
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

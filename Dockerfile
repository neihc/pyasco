FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Create virtual environment
ENV UV_SYSTEM_PYTHON=1

# Copy requirements and install dependencies
COPY pyproject.toml .
RUN uv pip install -r pyproject.toml

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /root/.pyasco/workspace /root/.pyasco/memories

CMD ["python", "-m", "pyasco.app.console"]

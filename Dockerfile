FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir uv
RUN uv sync

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /root/.pyasco/workspace /root/.pyasco/memories

CMD ["uv", "run", "-m", "pyasco.app.console"]

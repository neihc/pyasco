FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .
COPY uv.lock .

# Install Python dependencies
RUN pip install --no-cache-dir uv && \
    uv pip install -r uv.lock

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /root/.pyasco/workspace /root/.pyasco/memories

CMD ["python", "-m", "pyasco.app.console"]

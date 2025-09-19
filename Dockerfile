
# Use Python base image
FROM python:3.13-slim

LABEL MAINTAINER="John Naylor <jonaylor89@gmail.com>"

# Install uv
RUN pip install uv

# cd to /app
WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy source code to /app
COPY . /app/

RUN mkdir -p /app/datasets/Cancerous_cell_smears

# run script
CMD ["python3", "/app/main.py"]

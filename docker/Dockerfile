ARG BASE_IMAGE=pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies, including C/C++ compiler for Triton
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (torch/torchvision come from the base image)
COPY /requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Keep scripts available even without bind-mount
COPY /docker/ /app/docker/

# Default command is set by docker-compose
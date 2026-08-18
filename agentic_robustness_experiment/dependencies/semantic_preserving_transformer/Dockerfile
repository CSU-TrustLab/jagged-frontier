FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG CODEQL_VERSION=latest

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml README.md /app/
COPY semantic_transformer /app/semantic_transformer

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

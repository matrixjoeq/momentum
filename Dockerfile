FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install project (build from pyproject.toml)
COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src

RUN python3 -m pip install -U pip && \
    python3 -m pip install .

# WeChat Cloud Hosting usually provides PORT env var; default to 8080.
EXPOSE 8080

# NOTE: python:3.11-slim does not guarantee bash; use sh for portability.
# WeChat Cloud Hosting usually provides PORT env var; default to 8080.
CMD ["sh", "-lc", "uvicorn etf_momentum.app:app --host 0.0.0.0 --port ${PORT:-8080}"]


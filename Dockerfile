# syntax=docker/dockerfile:1.7
# ============================================================================
# AMLGuard - production container image
#
# Multi-stage build: a "builder" stage installs the Python dependencies into a
# virtualenv, then a slim "runtime" stage copies only what is needed to run
# the service. This keeps the final image small (~500 MB) and free of build
# toolchains that would otherwise be an attack surface.
#
# Build:
#     docker build -t amlguard:latest .
#
# Run (no model - useful for exploring /docs, /health):
#     docker run --rm -p 8000:8000 amlguard:latest
#
# Run (with a model already produced by `python -m src.models.train`):
#     docker run --rm -p 8000:8000 \
#         -v ${PWD}/artifacts/model.joblib:/app/artifacts/model.joblib:ro \
#         amlguard:latest
# ============================================================================

# ---------- builder stage ---------------------------------------------------- #
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build tools are only needed while installing wheels that may require compilation.
# Pinning apt package versions is deliberately skipped: it breaks whenever the
# base python:3.12-slim image refreshes its apt catalogue, which happens more
# often than build-essential itself changes in ways that affect this build.
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install -r requirements.txt \
    && /opt/venv/bin/pip install "fastapi>=0.110" "uvicorn[standard]>=0.29" "pydantic>=2.0"

# ---------- runtime stage --------------------------------------------------- #
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

# Run as a non-root user - defence in depth even for a stateless read model.
RUN groupadd --system amlguard \
    && useradd --system --gid amlguard --home /app --shell /usr/sbin/nologin amlguard

WORKDIR /app

# Copy the virtualenv from the builder stage (isolated from system Python).
COPY --from=builder /opt/venv /opt/venv

# Copy only what the runtime needs. .dockerignore filters out everything else,
# but being explicit here keeps the image lean and auditable.
COPY --chown=amlguard:amlguard src ./src
COPY --chown=amlguard:amlguard artifacts/baseline_metrics.json ./artifacts/baseline_metrics.json
COPY --chown=amlguard:amlguard pyproject.toml ./pyproject.toml

# The model artifact is provisioned at run-time (via a bind mount or a follow-up
# `docker cp`). Create the target directory with the right ownership so it
# exists even without a mount.
RUN mkdir -p /app/artifacts /app/data/raw \
    && chown -R amlguard:amlguard /app/artifacts /app/data

USER amlguard

EXPOSE 8000

# HEALTHCHECK asks the running process whether it is up. Orchestrators
# (Kubernetes, docker compose, ECS) use this to decide when to send traffic
# and when to restart a wedged container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
                    sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

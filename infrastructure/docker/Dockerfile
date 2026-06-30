# ---- Stage 1: builder — has UV, builds the virtual environment ----
FROM python:3.11-slim AS builder

# Install UV by copying its binary from the official UV image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy ONLY dependency manifests first. This layer is cached and reused
# on rebuilds as long as these two files are unchanged — so editing your
# source code below does NOT re-run the slow dependency install.
COPY pyproject.toml uv.lock ./

# --frozen: install exactly what uv.lock pins, never re-resolve (fails if stale).
# --no-dev: production only — skips pytest, ruff, mypy.
# --no-install-project: install deps only, not our own code (copied later).
RUN uv sync --frozen --no-dev --no-install-project

# ---- Stage 2: runtime — slim image with just Python + the built venv ----
FROM python:3.11-slim AS runtime

# OCI image metadata. Labels live in the runtime stage so they ship in the
# final image; labels in the builder stage are discarded with it.
LABEL org.opencontainers.image.title="inference-service" \
      org.opencontainers.image.description="Clinical AI Platform inference API" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.source="https://github.com/ajayaneesh/clinical-ai-platform"

WORKDIR /app

# Copy ONLY the finished virtual environment from the builder stage.
# UV, build caches, and dependency manifests are left behind — not shipped.
COPY --from=builder /app/.venv /app/.venv

# Put the venv on PATH so `uvicorn` resolves without activation.
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source last — the most frequently changing layer.
COPY backend/ ./backend/

# Run from backend/ so the absolute import `app.main:app` resolves.
WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

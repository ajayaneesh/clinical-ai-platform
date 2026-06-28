# Clinical AI Platform — CLAUDE.md

## Project Purpose

Personal AI project for career transition portfolio. A multi-service Clinical AI Platform built in Python demonstrating ML inference, RAG, orchestration, evaluation, and monitoring patterns common in production AI systems.

## Target Architecture

Microservices layout — each service is an independent Python package with its own `pyproject.toml` and FastAPI application:

```
clinical-ai-platform/
├── inference-service      # Model serving: LLM/ML inference endpoints
├── embedding-service      # Text → vector embeddings (sentence-transformers, OpenAI, etc.)
├── retrieval-service      # RAG retrieval: vector DB queries + reranking
├── evaluation-service     # Offline + online eval: metrics, LLM-as-judge, RAGAS
├── orchestration-service  # Workflow coordination across services (LangGraph / custom)
├── training-service       # Fine-tuning, dataset prep, experiment tracking
├── monitoring-service     # Observability: traces, metrics, drift detection
├── frontend               # UI (TBD — likely React or Streamlit)
└── infrastructure/        # Docker Compose, Kubernetes manifests, shared config
```

## Current State (as of 2026-06-27)

Scaffold only. The `backend/` directory predates the microservices decision and will be replaced/reorganised. No service code exists yet.

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.12+ |
| Package manager | **UV** (not pip/poetry) |
| Web framework | **FastAPI** |
| Async server | Uvicorn |
| Data validation | Pydantic v2 |
| Containerisation | Docker + Docker Compose |
| CI | GitHub Actions (`.github/workflows/`) |

## Conventions

### Python / UV

- Every service has its own `pyproject.toml`. Use `uv add <pkg>` to add dependencies, never `pip install`.
- Use `uv run` to execute scripts inside a service's virtual environment.
- Lock files (`uv.lock`) are committed.
- Python version pinned via `.python-version` file per service.

### FastAPI Services

- Entry point: `src/<service_name>/main.py` — creates the `FastAPI()` app instance.
- Routers live in `src/<service_name>/api/` and are registered in `main.py`.
- Business logic in `src/<service_name>/services/`, never inline in route handlers.
- Pydantic models for all request/response schemas in `src/<service_name>/schemas/`.
- Config via `pydantic-settings` reading from environment variables + `.env` files.
- Health check at `GET /health` on every service.

### Code Style

- No comments unless the WHY is non-obvious.
- No docstrings on trivial functions.
- `ruff` for linting and formatting (replaces black + flake8).
- `mypy` for type checking (strict mode target).

### Testing

- `pytest` + `httpx` (async test client for FastAPI).
- Tests live in `tests/` at the service root, mirroring `src/` structure.
- Use real dependencies where cheap; mock only external network calls.

### Git

- Branch from `main` for features.
- Commit messages: imperative, ≤72 chars, no trailing period.
- No force-push to `main`.

## Service Responsibilities (planned)

### inference-service
Wraps model backends (Hugging Face, vLLM, Ollama, OpenAI-compatible APIs). Exposes a unified `/infer` endpoint. Handles batching and streaming.

### embedding-service
Produces dense vector embeddings. Supports multiple encoder models. Used by retrieval-service and evaluation-service.

### retrieval-service
Manages a vector store (ChromaDB or Qdrant initially). Handles document ingestion, chunking, indexing, and similarity search with optional reranking.

### evaluation-service
Computes retrieval and generation metrics (RAGAS, BERTScore, BLEU, custom LLM-as-judge). Accepts trace payloads and returns scored results.

### orchestration-service
Coordinates multi-step pipelines (e.g. query → retrieve → augment → infer → evaluate). Will use LangGraph or a lightweight custom DAG runner.

### training-service
Dataset curation, fine-tuning jobs (LoRA/QLoRA via PEFT), experiment tracking (MLflow or W&B).

### monitoring-service
Collects traces (OpenTelemetry), exposes Prometheus metrics, detects data/concept drift, feeds dashboards (Grafana).

### frontend
Likely a chat UI + evaluation dashboard. Technology TBD (React + shadcn/ui or Streamlit for speed).

## Infrastructure

- `infrastructure/docker/` — per-service Dockerfiles and a root `docker-compose.yml` for local dev.
- All services communicate over HTTP (REST) internally; consider gRPC for embedding/inference hot paths later.
- Shared secrets via `.env` files locally; never committed.

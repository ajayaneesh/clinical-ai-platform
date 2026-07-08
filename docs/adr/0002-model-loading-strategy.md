# ADR-0002: Model Loading Strategy

- **Status:** Accepted
- **Date:** 2026-07-07
- **Deciders:** project owner

> Note: ADR-0001 has not been written yet. This is numbered 0002 as requested;
> a future ADR-0001 (e.g. "Queue-based inference pipeline") can backfill the gap.

## Context

The inference pipeline serves image classifications through a queue and a
batching worker (see the queue/worker design). The worker calls
`InferenceService.predict_batch`, which is backed by a concrete `InferenceModel`
(currently a placeholder `TorchInferenceModel`, or a real
`HuggingFaceInferenceModel` when `CLINICAL_AI_MODEL_ID` is set).

A model must be turned into a usable object before it can serve requests:
download/read weights, deserialize them, and move them to the compute device
(CPU / MPS / CUDA). This is expensive — seconds for a real model — and the
question is **when and how often** to pay that cost.

## Decision

**Load the model exactly once, at application startup, and hold it in memory for
the process lifetime.**

Concretely:
- The model is constructed in the FastAPI `lifespan` startup hook
  (`get_inference_service()` → `build_model_manager()` → `_build_current_model()`),
  before any request is served.
- The load is wrapped in timing and recorded as the `model_cold_start_seconds`
  gauge (see the metrics work).
- The constructed model is held by the `ModelManager` (a registry) and shared by
  the worker(s) for every request. It is **never** reloaded per request.
- Readiness (`GET /ready`) gates traffic on the model being loaded, so no request
  hits a half-initialized model.

## Rationale (the questions this ADR answers)

### Why load the model once?

Loading is slow (seconds for a real model: weight I/O + deserialization + device
transfer). If we loaded per request, **every** request would pay that cost —
turning a millisecond inference into a multi-second request, and making the
service unusably slow under any load. Loading once amortizes the cost across the
entire process lifetime: pay it at startup, then every request reuses the
already-resident model. This is the standard model-serving lifecycle: **load
once, infer many.**

### Why keep it in memory?

A model is a large in-memory object (weights as tensors on the device). Keeping
it resident means inference is a direct in-memory/on-device operation with no
reload. The alternatives are worse:
- **Reload from disk per request** → catastrophic latency (see above).
- **Load lazily on first request** → the first user eats the full cold-start
  latency (a cold-start spike), and concurrent first requests could trigger
  duplicate loads. We load eagerly at startup instead, so cold start is paid
  before traffic and is observable via the readiness probe + cold-start metric.

Keeping it in memory is what makes warm inference latency (the
`forward_pass_latency_seconds` metric) small and stable.

### What happens if the model is 20 GB?

This is where "just keep it in memory" hits physical limits, and the strategy
must adapt:

- **Memory sizing becomes the binding constraint.** A 20 GB model needs a host /
  container / pod with enough RAM (or GPU VRAM) to hold it plus working memory
  for batches. Container memory limits and Kubernetes `requests`/`limits` must be
  set from this — undersize and the process is OOM-killed. The
  `process_memory_bytes` gauge exists precisely to measure this.
- **Cold start grows.** Reading and deserializing 20 GB takes real time; the
  `model_cold_start_seconds` metric will be large. This lengthens autoscaling
  response (a new replica is useless until loaded) and argues for keeping
  replicas warm rather than scaling from zero on demand.
- **One model per process, shared aggressively.** You cannot afford multiple
  copies. The single-batching-worker design (one consumer, one loaded model)
  already avoids per-worker duplication. Multiple Python *processes* on one host
  would each hold their own 20 GB copy — prohibitive — which pushes toward one
  process per host, or GPU-sharing serving engines.
- **GPU/quantization considerations.** At 20 GB the model likely must live on a
  GPU (or be sharded across GPUs). Techniques like quantization (fp16/int8),
  tensor/pipeline parallelism, or a dedicated serving engine (vLLM/Triton) become
  necessary — the model stops being "an object we construct in `__init__`" and
  becomes infrastructure the API talks to over the network.
- **Lazy vs eager tradeoff resurfaces.** For very large models, some systems
  accept a lazy first-load or a warm-pool of pre-loaded replicas rather than
  loading in every replica's startup. The "load once at startup" rule holds
  *per replica*; the question becomes how many replicas can afford to.

### How would this change with multiple replicas?

"Load once" is **per process**, so with N replicas the model is loaded N times —
once in each. Consequences:

- **Total memory = N × model size.** 4 replicas of a 20 GB model = 80 GB of
  aggregate memory. This is the dominant cost of horizontal scaling for large
  models and constrains how many replicas are affordable.
- **Each replica is independent and self-sufficient.** A stateless replica that
  holds its own model needs no shared model store at request time — good for
  fault isolation and simple load balancing (any replica can serve any request).
- **Cold start is per replica.** Every new replica (autoscale-up, rollout,
  restart) pays the full load cost before it's ready. Readiness probes must
  account for this so traffic isn't routed to a still-loading replica.
- **Model versioning must be coordinated.** All replicas should serve the same
  version (or an intentional A/B split); the `ModelManager` registry is the seam
  where per-replica version selection would live. A rolling deployment swaps
  replicas one at a time, each cold-starting the new version.
- **For very large models, replicas may share a model server instead.** Rather
  than N copies, the API replicas become thin clients of a smaller pool of
  GPU-backed inference servers (the `inference-service` in the target
  architecture). Then "load once" moves to those servers, and the API replicas
  hold no model at all — the natural evolution as models outgrow the API process.

## Consequences

**Positive**
- Warm inference is fast and stable; startup cost is paid once and is observable.
- Simple, stateless replicas; any replica serves any request.
- The `ModelManager` seam supports versioning / future multi-model without
  changing the load-once lifecycle.

**Negative / risks**
- Memory footprint scales with replica count (N × model size) — the main cost
  driver for large models.
- Cold start is paid on every replica start; slow for large models, affecting
  autoscaling responsiveness.
- Above a certain model size, in-process loading stops being viable and the
  design must shift to a dedicated GPU serving tier.

## Related

- Queue + batching worker design (single batching worker feeds one loaded model).
- `ModelManager` registry (composition root for loaded models).
- Metrics: `model_cold_start_seconds`, `process_memory_bytes`,
  `forward_pass_latency_seconds` — the numbers that make this strategy observable.
- `docs/architecture/performance-baseline.md` — where measured cold-start /
  memory / latency figures are recorded.

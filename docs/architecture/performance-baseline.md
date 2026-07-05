# Performance Baseline — `/predict`

Client-side load-test results establishing the framework overhead floor for the
inference pipeline. This is a **reference artifact**, not a decision record —
re-run and update it when the model, concurrency model, or infrastructure changes.

## Context

- **Date:** 2026-07-04
- **Commit:** `a3ed1b9`
- **Model:** `DummyInferenceModel` — returns a fixed result, does **no real work**.
- **Environment:** Python 3.11.5, macOS (Darwin arm64), single local process.
- **Pipeline:** `POST /predict` → in-memory `LocalQueue` → single async worker
  (`run_in_executor` offload) → dummy model.
- **Tool:** `scripts/load_test.py` (client-side latency + throughput).
- **Command:**
  ```
  uv run python scripts/load_test.py --requests 100 500 1000 5000
  ```

## Results

| Requests | Concurrency | Wall (s) | Throughput (rps) | Success | Failed | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) |
|---------:|------------:|---------:|-----------------:|--------:|-------:|---------:|---------:|---------:|---------:|
| 100      | 50          | 0.106    | 947.2            | 100     | 0      | 37.39    | 59.77    | 62.46    | 62.46    |
| 500      | 50          | 0.488    | 1023.8           | 500     | 0      | 45.50    | 55.32    | 63.03    | 65.66    |
| 1000     | 50          | 0.964    | 1037.5           | 1000    | 0      | 45.45    | 55.10    | 61.47    | 71.18    |
| 5000     | 50          | 4.866    | 1027.5           | 5000    | 0      | 46.13    | 54.78    | 61.59    | 70.20    |

## Observations

- **Throughput plateaus at ~1000 rps** regardless of request count (947 → 1024 →
  1038 → 1028). This is the ceiling of the current setup at concurrency=50: a
  single worker on one event loop, offloading to the default thread pool. It is a
  *system/framework* limit, not a model limit (the model does nothing).
- **Latency is stable across load** — p50 ~45 ms, p95 ~55 ms, p99 ~62 ms hold
  steady from 500 to 5000 requests. No degradation or tail blow-up under sustained
  load, which indicates no queue backlog building up at this concurrency.
- **Zero failures** across 6600 total requests — the pipeline (queue, worker,
  timeout handling) is stable.
- The 100-request run shows lower throughput (947 rps) and higher p50 (37 ms) —
  warm-up / small-sample effects; the 500+ runs are the more reliable baseline.

## Caveats

- **These numbers are the overhead floor, not real performance.** The dummy model
  does no computation, so this measures *only* the framework/queue/worker stack —
  the fastest the system can possibly go. A real model adds its own latency on top;
  the value here is knowing how much of future latency is *stack* vs *model*.
- **Client-side timings.** Measured from the load-test client, so they include the
  local HTTP round-trip. Cross-check against server-side `inference_latency_seconds`
  and `http_request_latency_seconds` at `/metrics` — the gap is queue + transport.
- **Single local process, concurrency=50.** Not representative of a multi-worker or
  multi-instance production deployment.

## How to reproduce

```
# terminal 1
uv run serve
# terminal 2
uv run python scripts/load_test.py --requests 100 500 1000 5000
```

## Throughput-ceiling investigation (2026-07-05)

The ~1000 rps plateau was investigated. Root cause: **the single worker awaited
each job to completion before pulling the next**, so the pipeline was serial —
throughput capped at `1 / per-job-latency` regardless of thread-pool size or
client concurrency. (The default `run_in_executor` pool has 16 threads on this
machine, but only one was ever fed at a time.)

Controlled experiment with a fixed 5 ms model, 400 jobs:

| Workers (shared queue) | Throughput | Scaling |
|-----------------------:|-----------:|:--------|
| 1 (old design)         | 162 rps    | ~serial ceiling (1/5 ms ≈ 200) |
| 4                      | 642 rps    | ~4× |
| 8                      | 1253 rps   | ~8× |
| 16                     | 2456 rps   | ~16× |

Throughput scales ~linearly with worker count — confirming the single worker,
not the thread pool or client concurrency, was the ceiling.

**Fix:** worker count is now configurable via `CLINICAL_AI_WORKER_COUNT`
(default 4); `start_workers()` runs that many consumers on the shared queue.
`test_multiple_workers_increase_throughput` guards the scaling behavior.

### Correction: worker count does NOT help the dummy model (2026-07-05)

Re-running the *HTTP* load test with `worker_count=4` and `16` showed **no
significant rps gain, and 16 was slightly slower than 4.** The scaling table
above was measured with a 5 ms `sleep` model in isolation; it does not apply to
the dummy path. Decomposition of the real per-request cost:

| Step | Cost |
|---|---|
| `DummyInferenceModel.predict()` | ~0.1 µs (negligible) |
| `run_in_executor` thread handoff | ~32 µs (320× the model) |
| queue + worker round-trip throughput | 18k–39k rps (not the bottleneck) |
| `GET /health` (no queue/model at all) | ~3.8k rps |
| `POST /predict` (full HTTP path) | ~1.7k rps |

**Conclusion:** with a near-zero-work model the bottleneck is **per-request
overhead on the single event loop** (ASGI + middleware + Pydantic validation +
base64 decode + JSON serialization), NOT the workers. `/health` proves it: no
queue involved, still capped at ~3.8k rps.

Full model:

```
throughput = min(
    workers × (1 / per-job-work),     # worker-bound  -> matters for a SLOW real model
    1 / per-request-loop-overhead     # loop-bound     -> matters for a FAST/dummy model
)
```

The dummy is loop-bound, so more workers add coordination cost (extra executor
handoffs, event-loop contention) without benefit — hence 16 < 4. The
`worker_count` knob is still correct and will matter once a real, slow model
makes the pipeline worker-bound. To raise the *loop-bound* ceiling instead, run
multiple uvicorn processes (`--workers N`, one event loop each) for true
multi-core parallelism.

**~1000 rps is therefore the framework overhead floor** — the fastest this stack
moves a `/predict` request independent of the model — which is exactly what a
dummy baseline should measure.

## Follow-ups

- **Adopt Locust for load testing once a real model lands.** The current
  `scripts/load_test.py` fires a fixed pattern at fixed concurrency — it measures
  a *single point* on the latency-vs-load curve, which is all a dummy baseline
  needs. Locust adds what becomes valuable with a real (slow) model: ramp-up /
  spawn-rate control to find the **saturation point**, weighted multi-endpoint
  user journeys, a live dashboard, and distributed load across machines. Held off
  for now because against the zero-work dummy Locust would report the same ~1000
  rps loop-bound floor with more machinery — no new insight until there is real
  work to saturate. Likely keep `load_test.py` for quick/CI regression checks and
  use Locust for exploratory ramp analysis.

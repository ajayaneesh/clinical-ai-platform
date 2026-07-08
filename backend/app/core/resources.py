"""Background sampler that updates process CPU/memory gauges.

Prometheus Gauges hold a current value; something must refresh them. This task
samples the process every `interval` seconds and writes the gauges, so /metrics
always reflects recent resource usage.
"""

from __future__ import annotations

import asyncio
import logging
import os

import psutil

from app.core.metrics import PROCESS_CPU_PERCENT, PROCESS_MEMORY_BYTES

logger = logging.getLogger("app.resources")


async def sample_resources(interval: float) -> None:
    proc = psutil.Process(os.getpid())
    # First call establishes the baseline; cpu_percent is measured relative to
    # the previous call, so the first reading is discarded as 0.
    proc.cpu_percent()
    while True:
        await asyncio.sleep(interval)
        PROCESS_CPU_PERCENT.set(proc.cpu_percent())
        PROCESS_MEMORY_BYTES.set(proc.memory_info().rss)


def start_resource_sampler(interval: float) -> asyncio.Task[None]:
    return asyncio.create_task(sample_resources(interval))

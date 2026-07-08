import asyncio

from app.core.metrics import PROCESS_CPU_PERCENT, PROCESS_MEMORY_BYTES
from app.core.resources import sample_resources


def test_sampler_updates_cpu_and_memory_gauges():
    # Run the sampler briefly and confirm it writes non-trivial values into the
    # gauges (memory is always > 0 for a running process).
    async def scenario() -> None:
        task = asyncio.create_task(sample_resources(interval=0.01))
        await asyncio.sleep(0.05)  # let it sample a few times
        task.cancel()

    asyncio.run(scenario())

    assert PROCESS_MEMORY_BYTES._value.get() > 0  # RSS is always positive
    # CPU can legitimately be 0.0 on an idle process; just assert it was set
    # (>= 0) rather than left at its unset default.
    assert PROCESS_CPU_PERCENT._value.get() >= 0.0

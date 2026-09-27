"""
utils/memory_monitor.py

GPU VRAM monitoring for all training and inference stages.
Provides context managers and decorators to measure peak VRAM consumption
and issue warnings when approaching the configured limit.

Audit fix: PHASE 4 memory analysis — track actual vs estimated VRAM.
"""

import gc
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# VRAM limit for safety warnings (in bytes). 12 GB = 12884901888
DEFAULT_VRAM_LIMIT_GB = 12.0


@dataclass
class MemoryStats:
    """Snapshot of GPU memory state at a point in time."""
    stage: str
    allocated_gb: float
    reserved_gb: float
    peak_allocated_gb: float
    timestamp: float


class MemoryMonitor:
    """
    Tracks GPU memory consumption across pipeline stages.

    Records peak VRAM for each named stage. Warns if VRAM usage
    exceeds the configured safety threshold.

    Usage:
        monitor = MemoryMonitor(limit_gb=12.0)
        with monitor.track("unet_training"):
            train_unet(...)
        monitor.report()
    """

    def __init__(self, limit_gb: float = DEFAULT_VRAM_LIMIT_GB) -> None:
        """
        Args:
            limit_gb: VRAM limit in gigabytes. Warnings are issued above
                      90% of this value; errors logged above 95%.
        """
        self._limit_bytes = limit_gb * (1024 ** 3)
        self._warn_threshold = 0.90
        self._error_threshold = 0.95
        self._history: list[MemoryStats] = []
        self._device_available = torch.cuda.is_available()

        if not self._device_available:
            logger.warning("CUDA not available. Memory monitoring disabled.")

    @contextmanager
    def track(self, stage_name: str):
        """
        Context manager that measures peak VRAM during a code block.

        Args:
            stage_name: Human-readable label for this pipeline stage.

        Yields:
            None. Memory measurements are recorded internally.
        """
        if not self._device_available:
            yield
            return

        # Clear peak statistics before measurement
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_time = time.time()

        try:
            yield
        finally:
            torch.cuda.synchronize()
            elapsed = time.time() - start_time

            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

            stats = MemoryStats(
                stage=stage_name,
                allocated_gb=allocated,
                reserved_gb=reserved,
                peak_allocated_gb=peak,
                timestamp=elapsed,
            )
            self._history.append(stats)

            peak_bytes = peak * (1024 ** 3)
            util_fraction = peak_bytes / self._limit_bytes

            log_msg = (
                f"[MemoryMonitor] Stage='{stage_name}' | "
                f"Peak={peak:.2f}GB | Current={allocated:.2f}GB | "
                f"Reserved={reserved:.2f}GB | Time={elapsed:.1f}s | "
                f"Limit util={util_fraction*100:.1f}%"
            )

            if util_fraction >= self._error_threshold:
                logger.error(log_msg + " — CRITICAL: approaching OOM")
            elif util_fraction >= self._warn_threshold:
                logger.warning(log_msg + " — WARNING: high VRAM usage")
            else:
                logger.info(log_msg)

    def get_current_vram_gb(self) -> float:
        """Return currently allocated VRAM in gigabytes."""
        if not self._device_available:
            return 0.0
        return torch.cuda.memory_allocated() / (1024 ** 3)

    def clear_cache(self) -> None:
        """
        Aggressively free GPU memory between pipeline stages.
        Must be called when switching from U-Net to SAM.

        Audit fix: CRITICAL-11 — explicit memory clearing between models.
        """
        gc.collect()
        if self._device_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("GPU cache cleared.")

    def report(self) -> dict:
        """
        Generate a summary table of all tracked stages.

        Returns:
            Dictionary mapping stage_name → MemoryStats.
        """
        report = {}
        logger.info("=" * 60)
        logger.info("MEMORY USAGE REPORT")
        logger.info("=" * 60)
        for stats in self._history:
            logger.info(
                f"  {stats.stage:<30} peak={stats.peak_allocated_gb:.2f}GB  "
                f"time={stats.timestamp:.1f}s"
            )
            report[stats.stage] = stats
        logger.info("=" * 60)
        return report


def free_model_memory(model: Optional[torch.nn.Module], optimizer=None) -> None:
    """
    Explicitly remove a model (and optionally its optimizer) from GPU memory.

    Call this between U-Net and SAM stages to prevent simultaneous occupancy.

    Audit fix: CRITICAL-11 — prevents both models from being in VRAM simultaneously.

    Args:
        model: PyTorch model to delete from memory.
        optimizer: Optional optimizer to also delete.
    """
    if model is not None:
        del model
    if optimizer is not None:
        del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("Model memory freed and GPU cache cleared.")

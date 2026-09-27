"""
utils/reproducibility.py

Centralized reproducibility management.
Every training and evaluation script must call seed_everything() before
any computation. This ensures identical results across Colab sessions.

Audit fix: CRITICAL-21 — Set ALL random sources, not just torch seed.
"""

import os
import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """
    Set seeds for all random number generators used by the project.

    Sets: Python random, NumPy, PyTorch CPU, PyTorch GPU (all devices),
    and enables CUDA deterministic mode.

    Args:
        seed: Integer seed value. Must be identical across all runs for
              full reproducibility.

    Note:
        CUDA deterministic mode may reduce throughput by ~5-10%. This is
        acceptable given the reproducibility guarantee required.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU (not used, but defensive)

    # Ensure deterministic CUDA operations.
    # benchmark=False prevents cuDNN from selecting fastest (non-deterministic) kernel.
    # deterministic=True forces deterministic convolutions.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # CUBLAS determinism (introduced PyTorch 1.8+)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        # Older PyTorch versions may not have this
        pass

    logger.info(f"Global seed set to {seed}. CUDA deterministic mode enabled.")


def get_worker_init_fn(base_seed: int = 42):
    """
    Returns a DataLoader worker init function that seeds each worker
    independently but reproducibly.

    Args:
        base_seed: The global seed. Each worker gets seed = base_seed + worker_id.

    Returns:
        A callable suitable for DataLoader's worker_init_fn argument.
    """
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn

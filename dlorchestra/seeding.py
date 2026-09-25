"""Reproducibility: global and per-worker seeding."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42, cudnn_benchmark: bool = True) -> None:
    """Seed Python, NumPy, PyTorch (and MONAI, if importable).

    cudnn.benchmark auto-tunes convolution kernels for speed. Disable it
    when kernel compilation is unreliable for your hardware/input sizes
    (e.g. the ROCm MIOpen BatchNorm bug on large volumes).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        from monai.utils import set_determinism
        set_determinism(seed=seed)
    except ImportError:
        pass
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = False


def seed_worker(worker_id: int) -> None:
    """Per-DataLoader-worker seed for NumPy and stdlib random.

    PyTorch seeds its own per-worker RNG automatically, but NumPy and
    ``random`` (used internally by some transform libraries) need manual
    seeding.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

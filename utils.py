"""Reproducibility and filesystem helpers."""

import random
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch


def seed_everything(seed):
    pl.seed_everything(seed, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

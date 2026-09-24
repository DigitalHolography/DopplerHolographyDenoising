"""Small cross-stage helpers: JSON, hashing, deterministic seeds and devices."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


_MODULES = {
    "arterial_peaks": ".preparation.arterial_peaks",
    "dataset_workflow": ".preparation.dataset",
    "benchmark_splits": ".training.splits",
    "training_monitor": ".training.monitor",
    "noise2time_report": ".evaluation.report",
}


def load_sibling(name):
    """Resolve former sibling modules through the package layout.

    The public name is retained for old callers, while imports now use normal
    package semantics and therefore work after installation.
    """
    try:
        module = _MODULES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown workflow module: {name}") from exc
    return importlib.import_module(module, package="doppler_denoising")

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)



def get_device(name):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)




"""Backward-compatible public API.

Implementation lives in stage-specific packages. Existing code can continue to
import ``doppler_denoising.noise2time`` while new code imports focused modules.
"""
from dataclasses import asdict

import cv2

from .cli import main
from .common import get_device, load_sibling, seed_all, sha256, write_json
from .config import Config, SAMPLING_SCHEMA
from .evaluation.inference import denoise, export_denoised
from .evaluation.metrics import evaluate, evaluate_regional, load_evaluation_mask, summarize
from .preparation.core import (
    circle_mask,
    cycle_coordinates,
    load_frames,
    phases_from_peaks,
    prepare,
    prepare_folder,
    prepare_one,
    save_preparation_previews,
)
from .training.losses import loss_terms
from .training.model import ConvBlock, ConvLSTM, Noise2Time
from .training.runner import train
from .training.sampling import (
    PhaseDonor,
    Record,
    interpolate_donor,
    replacement,
    resolve_training_records,
    select_train_samples_for_epoch,
    split_samples,
)


if __name__ == "__main__":
    raise SystemExit(main())

"""Training-free centered temporal-average benchmark baseline."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ..common import sha256, write_json
from ..training.sampling import Record


SCHEMA = "doppler_denoising.analytic_baseline.v1"
METHOD = "temporal_average"
DEFAULT_WINDOW_SIZE = 11


def default_config(base=None, overrides=None):
    """Build the strict benchmark config for the temporal-average baseline."""
    source = dict(base or {})
    config = dict(
        schema=SCHEMA,
        method=METHOD,
        epochs=0,
        evaluation_prefix=int(source.get("evaluation_prefix", source.get("history", 9))),
        window_size=DEFAULT_WINDOW_SIZE,
        boundary_mode="reflect",
    )
    overrides = dict(overrides or {})
    unknown = set(overrides) - {"window_size"}
    if unknown:
        raise ValueError(f"Unknown temporal_average configuration fields: {sorted(unknown)}")
    config.update(overrides)
    validate_config(config)
    return config


def validate_config(config):
    expected = {"schema", "method", "epochs", "evaluation_prefix",
                "window_size", "boundary_mode"}
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError(f"Temporal-average config fields must be {sorted(expected)}")
    if config["schema"] != SCHEMA or config["method"] != METHOD or config["epochs"] != 0:
        raise ValueError("Invalid temporal-average schema, method, or analytic epoch count")
    window = config["window_size"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("temporal_average window_size must be a positive integer")
    prefix = config["evaluation_prefix"]
    if isinstance(prefix, bool) or not isinstance(prefix, int) or prefix < 0:
        raise ValueError("temporal_average evaluation_prefix must be a nonnegative integer")
    if config["boundary_mode"] != "reflect":
        raise ValueError("temporal_average boundary_mode must be reflect")


def window_offsets(window_size):
    """Return inclusive offsets; even windows follow the report's past-heavy convention."""
    left = int(window_size)//2
    right = int(window_size)-left-1
    return -left, right


def averaged_frames(frames, window_size):
    """Yield a fixed-size centered mean for every frame using reflected boundaries.

    Only a 2-D rolling sum is retained, so long videos are not copied into RAM.
    """
    if len(frames) < 2:
        raise ValueError("Temporal averaging requires at least two frames")
    first, last = window_offsets(window_size)
    indices = np.pad(np.arange(len(frames)), (-first, last), mode="reflect")
    rolling = np.zeros(frames.shape[1:], dtype=np.float64)
    for index in indices[:window_size]:
        rolling += np.asarray(frames[index], dtype=np.float64)
    for frame in range(len(frames)):
        yield (rolling/window_size).astype(np.float32)
        if frame + 1 < len(frames):
            rolling += np.asarray(frames[indices[frame+window_size]], dtype=np.float64)
            rolling -= np.asarray(frames[indices[frame]], dtype=np.float64)


def source_hashes():
    path = Path(__file__)
    return {path.name: sha256(path)}


def _atomic_checkpoint(checkpoint, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def ensure_run(folder, config):
    """Create the immutable epoch-zero checkpoint used by benchmark provenance."""
    validate_config(config)
    folder = Path(folder)
    files = source_hashes()
    expected = dict(schema=SCHEMA, method=METHOD, config=config, epoch=0,
                    method_files=files, analytic=True)
    paths = (folder/"best.pt", folder/"last.pt")
    rewrite = not all(path.exists() for path in paths)
    if not rewrite:
        for path in paths:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if any(saved.get(key) != value for key, value in expected.items()):
                rewrite = True
                break
    if rewrite:
        for path in paths:
            _atomic_checkpoint(expected, path)
    write_json(folder/"provenance.json", {
        "method": METHOD, "training": False, "window_size": config["window_size"],
        "boundary_mode": config["boundary_mode"], "method_files": files,
    })
    return paths[0]


def denoise_bundle(record_path, checkpoint_path, output_root, excluded_prefix=None,
                   write_avi=True):
    """Export one benchmark-compatible temporal-average result bundle."""
    checkpoint_path = Path(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved.get("schema") != SCHEMA or saved.get("method") != METHOD:
        raise ValueError("Checkpoint is not a temporal-average analytic checkpoint")
    config = saved["config"]
    validate_config(config)
    record = Record(record_path)
    if config["window_size"] > len(record.frames):
        raise ValueError(
            f"{record.name}: temporal-average window {config['window_size']} exceeds "
            f"the {len(record.frames)} available frames"
        )
    excluded = int(config["evaluation_prefix"] if excluded_prefix is None else excluded_prefix)
    if not 0 <= excluded < len(record.frames):
        raise ValueError("excluded_prefix falls outside recording")
    destination = Path(output_root).resolve()/record.name
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fps = float(record.metadata["fps"])
    if write_avi and (not np.isfinite(fps) or fps <= 0):
        raise ValueError("AVI output requires a finite positive acquisition FPS")
    height, width = record.frames.shape[1:]
    with tempfile.TemporaryDirectory(prefix=".temporal-average-", dir=destination.parent) as temporary:
        bundle = Path(temporary)/"record"
        bundle.mkdir()
        array = np.lib.format.open_memmap(
            bundle/"denoised.npy", mode="w+", dtype=np.float32,
            shape=record.frames.shape,
        )
        denoised_writer = original_writer = None
        try:
            if write_avi:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                denoised_writer = cv2.VideoWriter(
                    str(bundle/"denoised.avi"), fourcc, fps, (width, height), True
                )
                original_writer = cv2.VideoWriter(
                    str(bundle/"original.avi"), fourcc, fps, (width, height), True
                )
                if not denoised_writer.isOpened() or not original_writer.isOpened():
                    raise RuntimeError("Cannot initialize temporal-average AVI writers")
            iterator = averaged_frames(record.frames, config["window_size"])
            for frame, prediction in enumerate(tqdm(
                    iterator, total=len(record.frames), desc=f"Averaging {record.name}",
                    unit="frame", dynamic_ncols=True, mininterval=1.0)):
                if record.metadata.get("diaphragm_mask_applied", False):
                    prediction = np.where(record.roi, prediction, 0).astype(np.float32)
                if not np.isfinite(prediction).all():
                    raise FloatingPointError(f"Non-finite temporal average at frame {frame}")
                array[frame] = prediction
                if denoised_writer is not None:
                    view = np.rint(np.clip(prediction, 0, 1)*255).astype(np.uint8)
                    denoised_writer.write(cv2.cvtColor(view, cv2.COLOR_GRAY2BGR))
                    source = np.rint(np.clip(record.frames[frame], 0, 1)*255).astype(np.uint8)
                    original_writer.write(cv2.cvtColor(source, cv2.COLOR_GRAY2BGR))
        finally:
            if denoised_writer is not None:
                denoised_writer.release()
            if original_writer is not None:
                original_writer.release()
            array.flush()
            array._mmap.close()
        first, last = window_offsets(config["window_size"])
        write_json(bundle/"denoised.json", {
            "method": METHOD, "record": str(record.path),
            "record_sha256": sha256(record.path/"frames.npy"),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256(checkpoint_path), "epoch": 0,
            "copied_prefix": 0, "excluded_prefix": excluded,
            "exclusion_reason": "Common Noise2Time history window for comparable reports",
            "first_original_frame": record.metadata.get("first_original_frame", 0),
            "fps": fps, "intensities": "float32, unclipped",
            "intensity_scale": record.metadata.get("intensity_scale", 1.),
            "input_mode": record.metadata.get("input_mode", "legacy"),
            "diaphragm_mask_applied": bool(
                record.metadata.get("diaphragm_mask_applied", False)
            ),
            "window_size": config["window_size"],
            "window_offsets_inclusive": [first, last],
            "uses_future_frames": last > 0,
            "boundary_policy": "Reflect the temporal sequence to retain a fixed window at both ends",
            "method_sources": source_hashes(),
            "avi": str((destination/"denoised.avi").resolve()) if write_avi else None,
        })
        bundle.replace(destination)
    print(f"Saved {destination}", flush=True)
    return destination

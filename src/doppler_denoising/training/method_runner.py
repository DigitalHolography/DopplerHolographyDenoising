"""Train and infer published self-supervised denoising baselines.

This runner deliberately shares prepared inputs, split policy, monitoring,
checkpoint conventions and report metadata with the Noise2Time workflow.  The
method-specific modules own only their architecture, prediction and loss.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time

import cv2
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from ..common import get_device, seed_all, sha256, write_json
from ..methods import METHODS
from .sampling import resolve_training_records


SCHEMA = "doppler_denoising.method_baseline.v1"


def method_module(method):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    from importlib import import_module
    return import_module(f"doppler_denoising.methods.{method}")


def _base_values(base):
    if hasattr(base, "__dataclass_fields__"):
        from dataclasses import asdict
        base = asdict(base)
    return dict(base or {})


def default_config(method, base=None, overrides=None):
    """Resolve a strict method config while inheriting the benchmark budget."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    source = _base_values(base)
    learning_rates = {"udvd": 1e-4, "blind2unblind": 3e-4,
                      "neighbor2neighbor": 3e-4}
    patch_sizes = {"udvd": 128, "blind2unblind": 128,
                   "neighbor2neighbor": 256}
    config = dict(
        schema=SCHEMA, method=method,
        seed=int(source.get("seed", 2026)),
        epochs=int(source.get("epochs", 100)),
        samples_per_epoch=int(source.get("samples_per_epoch", 3000)),
        batch_size=int(source.get("batch_size", 2)),
        validation_samples=int(source.get("validation_samples", 35)),
        learning_rate=learning_rates[method],
        weight_decay=(1e-8 if method == "blind2unblind" else 0.),
        patience=int(source.get("patience", 10)),
        split_mode=source.get("split_mode", "mixed"),
        validation_records=list(source.get("validation_records", ())),
        validation_fraction=float(source.get("validation_fraction", .5)),
        patch_size=patch_sizes[method], inference_batch_size=1,
        use_amp=True, max_grad_norm=5.,
        evaluation_prefix=int(source.get("history", 9)),
    )
    if method == "udvd":
        config.update(temporal_frames=5)
    elif method == "blind2unblind":
        config.update(features=48, mask_width=4, mask_chunk_size=4,
                      lambda_blind=1., beta_schedule=dict(
                          schedule_epochs=config["epochs"], start_ratio=.4,
                          end_ratio=1., start_beta=2., end_beta=20.))
    else:
        config.update(features=48, increase_ratio=2.,
                      schedule_epochs=config["epochs"])
    overrides = dict(overrides or {})
    allowed = set(config)
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"Unknown {method} configuration fields: {sorted(unknown)}")
    old_epochs = config["epochs"]
    config.update(overrides)
    if "epochs" in overrides and overrides["epochs"] != old_epochs:
        if method == "neighbor2neighbor" and "schedule_epochs" not in overrides:
            config["schedule_epochs"] = config["epochs"]
        if method == "blind2unblind" and "beta_schedule" not in overrides:
            config["beta_schedule"]["schedule_epochs"] = config["epochs"]
    validate_config(method, config)
    return config


def validate_config(method, config):
    if method not in METHODS or config.get("method") != method or config.get("schema") != SCHEMA:
        raise ValueError("Method configuration identity is invalid")
    positive_ints = ("seed", "epochs", "samples_per_epoch", "batch_size",
                     "validation_samples", "patience", "patch_size",
                     "inference_batch_size", "evaluation_prefix")
    for key in positive_ints:
        if not isinstance(config.get(key), int) or config[key] < (0 if key == "evaluation_prefix" else 1):
            raise ValueError(f"{key} must be a positive integer")
    if config["split_mode"] not in ("mixed", "temporal", "record"):
        raise ValueError("split_mode must be mixed, temporal or record")
    if config["split_mode"] == "record" and not config["validation_records"]:
        raise ValueError("record split requires validation_records")
    if not 0 < float(config["validation_fraction"]) < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if float(config["learning_rate"]) <= 0 or float(config["weight_decay"]) < 0:
        raise ValueError("Invalid optimizer settings")
    if method == "udvd":
        if config.get("temporal_frames") != 5:
            raise ValueError("UDVD uses exactly five temporal frames")
        if config["patch_size"] % 4:
            raise ValueError("UDVD patch_size must be divisible by 4")
    else:
        if config["patch_size"] % 64:
            raise ValueError(f"{method} patch_size must be divisible by 64")
        if not isinstance(config.get("features"), int) or config["features"] < 1:
            raise ValueError("features must be a positive integer")
    if method == "neighbor2neighbor":
        if not isinstance(config.get("schedule_epochs"), int) or config["schedule_epochs"] < 1:
            raise ValueError("schedule_epochs must be positive")
        if float(config.get("increase_ratio", -1)) < 0:
            raise ValueError("increase_ratio must be nonnegative")
    if method == "blind2unblind":
        width, chunk = config.get("mask_width"), config.get("mask_chunk_size")
        if not isinstance(width, int) or width < 2 or not isinstance(chunk, int) or not 1 <= chunk <= width * width:
            raise ValueError("Invalid Blind2Unblind mask configuration")
        schedule = config.get("beta_schedule")
        needed = {"schedule_epochs", "start_ratio", "end_ratio", "start_beta", "end_beta"}
        if not isinstance(schedule, dict) or set(schedule) != needed:
            raise ValueError("Invalid Blind2Unblind beta schedule")
        if not isinstance(schedule["schedule_epochs"], int) or schedule["schedule_epochs"] < 1:
            raise ValueError("Invalid Blind2Unblind beta schedule")
        if not 0 <= schedule["start_ratio"] < schedule["end_ratio"] <= 1:
            raise ValueError("Invalid Blind2Unblind beta schedule")


def with_epochs(config, epochs):
    """Extend a run and its method schedule when the schedule tracked epochs."""
    result = json.loads(json.dumps(config)); old = int(result["epochs"])
    result["epochs"] = int(epochs)
    if result["method"] == "neighbor2neighbor" and result["schedule_epochs"] == old:
        result["schedule_epochs"] = int(epochs)
    if result["method"] == "blind2unblind" and result["beta_schedule"]["schedule_epochs"] == old:
        result["beta_schedule"]["schedule_epochs"] = int(epochs)
    validate_config(result["method"], result)
    return result
class PreparedRecord:
    """Only the common arrays required by the published baselines."""
    def __init__(self, directory):
        self.path = Path(directory).resolve(); self.name = self.path.name
        self.frames = np.load(self.path / "frames.npy", mmap_mode="r", allow_pickle=False)
        self.roi = np.load(self.path / "roi.npy", allow_pickle=False).astype(bool)
        valid = self.path / "valid_frames.npy"
        self.valid = (np.load(valid, allow_pickle=False).astype(bool)
                      if valid.exists() else np.ones(len(self.frames), dtype=bool))
        self.metadata = json.loads((self.path / "metadata.json").read_text(encoding="utf-8"))
        linked = self.metadata.get("dataset_measure")
        self.dataset_measure = Path(linked) if linked else None
        if self.dataset_measure is not None and not self.dataset_measure.is_dir():
            candidate = self.path.parent.parent.parent / Path(linked.replace("\\", "/")).parent.name / self.name
            if candidate.is_dir(): self.dataset_measure = candidate.resolve()
        if self.frames.ndim != 3 or self.frames.dtype != np.float32:
            raise ValueError(f"{self.name}: frames.npy must be float32 [T,H,W]")
        if self.roi.shape != self.frames.shape[1:] or self.valid.shape != (len(self.frames),):
            raise ValueError(f"{self.name}: prepared masks have incompatible shapes")
        if any(size % 32 for size in self.frames.shape[1:]):
            raise ValueError(f"{self.name}: image dimensions must be divisible by 32")
        if not np.isfinite(self.frames).all() or self.frames.min() < 0 or self.frames.max() > 1:
            raise ValueError(f"{self.name}: prepared intensities must be finite in [0,1]")
        if self.metadata.get("diaphragm_mask_applied") is not True:
            raise ValueError(f"{self.name}: benchmark requires diaphragm-masked preparation")


def _eligible(record, method, lo, hi):
    radius = 2 if method == "udvd" else 0
    result = []
    for frame in range(lo + radius, hi - radius):
        support = slice(frame - radius, frame + radius + 1)
        if record.valid[frame] and np.all(record.valid[support]):
            result.append(frame)
    return result


def _temporal_ranges(record, config):
    peaks = np.asarray(record.metadata.get("peaks", []), dtype=int)
    if len(peaks) < 4:
        raise ValueError(f"{record.name}: temporal split needs at least 4 peaks (3 complete cycles)")
    fps = float(record.metadata["fps"])
    guard = int(4 * max(.5, .02 * fps) + .5)
    ranges = {"train": (0, int(peaks[-2] - guard)),
              "valid": (int(peaks[-2] + guard), int(peaks[-1]))}
    if any(stop <= start for start, stop in ranges.values()):
        raise ValueError(f"{record.name}: temporal split is empty after its guard")
    return ranges


def split_samples(method, records, config):
    """Build deterministic mixed, per-cycle, or record-disjoint frame pools."""
    validate_config(method, config)
    rng = np.random.default_rng(config["seed"])
    ranges = {}
    if config["split_mode"] == "temporal":
        training, candidates = [], []
        for index, record in enumerate(records):
            record_ranges = _temporal_ranges(record, config)
            ranges[record.name] = record_ranges
            for stage, destination in (("train", training), ("valid", candidates)):
                lo, hi = record_ranges[stage]
                samples = [(index, frame) for frame in _eligible(record, method, lo, hi)]
                if not samples:
                    raise ValueError(f"{record.name}: no {stage} method samples after temporal separation")
                destination.extend(samples)
    else:
        pools = [[(index, frame) for frame in _eligible(record, method, 0, len(record.frames))]
                 for index, record in enumerate(records)]
        if any(not pool for pool in pools):
            raise ValueError("Every record needs at least one eligible method frame")
        validation_names = set(config["validation_records"])
        if config["split_mode"] == "record":
            unknown = validation_names - {record.name for record in records}
            if unknown:
                raise ValueError(f"Unknown validation records: {sorted(unknown)}")
            training = [sample for i, pool in enumerate(pools)
                        if records[i].name not in validation_names for sample in pool]
            candidates = [sample for i, pool in enumerate(pools)
                          if records[i].name in validation_names for sample in pool]
        else:
            reserved = [int(rng.integers(len(pool))) for pool in pools]
            training = [pool[position] for pool, position in zip(pools, reserved)]
            candidates = [sample for pool, position in zip(pools, reserved)
                          for j, sample in enumerate(pool) if j != position]
    if not training or len(candidates) < config["validation_samples"]:
        raise ValueError("Not enough independent method samples for the requested split")
    selected = set(rng.choice(len(candidates), config["validation_samples"], replace=False).tolist())
    validation = [sample for j, sample in enumerate(candidates) if j in selected]
    if config["split_mode"] == "mixed":
        training += [sample for j, sample in enumerate(candidates) if j not in selected]
    return training, validation, ranges


def _select_training(pool, count, rng):
    by_record = {}
    for position, (record, _) in enumerate(pool):
        by_record.setdefault(record, []).append(position)
    if len(by_record) > count:
        raise ValueError("samples_per_epoch must cover every training record")
    chosen = [int(rng.choice(positions)) for positions in by_record.values()]
    chosen.extend(rng.choice(len(pool), count - len(chosen), replace=True).tolist())
    rng.shuffle(chosen)
    return np.asarray(chosen, dtype=int)


def _sample(record, frame, method, config, rng, augment):
    patch = int(config["patch_size"]); height, width = record.roi.shape
    if patch > height or patch > width:
        raise ValueError(f"{record.name}: patch_size {patch} exceeds {height}x{width}")
    top = int(rng.integers(0, height - patch + 1)); left = int(rng.integers(0, width - patch + 1))
    if method == "udvd":
        images = np.array(record.frames[frame - 2:frame + 3, top:top + patch, left:left + patch], copy=True)
    else:
        images = np.array(record.frames[frame:frame + 1, top:top + patch, left:left + patch], copy=True)
    roi = np.array(record.roi[None, top:top + patch, left:left + patch], dtype=np.float32, copy=True)
    if augment:
        if rng.random() < .5: images = images[..., ::-1]; roi = roi[..., ::-1]
        if rng.random() < .5: images = images[..., ::-1, :]; roi = roi[..., ::-1, :]
        turns = int(rng.integers(4))
        images = np.rot90(images, turns, axes=(-2, -1)); roi = np.rot90(roi, turns, axes=(-2, -1))
        if method == "udvd" and rng.random() < .5: images = images[::-1]
    return np.ascontiguousarray(images), np.ascontiguousarray(roi)


def make_batch(records, pool, selections, method, config, rng, augment, device):
    samples = [_sample(records[pool[int(index)][0]], pool[int(index)][1], method,
                       config, rng, augment) for index in selections]
    images = torch.from_numpy(np.stack([sample[0] for sample in samples])).to(device)
    roi = torch.from_numpy(np.stack([sample[1] for sample in samples])).to(device)
    return images, roi


def run_stage(model, implementation, records, pool, config, epoch, training,
              optimizer, numpy_rng, sampling_rng, scaler, device, no_progress=False):
    if training:
        choices = _select_training(pool, config["samples_per_epoch"], numpy_rng)
        sample_rng = numpy_rng
    else:
        choices = np.arange(len(pool)); sample_rng = np.random.default_rng(config["seed"] + 10000)
        sampling_rng = torch.Generator(device="cpu").manual_seed(config["seed"] + 10000)
    model.train(training); totals = {}; seen = 0
    batch_size = config["batch_size"] if training else 1
    progress = tqdm(range(0, len(choices), batch_size),
                    desc=f"Epoch {epoch}/{config['epochs']} {'train' if training else 'valid'}",
                    unit="batch", file=sys.stdout, dynamic_ncols=True, mininterval=1.,
                    disable=no_progress)
    amp = bool(config["use_amp"] and device.type == "cuda")
    for start in progress:
        selected = choices[start:start + batch_size]
        images, roi = make_batch(records, pool, selected, config["method"], config,
                                 sample_rng, training, device)
        with torch.set_grad_enabled(training), torch.amp.autocast("cuda", enabled=amp):
            parts = implementation.loss(model, images, roi, config, epoch, sampling_rng)
            loss = parts["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite method loss")
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]), error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update()
        count = len(selected); seen += count
        for key, value in parts.items():
            scalar = float(value.detach()) if torch.is_tensor(value) else float(value)
            totals[key] = totals.get(key, 0.) + scalar * count
        progress.set_postfix(loss=f"{totals['total']/seen:.6g}", samples=f"{seen}/{len(choices)}", refresh=False)
    return {key: value / seen for key, value in totals.items()}


def _manifest(records):
    names = ("frames.npy", "roi.npy", "valid_frames.npy", "metadata.json")
    return [{"name": record.name,
             "hashes": {name: sha256(record.path / name) for name in names if (record.path / name).exists()}}
            for record in records]


def method_source_hashes(method):
    # Import directly so test doubles for method_module cannot alter provenance.
    from importlib import import_module
    implementation = import_module(f"doppler_denoising.methods.{method}")
    files = [Path(__file__), Path(implementation.__file__)]
    if method == "udvd":
        from ..methods import udvd_arch
        files.append(Path(udvd_arch.__file__))
    elif method in ("blind2unblind", "neighbor2neighbor"):
        from ..methods import neighbor_arch
        files.append(Path(neighbor_arch.__file__))
    return {path.name: sha256(path) for path in files}


def train(args):
    from .. import noise2time as api
    from . import monitor as support
    output = Path(args.output); resume = bool(getattr(args, "resume", False))
    saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True) if resume else None
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    elif saved:
        config = saved["config"]
    else:
        raise ValueError("Method training requires --config")
    if getattr(args, "epochs", None) is not None:
        config = with_epochs(config, args.epochs)
    validate_config(args.method, config)
    if saved:
        old = dict(saved["config"]); new = dict(config)
        for values in (old, new):
            values.pop("epochs", None)
            if args.method == "neighbor2neighbor": values.pop("schedule_epochs", None)
            if args.method == "blind2unblind":
                values["beta_schedule"] = dict(values["beta_schedule"]); values["beta_schedule"].pop("schedule_epochs", None)
        if old != new:
            raise ValueError("Resume must keep the method configuration; only total epochs may change")
    seed_all(config["seed"]); device = get_device(args.device)
    print(f"Training {args.method} on {device}", flush=True)
    records = [PreparedRecord(path) for path in resolve_training_records(args.records)]
    if len({record.name for record in records}) != len(records):
        raise ValueError("Record folder names must be unique")
    if len({record.frames.shape[1:] for record in records}) != 1:
        raise ValueError("All method records must have identical image dimensions")
    for record in records:
        if config["patch_size"] > min(record.roi.shape):
            raise ValueError(f"{record.name}: patch_size exceeds the prepared dimensions")
    training, validation, ranges = split_samples(args.method, records, config)
    manifest = _manifest(records)
    split = {"training": training, "validation": validation, "ranges": ranges,
             "records": [record.name for record in records], "method": args.method}
    # JSON normalization makes tuple/list representation stable across resume.
    split = json.loads(json.dumps(split))
    if resume:
        if json.loads((output / "provenance.json").read_text())["records"] != manifest:
            raise ValueError("Prepared records changed since method training")
        if json.loads((output / "split.json").read_text()) != split:
            raise ValueError("Method training/validation split changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "split.json", split)
        write_json(output / "provenance.json", {
            "records": manifest, "method": args.method,
            "method_files": method_source_hashes(args.method),
            "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "device": str(device)})
    write_json(output / "config.json", config)
    implementation = method_module(args.method)
    model = implementation.build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]),
                                 weight_decay=float(config["weight_decay"]))
    amp = bool(config["use_amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    numpy_rng = np.random.default_rng(config["seed"])
    sampling_rng = torch.Generator(device="cpu").manual_seed(config["seed"])
    best, stale, first_epoch, rows = float("inf"), 0, 1, []
    if saved:
        if saved.get("schema") != SCHEMA or saved.get("method") != args.method:
            raise ValueError("Checkpoint is not from the requested method")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        if saved.get("scaler"): scaler.load_state_dict(saved["scaler"])
        numpy_rng.bit_generator.state = saved["numpy_rng"]
        sampling_rng.set_state(saved["sampling_rng"])
        torch.set_rng_state(saved["torch_rng"]); random.setstate(saved["python_rng"])
        if device.type == "cuda" and saved.get("cuda_rng"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        first_epoch = int(saved["epoch"]) + 1; best = float(saved["best_validation"]); stale = int(saved["stale"])
        history = output / "metrics.jsonl"
        if history.exists():
            rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
            rows = [row for row in rows if row["epoch"] <= saved["epoch"]]
        print(f"Resuming after epoch {saved['epoch']}; target {config['epochs']}", flush=True)
    monitors = {}
    if not getattr(args, "no_epoch_metrics", False):
        for record in records[:1]:
            if record.metadata.get("dataset_measure"):
                monitors[record.name] = support.Monitor(record, config["evaluation_prefix"], api,
                                                         getattr(args, "background_dilation_radius", 2))
    for epoch in range(first_epoch, config["epochs"] + 1):
        if stale >= config["patience"]:
            print("Saved run already reached early stopping.", flush=True); break
        started = time.monotonic(); summaries = {}
        summaries["train"] = run_stage(model, implementation, records, training, config, epoch,
                                         True, optimizer, numpy_rng, sampling_rng, scaler, device,
                                         getattr(args, "no_progress", False))
        summaries["valid"] = run_stage(model, implementation, records, validation, config, epoch,
                                         False, optimizer, numpy_rng, sampling_rng, scaler, device,
                                         getattr(args, "no_progress", False))
        value = summaries["valid"]["total"]; improved = value < best
        best, stale = (value, 0) if improved else (best, stale + 1)
        checkpoint = {
            "schema": SCHEMA, "method": args.method, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "config": config, "epoch": epoch, "best_validation": best,
            "stale": stale, "metrics": summaries,
            "numpy_rng": numpy_rng.bit_generator.state,
            "sampling_rng": sampling_rng.get_state(), "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "method_files": method_source_hashes(args.method),
        }
        if improved: support.atomic_checkpoint(torch, checkpoint, output / "best.pt")
        support.atomic_checkpoint(torch, checkpoint, output / "last.pt")
        checkpoint_folder = output / "checkpoints"; checkpoint_folder.mkdir(exist_ok=True)
        compact = {key: checkpoint[key] for key in ("schema", "method", "model", "config", "epoch", "metrics", "method_files")}
        support.atomic_checkpoint(torch, compact, checkpoint_folder / f"epoch_{epoch:03d}.pt")
        rows.append({"epoch": epoch, **summaries}); support.save_history(output, rows)
        if monitors or not getattr(args, "no_epoch_previews", False):
            record = records[0]; monitor = monitors.get(record.name)
            if monitor: monitor.reset()
            preview = None if getattr(args, "no_epoch_previews", False) else output / "previews" / f"epoch_{epoch:03d}" / f"{record.name}.avi"
            export_denoised(model, record, config, device, avi_path=preview,
                            frame_callback=monitor.update if monitor else None)
            if monitor: rows[-1]["diagnostics"] = {record.name: monitor.result()}
            support.save_history(output, rows)
            if preview:
                write_json(preview.with_suffix(".json"), {
                    "method": args.method, "epoch": epoch, "record": str(record.path),
                    "record_sha256": sha256(record.path / "frames.npy"),
                    "copied_prefix": 0, "excluded_prefix": config["evaluation_prefix"],
                    "intensities": "MJPG preview; clip [0,1], round to uint8"})
        print(f"Epoch {epoch}/{config['epochs']}: train={summaries['train']['total']:.6g}, "
              f"valid={value:.6g}, best={best:.6g}, no improvement={stale}/{config['patience']}, "
              f"elapsed={time.monotonic()-started:.1f}s", flush=True)
        if stale >= config["patience"]:
            print(f"Early stopping after {stale} epochs without improvement.", flush=True); break
    print(f"Training finished. Best checkpoint: {output / 'best.pt'}", flush=True)


def _predict_batches(model, record, config, device):
    implementation = method_module(config["method"]); batch_size = config["inference_batch_size"]
    with torch.inference_mode():
        for start in range(0, len(record.frames), batch_size):
            stop = min(start + batch_size, len(record.frames)); samples = []
            for frame in range(start, stop):
                if config["method"] == "udvd":
                    indices = np.clip(np.arange(frame - 2, frame + 3), 0, len(record.frames) - 1)
                    samples.append(np.array(record.frames[indices], copy=True))
                else:
                    samples.append(np.array(record.frames[frame:frame + 1], copy=True))
            images = torch.from_numpy(np.stack(samples)).to(device)
            predictions = implementation.predict(model, images, config).cpu().numpy()[:, 0]
            for frame, prediction in zip(range(start, stop), predictions):
                yield frame, np.where(record.roi, prediction, 0).astype(np.float32, copy=False)


def export_denoised(model, record, config, device, npy_path=None, avi_path=None,
                    original_avi_path=None, frame_callback=None):
    paths = [Path(path) for path in (npy_path, avi_path, original_avi_path) if path is not None]
    if not paths and frame_callback is None: raise ValueError("At least one output is required")
    if any(path.exists() for path in paths): raise FileExistsError(next(path for path in paths if path.exists()))
    parent = paths[0].parent.resolve() if paths else Path(tempfile.gettempdir()).resolve()
    parent.mkdir(parents=True, exist_ok=True); previous = model.training; model.eval()
    fps = float(record.metadata["fps"]); height, width = record.roi.shape
    try:
        with tempfile.TemporaryDirectory(prefix=".method-denoise-", dir=parent) as temporary:
            staging = Path(temporary); array = writer = original_writer = None
            try:
                if npy_path is not None:
                    array = np.lib.format.open_memmap(staging / "denoised.npy", mode="w+", dtype=np.float32, shape=record.frames.shape)
                if avi_path is not None:
                    writer = cv2.VideoWriter(str(staging / "denoised.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height), True)
                    if not writer.isOpened(): raise RuntimeError("Cannot initialize denoised AVI")
                if original_avi_path is not None:
                    original_writer = cv2.VideoWriter(str(staging / "original.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height), True)
                    if not original_writer.isOpened(): raise RuntimeError("Cannot initialize original AVI")
                iterator = tqdm(_predict_batches(model, record, config, device), total=len(record.frames),
                                desc=f"Denoising {record.name}", unit="frame", file=sys.stdout,
                                dynamic_ncols=True, mininterval=1.)
                for frame, prediction in iterator:
                    if not np.isfinite(prediction).all(): raise FloatingPointError(f"Non-finite output at frame {frame}")
                    if frame_callback: frame_callback(frame, prediction)
                    if array is not None: array[frame] = prediction
                    if writer is not None:
                        view = np.rint(np.clip(prediction, 0, 1) * 255).astype(np.uint8)
                        writer.write(cv2.cvtColor(view, cv2.COLOR_GRAY2BGR))
                    if original_writer is not None:
                        view = np.rint(np.clip(record.frames[frame], 0, 1) * 255).astype(np.uint8)
                        original_writer.write(cv2.cvtColor(view, cv2.COLOR_GRAY2BGR))
            finally:
                if writer is not None: writer.release()
                if original_writer is not None: original_writer.release()
                if array is not None: array.flush(); array._mmap.close()
            if npy_path is not None: (staging / "denoised.npy").replace(npy_path)
            if avi_path is not None: (staging / "denoised.avi").replace(avi_path)
            if original_avi_path is not None: (staging / "original.avi").replace(original_avi_path)
    finally:
        model.train(previous)


def denoise(args):
    checkpoint_path = Path(args.checkpoint); saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    method = args.method or saved.get("method")
    if saved.get("schema") != SCHEMA or method != saved.get("method"):
        raise ValueError("Checkpoint method does not match the request")
    config = saved["config"]; validate_config(method, config)
    record = PreparedRecord(args.record); device = get_device(args.device)
    model = method_module(method).build_model(config).to(device); model.load_state_dict(saved["model"])
    destination = Path(args.output).resolve() / record.name
    if destination.exists(): raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    requested_prefix = getattr(args, "excluded_prefix", None)
    excluded = int(config["evaluation_prefix"] if requested_prefix is None else requested_prefix)
    if not 0 <= excluded < len(record.frames): raise ValueError("excluded_prefix falls outside recording")
    with tempfile.TemporaryDirectory(prefix=".method-record-", dir=destination.parent) as temporary:
        bundle = Path(temporary) / "record"
        npy = bundle / "denoised.npy"
        avi = None if getattr(args, "npy_only", False) else bundle / "denoised.avi"
        original = None if getattr(args, "npy_only", False) else bundle / "original.avi"
        export_denoised(model, record, config, device, npy_path=npy,
                        avi_path=avi, original_avi_path=original)
        write_json(bundle / "denoised.json", {
            "method": method, "record": str(record.path),
            "record_sha256": sha256(record.path / "frames.npy"),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256(checkpoint_path), "epoch": saved["epoch"],
            "copied_prefix": 0, "excluded_prefix": excluded,
            "exclusion_reason": "Common Noise2Time history window for comparable reports",
            "first_original_frame": record.metadata.get("first_original_frame", 0),
            "fps": record.metadata["fps"], "intensities": "float32, unclipped",
            "intensity_scale": record.metadata.get("intensity_scale", 1.),
            "input_mode": record.metadata.get("input_mode", "legacy"),
            "diaphragm_mask_applied": True,
            "method_sources": method_source_hashes(method),
            "avi": str(avi.resolve()) if avi else None,
        })
        bundle.replace(destination)
    print(f"Saved {destination}", flush=True)


def inspect(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate_config(args.method, config)
    records = [PreparedRecord(path) for path in resolve_training_records(args.records)]
    train_pool, valid_pool, ranges = split_samples(args.method, records, config)
    print(json.dumps({"method": args.method, "training": len(train_pool),
                      "validation": len(valid_pool), "ranges": ranges}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "inspect"):
        command = commands.add_parser(name); command.add_argument("--method", choices=METHODS, required=True)
        command.add_argument("--records", nargs="+", required=True); command.add_argument("--config", required=True)
        if name == "train":
            command.add_argument("--output", required=True); command.add_argument("--device", default="auto")
            command.add_argument("--resume", action="store_true"); command.add_argument("--epochs", type=int)
            command.add_argument("--no-epoch-metrics", action="store_true"); command.add_argument("--no-epoch-previews", action="store_true")
            command.add_argument("--no-progress", action="store_true"); command.add_argument("--background-dilation-radius", type=int, default=2)
    command = commands.add_parser("denoise"); command.add_argument("--method", choices=METHODS)
    command.add_argument("--record", required=True); command.add_argument("--checkpoint", required=True)
    command.add_argument("--output", required=True); command.add_argument("--device", default="auto")
    command.add_argument("--excluded-prefix", type=int); command.add_argument("--npy-only", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "train": train(args)
    elif args.command == "denoise": denoise(args)
    else: inspect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


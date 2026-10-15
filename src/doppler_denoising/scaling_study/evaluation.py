"""Compact evaluation and repeatability statistics for scaling experiments."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ..common import get_device, seed_all, sha256, write_json
from ..config import Config
from ..evaluation.inference import export_denoised
from ..evaluation.metrics import load_evaluation_mask
from ..evaluation.report import derive_background, off_harmonic_noise, waveform_metrics
from ..training.model import Noise2Time
from ..training.sampling import Record


def _mask(path, shape):
    result = load_evaluation_mask(path, shape)
    if not result.any():
        raise ValueError(f"Empty mask: {path}")
    return result


def score_measure(checkpoint, prepared, row, output, device="auto", dilation_radius=2):
    """Denoise and score one measure; small-vessel metrics are optional."""
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    record = Record(prepared)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = Config(**saved["config"]); cfg.validate(); seed_all(cfg.seed)
    device = get_device(device)
    model = Noise2Time(cfg.base_channels, cfg.convlstm, cfg.effective_model()).to(device)
    model.load_state_dict(saved["model"])
    denoised_path = output / "denoised.npy"
    export_denoised(model, record, cfg, device, npy_path=denoised_path,
                    avi_path=output / "denoised.avi", original_avi_path=output / "original.avi")
    restored = np.load(denoised_path, mmap_mode="r")
    raw = {"retinal_artery": _mask(row["artery_mask"], record.roi.shape),
           "retinal_vein": _mask(row["vein_mask"], record.roi.shape),
           "choroidal": _mask(row["choroidal_mask"], record.roi.shape)}
    if row.get("small_vessels_mask"):
        raw["small_vessels"] = _mask(row["small_vessels_mask"], record.roi.shape)
    # Remove primary overlap symmetrically. Small-vessel masks are an optional,
    # independent size-based region and deliberately retain overlap.
    primary = ("retinal_artery", "retinal_vein", "choroidal")
    masks = {}
    for name in primary:
        other = np.logical_or.reduce([raw[x] for x in primary if x != name])
        masks[name] = raw[name] & ~other & record.roi
    if "small_vessels" in raw:
        masks["small_vessels"] = raw["small_vessels"] & record.roi
    background_inputs = dict(raw)
    if "small_vessels" not in background_inputs:
        background_inputs["small_vessels"] = np.zeros(record.roi.shape, bool)
    masks["background"] = derive_background(background_inputs, record.roi, dilation_radius)
    if any(not mask.any() for mask in masks.values()):
        raise ValueError(f"A cleaned evaluation mask is empty for {row['measure']}")
    first = cfg.inference_prefix()
    original, denoised = np.asarray(record.frames[first:]), np.asarray(restored[first:])
    fps = float(record.metadata["fps"])
    peaks = np.asarray(record.metadata.get("peaks", []), float)
    intervals = np.diff(peaks); intervals = intervals[intervals > 0]
    if not len(intervals):
        raise ValueError("Detector peaks are required for cardiac-frequency metrics")
    f0 = float(fps / np.median(intervals))
    metrics = {"schema": "doppler-denoising.scaling-evaluation.v1", "measure": row["measure"],
               "patient": row["patient"], "eye": row["eye"], "repeat_role": row["repeat_role"],
               "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha256(checkpoint),
               "frames_scored": len(original), "cardiac_frequency_hz": f0,
               "small_vessels_available": "small_vessels" in masks, "regions": {}}
    for name, mask in masks.items():
        before = original[:, mask].mean(axis=1, dtype=np.float64)
        after = denoised[:, mask].mean(axis=1, dtype=np.float64)
        values = waveform_metrics(before, after, fps, f0, max_lag_seconds=0.1)
        values.update(off_harmonic_noise(before, after, fps, f0))
        values.update(pixels=int(mask.sum()), mean_original=float(before.mean()),
                      mean_denoised=float(after.mean()))
        metrics["regions"][name] = values
    before_std = original[:, masks["background"]].std(axis=0, dtype=np.float64).mean()
    after_std = denoised[:, masks["background"]].std(axis=0, dtype=np.float64).mean()
    metrics["background_std_original"] = float(before_std)
    metrics["background_std_denoised"] = float(after_std)
    metrics["NRR"] = float(1 - after_std / before_std) if before_std > 1e-12 else None
    write_json(output / "metrics.json", metrics)
    flattened = flatten_metrics(metrics)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flattened)); writer.writeheader(); writer.writerow(flattened)
    return metrics


def flatten_metrics(result):
    row = {key: result[key] for key in ("measure", "patient", "eye", "repeat_role",
                                        "frames_scored", "cardiac_frequency_hz", "NRR")}
    for region, values in result["regions"].items():
        for key in ("mean_original", "mean_denoised", "temporal_correlation", "amplitude_original",
                    "amplitude_denoised", "amplitude_ratio", "residual_pulsatility_ratio",
                    "off_harmonic_noise_reduction_db"):
            if key in values:
                row[f"{region}_{key}"] = values[key]
    return row


def icc_2_1(matrix):
    """Two-way random-effects, absolute-agreement single-measure ICC(2,1)."""
    values = np.asarray(matrix, float)
    n, k = values.shape
    if n < 2 or k < 2:
        return None
    grand = values.mean(); row_means = values.mean(1); column_means = values.mean(0)
    ms_rows = k * ((row_means - grand) ** 2).sum() / (n - 1)
    ms_columns = n * ((column_means - grand) ** 2).sum() / (k - 1)
    residual = values - row_means[:, None] - column_means[None, :] + grand
    ms_error = (residual ** 2).sum() / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if abs(denominator) > 1e-15 else None


def repeatability(rows, metric):
    """Summarize B/C repeat agreement for patient-eye subjects."""
    groups = {}
    for row in rows:
        if row.get("repeat_role") in ("B", "C") and row.get(metric) is not None:
            groups.setdefault((row["patient"], row["eye"]), {})[row["repeat_role"]] = float(row[metric])
    pairs = np.asarray([[x["B"], x["C"]] for x in groups.values() if set(x) >= {"B", "C"}], float)
    if not len(pairs):
        return {"n_subject_eyes": 0, "icc_2_1": None, "within_subject_cv": None,
                "bland_altman_bias": None, "bland_altman_lower": None, "bland_altman_upper": None}
    differences = pairs[:, 1] - pairs[:, 0]
    within_sd = np.sqrt(np.mean(np.var(pairs, axis=1, ddof=1)))
    mean = float(pairs.mean())
    bias = float(differences.mean()); sd = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    return {"n_subject_eyes": len(pairs), "icc_2_1": icc_2_1(pairs),
            "within_subject_cv": float(within_sd / abs(mean)) if abs(mean) > 1e-15 else None,
            "bland_altman_bias": bias, "bland_altman_lower": bias - 1.96 * sd,
            "bland_altman_upper": bias + 1.96 * sd}

"""Legacy metrics and the compact regional evaluation report entry point."""
import json
from pathlib import Path
import tempfile
import warnings

import cv2
import numpy as np

from ..common import sha256, write_json
from ..training.sampling import Record

def load_evaluation_mask(path, expected_shape):
    """Load a binary NPY or PNG mask without changing polarity."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
        if values.ndim != 2 or values.dtype.kind not in "buif" or not np.isfinite(values).all():
            raise ValueError(f"Mask must be a finite numeric 2D array: {path}")
        mask = values != 0
    elif path.suffix.lower() == ".png":
        # imdecode supports Unicode paths on Windows via NumPy file reading.
        encoded = np.fromfile(path, dtype=np.uint8)
        values = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) if encoded.size else None
        if values is None:
            raise ValueError(f"Cannot decode PNG mask: {path}")
        if values.ndim == 2:
            mask = values != 0
        elif values.ndim == 3 and values.shape[2] in (3, 4):
            mask = np.any(values[..., :3] != 0, axis=2)
            if values.shape[2] == 4:
                mask &= values[..., 3] != 0  # Ignore fully transparent pixels.
        else:
            raise ValueError(f"Unsupported PNG mask dimensions: {path}")
    else:
        raise ValueError(f"Mask must be .npy or .png: {path}")
    if mask.shape != tuple(expected_shape):
        mask = cv2.resize(mask.astype(np.uint8), tuple(reversed(expected_shape)), interpolation=cv2.INTER_NEAREST).astype(bool)
        warnings.warn(f"Mask {path} resized to match ROI shape {expected_shape}")
    return mask


def evaluate_regional(args, record, restored, metadata):
    from . import report
    if args.vessel_mask:
        raise ValueError("Use either --vessel-mask or the three regional vessel masks")
    paths = dict(retinal_artery=args.retinal_artery_mask, retinal_vein=args.retinal_vein_mask,
                 choroidal=args.choroidal_masks)
    if not all(paths.values()):
        raise ValueError("Regional evaluation requires retinal artery, retinal vein and choroidal masks")
    if args.background_mask or args.background_masks:
        raise ValueError("Regional background is automatic; omit --background-mask/--background-masks")
    raw, provenance = {}, {}
    for name, sources in paths.items():
        raw[name] = np.zeros(record.roi.shape, bool)
        provenance[name] = []
        for source in sources:
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter("always")
                raw[name] |= load_evaluation_mask(source, record.roi.shape)
            messages = [str(notice.message) for notice in notices]
            for message in messages:
                warnings.warn(message)
            provenance[name].append(dict(path=str(Path(source).resolve()), sha256=sha256(source), warnings=messages))
    raw["background"] = report.derive_background(raw, record.roi, args.background_dilation_radius)
    provenance["background"] = dict(
        method="ROI & ~(dilate(original retinal artery | original retinal vein) | original choroidal)",
        retinal_dilation_radius_pixels=args.background_dilation_radius,
        kernel="Euclidean disk", choroidal_dilated=False,
        source_masks="Original input unions before overlap removal; dilation before ROI clipping")
    first = metadata.get("copied_prefix")
    if not isinstance(first, int) or not 0 <= first < len(restored):
        raise ValueError("Invalid copied_prefix in denoised metadata")
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata, denoised_sha256=sha256(args.denoised),
                    denoised_path=str(Path(args.denoised).resolve()))
    with tempfile.TemporaryDirectory(prefix=".evaluation-", dir=destination.parent) as staging:
        output = Path(staging)/"report"
        report.build_report(record, restored, metadata, raw, provenance, args, output)
        output.rename(destination)
    print(f"Saved evaluation report: {destination/'report.html'}")


def evaluate(args):
    record = Record(args.record)
    restored = np.load(args.denoised, mmap_mode="r", allow_pickle=False)
    metadata = json.loads(Path(args.denoised).with_suffix(".json").read_text(encoding="utf-8"))
    if metadata["record_sha256"] != sha256(record.path/"frames.npy"):
        raise ValueError("Output does not correspond to this source array")
    if restored.shape != record.frames.shape or not np.isfinite(restored).all():
        raise ValueError("Output dimensions/values invalid")
    if any(getattr(args, key, None) for key in
           ("retinal_artery_mask", "retinal_vein_mask", "choroidal_masks", "background_masks")):
        return evaluate_regional(args, record, restored, metadata)
    if not args.vessel_mask or not args.background_mask:
        raise ValueError("Supply the regional masks, or legacy --vessel-mask and --background-mask")
    vessel = load_evaluation_mask(args.vessel_mask, record.roi.shape)
    background = load_evaluation_mask(args.background_mask, record.roi.shape)
    if (not vessel.any()):
        raise ValueError("Vessel mask must be non-empty")
    if (not background.any()):
        raise ValueError("Background mask must be non-empty")
    if (np.any(vessel & background)):
        vessel &= ~(vessel & background)
        background &= ~(vessel & background)
        warnings.warn("Vessel and background masks must be non-overlapping")
    print(f"Vessel mask: {vessel.sum()} pixels; background mask: {background.sum()} pixels")
    print(f"ROI mask: {record.roi.sum()} pixels; vessel & background overlap: {(vessel & background).sum()} pixels")
    warnings.warn("Vessel/background masks must be non-empty, non-overlapping, and match ROI shape")

    vessel &= record.roi
    background &= record.roi

    first = metadata["copied_prefix"]
    original = record.frames[first:]
    denoised = restored[first:]
    fps = record.metadata["fps"]
    if len(original) < 4 or not 0 < args.min_hz < args.max_hz < fps/2:
        raise ValueError("Need >=4 scored frames and 0 < min_hz < max_hz < Nyquist")
    before, after = [x[:,vessel].mean(axis=1, dtype=np.float64) for x in (original,denoised)]
    freq = np.fft.rfftfreq(len(before), d=1/fps)
    candidates = np.flatnonzero((freq >= args.min_hz) & (freq <= args.max_hz))
    if not len(candidates):
        raise ValueError("Recording too short for requested frequency band")
    detected_intervals = np.diff(np.asarray(record.metadata.get("peaks", []), dtype=float))
    detected_intervals = detected_intervals[np.isfinite(detected_intervals) & (detected_intervals > 0)]
    if len(detected_intervals) >= 2:
        detector_f0 = float(fps / np.median(detected_intervals))
        if not args.min_hz <= detector_f0 <= args.max_hz:
            raise ValueError(f"Detector-derived cardiac frequency {detector_f0:.5g} Hz falls outside [{args.min_hz}, {args.max_hz}] Hz")
        f0 = detector_f0
        frequency_source = "Median interval of arterial detector peaks"
    else:
        k = candidates[np.argmax(np.abs(np.fft.rfft(before-before.mean()))[candidates])]
        f0 = float(freq[k])
        frequency_source = "Largest original vessel FFT component (legacy fallback; no peak intervals in metadata)"
    time = np.arange(len(before))/fps
    design = np.column_stack([np.ones(len(time)), np.cos(2*np.pi*f0*time), np.sin(2*np.pi*f0*time)])
    amplitudes = [float(np.linalg.norm(np.linalg.lstsq(design, curve, rcond=None)[0][1:])) for curve in (before,after)]
    bg_std = [float(x[:,background].std(axis=0, ddof=0, dtype=np.float64).mean()) for x in (original,denoised)]
    correlation = float(np.corrcoef(before,after)[0,1]) if min(before.std(),after.std()) > 1e-12 else None
    result = dict(schema="noise2time.legacy.v2",metric_protocol="noise2time_metric_audit_v1",
                  record=record.name, frames_scored=len(before), excluded_prefix=first,
                  background_std_original=bg_std[0], background_std_denoised=bg_std[1],
                  NRR=1-bg_std[1]/bg_std[0] if bg_std[0] > 1e-12 else None,
                  temporal_correlation=correlation, cardiac_frequency_hz=f0,
                  frequency_source=frequency_source,
                  frequency_diagnostics=dict(peak_intervals_frames=detected_intervals.tolist(),
                                             median_period_frames=float(np.median(detected_intervals)) if len(detected_intervals) else None,
                                             fft_resolution_hz=float(fps/len(before))),
                  amplitude_original=amplitudes[0], amplitude_denoised=amplitudes[1],
                  amplitude_ratio=amplitudes[1]/amplitudes[0] if amplitudes[0] > 1e-12 else None,
                  vessel_mean_original=float(before.mean()), vessel_mean_denoised=float(after.mean()),
                  vessel_mask_sha256=sha256(args.vessel_mask), background_mask_sha256=sha256(args.background_mask),
                  mask_interpretation="Nonzero pixels selected; for PNG any nonzero color channel with nonzero alpha when present",
                  caveat="No clean reference: these are fluctuation and waveform diagnostics, not accuracy scores")
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    write_json(args.output,result)
    print(json.dumps(result,indent=2))


def summarize(args):
    rows = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.metrics]
    if len({row["record"] for row in rows}) != len(rows):
        raise ValueError("Duplicate record names in summary")
    keys = ("background_std_original", "background_std_denoised", "NRR",
            "temporal_correlation", "amplitude_ratio")
    summary = {}
    for key in keys:
        values = [row[key] for row in rows if row[key] is not None]
        if not all(np.isfinite(values)):
            raise ValueError(f"Non-finite metric: {key}")
        summary[key] = dict(n=len(values), missing=len(rows)-len(values),
                            mean=float(np.mean(values)) if values else None,
                            sample_sd=float(np.std(values, ddof=1)) if len(values)>1 else None)
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    write_json(args.output, dict(record_count=len(rows), records=rows, summary=summary))
    print(json.dumps(summary,indent=2))




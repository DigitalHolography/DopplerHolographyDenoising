"""Prepare AVI, NPY and linked-measurement inputs for training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import warnings

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm

from ..common import load_sibling, sha256, write_json

def load_frames(path):
    """Return normalized float32 T,H,W frames and video/container FPS if available."""
    path = Path(path)
    fps = None
    if path.suffix.lower() == ".npy":
        frames = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot read {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        buffer = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        finally:
            cap.release()
        if not buffer:
            raise ValueError(f"No frames in {path}")
        frames = np.stack(buffer)
    if frames.ndim != 3 or min(frames.shape) < 1:
        raise ValueError(f"Expected T,H,W: {path}")
    if frames.dtype == np.uint8:
        frames = frames.astype(np.float32) / 255.0
    elif np.issubdtype(frames.dtype, np.floating):
        frames = np.asarray(frames, dtype=np.float32)
    else:
        raise ValueError("Use uint8 or floating-point arrays in [0,1]; no automatic rescaling")
    if not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
        raise ValueError(f"Input intensities must be finite and in [0,1]: {path}")
    return frames, fps


def circle_mask(height, width, cx, cy, radius):
    mask = np.zeros((height, width), np.uint8)
    cv2.circle(mask, (cx, cy), radius, 1, thickness=-1)
    return mask.astype(bool)


def phases_from_peaks(count, peaks):
    """Legacy discrete phase: elapsed frames since peak, not fractional phase."""
    phase = np.full(count, -1, dtype=np.int64)
    for start, end in zip(peaks, list(peaks[1:]) + [count]):
        phase[start:end] = np.arange(end - start)
    return phase


def cycle_coordinates(count, peaks):
    """Describe frames belonging to complete peak-to-peak cardiac cycles.

    ``fraction`` is in [0, 1), so it remains comparable when two cycles have
    different frame counts. Frames outside complete cycles keep cycle -1 and
    phase NaN.
    """
    peaks = np.asarray(peaks, dtype=np.int64)
    if peaks.ndim != 1 or len(peaks) < 2 or np.any(np.diff(peaks) <= 0):
        raise ValueError("Peaks must be a strictly increasing one-dimensional array")
    if peaks[0] < 0 or peaks[-1] >= count:
        raise ValueError("Peak indices fall outside the prepared sequence")
    cycle = np.full(count, -1, dtype=np.int64)
    fraction = np.full(count, np.nan, dtype=np.float64)
    intervals = []
    for cycle_id, (start, stop) in enumerate(zip(peaks[:-1], peaks[1:])):
        start, stop = int(start), int(stop)
        indices = np.arange(start, stop)
        cycle[indices] = cycle_id
        fraction[indices] = (indices - start) / float(stop - start)
        intervals.append((start, stop))
    return cycle, fraction, tuple(intervals)

def prepare(args):
    """Prepare one file, or all supported files directly inside an input folder."""
    if Path(args.input).is_dir():
        if getattr(args, "dataset_measure", None):
            raise ValueError("--dataset-measure is only for one AVI/NPY input file")
        if any(p.is_dir() and list(p.glob("*.h5")) for p in Path(args.input).iterdir()):
            from .. import noise2time as api
            return load_sibling("dataset_workflow").prepare_dataset(args, api)
        if getattr(args,"avi",False) or getattr(args,"measures",None):
            raise ValueError("--avi and --measures require a dataset of HDF5 measurement folders")
        return prepare_folder(args)
    if getattr(args,"avi",False) or getattr(args,"measures",None):
        raise ValueError("--avi and --measures require a dataset of HDF5 measurement folders")
    return prepare_one(args)


def prepare_folder(args):
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if source == output:
        raise ValueError("Use separate input and prepared-output folders")
    pattern = getattr(args, "pattern", "*")
    if "/" in pattern or "\\" in pattern or "**" in pattern:
        raise ValueError("--pattern must select filenames in the input folder, without recursion")
    paths = sorted((p for p in source.glob(pattern)
                    if p.is_file() and p.suffix.lower() in (".avi", ".npy")),
                   key=lambda p:p.name.casefold())
    if not paths:
        raise ValueError(f"No AVI/NPY files matched in {source}")
    names = [p.stem.casefold() for p in paths]
    if len(set(names)) != len(names):
        raise ValueError("Multiple input files have the same stem; use --pattern '*.avi' or '*.npy'")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for number, path in enumerate(paths, 1):
        destination = output / path.stem
        row = dict(source=str(path), output=str(destination))
        print(f"[{number}/{len(paths)}] Preparing {path.name}", flush=True)
        try:
            if destination.exists():
                if getattr(args, "skip_existing", False):
                    results.append(dict(row, status="skipped", reason="Existing output not verified"))
                    print("  Skipped existing output (not verified)", flush=True)
                    continue
                raise FileExistsError(f"Output already exists: {destination}; use --skip-existing to leave it untouched")
            # Process one video at a time and publish only complete recordings.
            with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output) as temporary:
                temporary_path = Path(temporary).resolve()
                if temporary_path.parent != output or destination.resolve().parent != output:
                    raise ValueError("Temporary/output path escaped prepared-output folder")
                staging = temporary_path / "record"
                options = argparse.Namespace(**vars(args))
                options.input, options.output = str(path), str(staging)
                prepare_one(options)
                staging.rename(destination)
            results.append(dict(row, status="prepared"))
        except Exception as exc:
            results.append(dict(row, status="error", error=f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR: {exc}", flush=True)
    counts = {key:sum(row["status"] == key for row in results) for key in ("prepared", "skipped", "error")}
    fd, report_path = tempfile.mkstemp(prefix="preparation_", suffix=".json", dir=output)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(dict(created_utc=datetime.now(timezone.utc).isoformat(),
                       input=str(source), counts=counts, results=results), stream, indent=2)
    print(f"Preparation summary: {counts}\nReport: {report_path}", flush=True)
    return 1 if counts["error"] else 0


def prepare_one(args):
    frames, container_fps = load_frames(args.input)
    fps = args.fps if args.fps is not None else container_fps
    if fps is None or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Supply --fps with the acquisition frame rate")
    if args.smooth_window < 1 or args.peak_distance < 1 or args.prominence <= 0 or args.radius < 1:
        raise ValueError("Invalid preprocessing parameters")
    roi = circle_mask(*frames.shape[1:], args.cx, args.cy, args.radius)
    if not roi.any():
        raise ValueError("Circular ROI is empty")
    frames = frames * roi
    linked_measure = getattr(args, "dataset_measure", None)
    linked_folder = Path(linked_measure).resolve() if linked_measure else None
    if linked_folder is not None:
        if not linked_folder.is_dir():
            raise ValueError(f"Dataset measurement folder does not exist: {linked_folder}")
        if args.brightness_mask:
            raise ValueError("--dataset-measure uses its manual retinal artery mask; omit --brightness-mask")
        artery_path = str(load_sibling("dataset_workflow").manual_mask(linked_folder, "artery"))
    else:
        artery_path = getattr(args, "artery_mask", None)
    # Article Eq. 2 uses all image pixels. Optional external vessel mask reproduces
    # the legacy preprocessing's different brightness definition.
    if args.brightness_mask:
        bm = np.load(args.brightness_mask, allow_pickle=False).astype(bool)
        if bm.shape != roi.shape or not (bm & roi).any():
            raise ValueError("Invalid brightness mask")
        raw = frames[:, bm & roi].mean(axis=1)
    else:
        raw = frames.mean(axis=(1, 2))
    win = args.smooth_window
    smooth = np.convolve(np.pad(raw, (win // 2, win - 1 - win // 2), mode="edge"),
                         np.ones(win) / win, mode="valid")
    method = getattr(args, "peak_method", "auto")
    method = ("arterial" if artery_path else "legacy") if method == "auto" else method
    peak_diagnostics = None
    if method == "arterial":
        if not artery_path:
            raise ValueError("Arterial peak detection requires --artery-mask")
        from . import arterial_peaks as detector
        # Strict geometry for a scientific signal; unlike visualization masks,
        # resizing without registration can select different vessels.
        if Path(artery_path).suffix.lower() == ".npy":
            mask_values = np.load(artery_path, allow_pickle=False)
        else:
            mask_values = cv2.imdecode(np.fromfile(artery_path, np.uint8), cv2.IMREAD_UNCHANGED)
        if mask_values is None or mask_values.shape[:2] != roi.shape:
            raise ValueError("Artery mask must match the prepared video geometry; no automatic resizing")
        from ..evaluation.metrics import load_evaluation_mask
        artery_mask = load_evaluation_mask(artery_path, roi.shape) & roi
        if not artery_mask.any():
            raise ValueError("Artery mask has no pixels inside the ROI")
        arterial_signal = np.array([frame[artery_mask].mean(dtype=np.float64) for frame in frames])
        peak_diagnostics = detector.detect_arterial_peaks(arterial_signal, float(fps),
            min_hz=getattr(args, "peak_min_hz", .5), max_hz=getattr(args, "peak_max_hz", 2.5))
        peaks = peak_diagnostics["peaks"]
        for message in peak_diagnostics["warnings"]:
            warnings.warn(message)
        if linked_folder is not None:
            # Linked AVI benchmarks use the same arterial brightness definition
            # as raw-HDF5 dataset preparations, but retain the AVI's pixels and FPS.
            raw = arterial_signal
            smooth = gaussian_filter1d(raw, max(.5, .02*fps), mode="reflect")
    else:
        amplitude = np.percentile(smooth, 95) - np.percentile(smooth, 5)
        peaks, _ = find_peaks(smooth, distance=args.peak_distance,
                             prominence=args.prominence * amplitude)
        if amplitude <= 0:
            raise ValueError("Brightness signal has no usable variation")
    if len(peaks) < 2:
        raise ValueError("Need at least two reliable detected peaks for donor pairing")
    # A linked AVI is itself the requested scientific input, so retain its
    # entire timeline. Legacy standalone preparation keeps its historical trim.
    first = 0 if linked_folder is not None else int(peaks[0])
    phase = phases_from_peaks(len(frames) - first, peaks - first)
    cycle, fractional_phase, _ = cycle_coordinates(len(frames) - first, peaks - first)
    if linked_folder is not None:
        phase[(peaks[-1]-first):] = -1
        valid = (cycle >= 0) & ~peak_diagnostics["artifact_mask"][first:]
    if any(size % 2 for size in frames.shape[1:]):
        raise ValueError("AVI previews require even image dimensions to avoid codec cropping")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "frames.npy", frames[first:].astype(np.float32))
    np.save(output / "roi.npy", roi)
    np.save(output / "brightness.npy", smooth[first:].astype(np.float32))
    np.save(output / "brightness_raw.npy", raw[first:].astype(np.float32))
    np.save(output / "phase.npy", phase)
    np.save(output / "cycle.npy", cycle)
    np.save(output / "fractional_phase.npy", fractional_phase)
    if linked_folder is not None:
        np.save(output / "valid_frames.npy", valid)
    if peak_diagnostics is not None:
        np.savez_compressed(output/"arterial_peak_diagnostics.npz", raw=arterial_signal,
                           **{k:v for k,v in peak_diagnostics.items() if isinstance(v,np.ndarray)})
        peak_summary = {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in peak_diagnostics.items()
                        if k not in ("smoothed","detection_signal","repaired","artifact_mask","acf")}
        peak_summary.update(method="arterial", mask_sha256=sha256(artery_path),
                            min_hz=getattr(args,"peak_min_hz",.5),max_hz=getattr(args,"peak_max_hz",2.5),
                            frames=len(frames),fps=float(fps),indices="original input frames, before trimming",
                            artifact_frames=np.flatnonzero(peak_diagnostics["artifact_mask"]).tolist(),
                            detector_sha256=sha256(Path(__file__).with_name("arterial_peaks.py")))
        write_json(output/"arterial_peaks.json",peak_summary)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        figure=Figure(figsize=(12,4),layout="constrained");FigureCanvasAgg(figure)
        ax=figure.subplots();times=np.arange(len(frames))/fps
        ax.plot(times,arterial_signal,color=".65",label="Raw arterial mean")
        ax.plot(times,peak_diagnostics["smoothed"],color="#147c91",label="Detection smoothing")
        ax.scatter(times[peaks],peak_diagnostics["smoothed"][peaks],color="red",marker="v",label="Selected maxima")
        ax.set(xlabel="Time from original input start (s)",ylabel="Arterial mean intensity",
               title="Arterial peaks — inspect warnings in arterial_peaks.json")
        ax.legend();figure.savefig(output/"arterial_peaks.png",dpi=130);figure.clear()
    save_preparation_previews(output, frames[first:], raw[first:], smooth[first:],
                              phase, cycle, fractional_phase, peaks-first, first,
                              float(fps), Path(args.input).stem)
    metadata = dict(source=str(Path(args.input).resolve()), source_sha256=sha256(args.input),
                    first_original_frame=first, original_frame_count=len(frames), fps=float(fps),
                    peaks=(peaks - first).tolist(), phase_definition="frames_since_peak",
                    matching_phase_definition="fractional position in complete peak-to-peak cycles; linear donor interpolation",
                    peak_method=method,
                    brightness_definition="vessel_mask" if args.brightness_mask else "whole_masked_frame",
                    brightness_mask_sha256=sha256(args.brightness_mask) if args.brightness_mask else None,
                    smooth_window=win, peak_distance=args.peak_distance, prominence_ratio=args.prominence,
                    circle=dict(cx=args.cx, cy=args.cy, radius=args.radius),
                    normalization="uint8 / 255 or supplied [0,1] floats; no contrast enhancement",
                    preview=dict(video="prepared.avi", brightness_plot="brightness.png",
                                 brightness_table="brightness.csv",
                                 video_conversion="MJPG; clip [0,1], round to uint8; no contrast normalization",
                                 table_indices="zero-based; times in seconds"))
    if linked_folder is not None:
        metadata.update(schema="noise2time.linked_avi.v1", record=linked_folder.name,
                        dataset_measure=str(linked_folder), input_mode="source_avi",
                        diaphragm_mask_applied=True,
                        artery_mask=str(Path(artery_path).resolve()),
                        artery_mask_sha256=sha256(artery_path),
                        brightness_definition="manual retinal artery intersected with diaphragm; mean then Gaussian sigma 20 ms",
                        valid_frames_definition="inside complete peak-to-peak intervals and not flagged as artifacts",
                        peak_min_hz=getattr(args,"peak_min_hz",.5),
                        peak_max_hz=getattr(args,"peak_max_hz",2.5),
                        source_representation="existing AVI supplied by the user; no HDF5 substitution or AVI round trip")
    write_json(output / "metadata.json", metadata)
    print(f"Prepared {len(frames)-first} frames; peaks={len(peaks)}; original offset={first}")


def save_preparation_previews(output, frames, raw, smooth, phase, cycle,
                              fractional_phase, peaks, first, fps, title):
    """Viewable, aligned artifacts; NPY remains the scientific training input."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    height, width = frames.shape[1:]
    writer = cv2.VideoWriter(str(output/"prepared.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                             fps, (width,height), True)
    try:
        if not writer.isOpened():
            raise RuntimeError("Cannot initialize prepared MJPG AVI writer")
        for frame in tqdm(frames, desc="Saving prepared AVI", unit="frame",
                          file=sys.stdout, dynamic_ncols=True, mininterval=1.0,
                          disable=False):
            view = np.rint(np.clip(frame,0,1)*255).astype(np.uint8)
            writer.write(cv2.cvtColor(view,cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()
    indices = np.arange(len(frames))
    is_peak = np.zeros(len(frames), dtype=np.uint8)
    is_peak[peaks] = 1
    times = indices/fps
    table = np.column_stack((indices, indices+first, times, (indices+first)/fps,
                             raw, smooth, phase, cycle, fractional_phase, is_peak))
    np.savetxt(output/"brightness.csv", table, delimiter=",", comments="",
               header="frame_index,original_frame_index,time_seconds,original_time_seconds,raw_brightness,smooth_brightness,phase_frames,cycle,fractional_phase,is_peak",
               fmt=["%d","%d","%.9g","%.9g","%.9g","%.9g","%d","%d","%.9g","%d"])
    figure = Figure(figsize=(12,6), constrained_layout=True)
    FigureCanvasAgg(figure)  # Headless: no GUI windows or global backend changes.
    curve, phase_axis = figure.subplots(2,1,sharex=True,gridspec_kw={"height_ratios":[3,1]})
    curve.plot(indices,raw,label="Raw brightness",color="#8597a6",linewidth=1)
    curve.plot(indices,smooth,label="Smoothed brightness",color="#126782",linewidth=1.6)
    curve.scatter(indices[peaks],smooth[peaks],label="Detected peaks",color="#cf5735",s=28,zorder=3)
    for peak in peaks:
        curve.axvline(indices[peak],color="#cf5735",alpha=.18,linewidth=.8)
    curve.set_ylabel("Mean intensity [0,1]")
    curve.set_title(f"{title}\nPrepared frames; original starting frame {first} (zero-based)")
    curve.legend(loc="best")
    curve.grid(alpha=.2)
    phase_axis.step(indices,phase,where="post",color="#126782",linewidth=1)
    phase_axis.set_ylabel("Phase\n(frames since peak)")
    phase_axis.set_xlabel("Frame number (zero-based, prepared video)")
    phase_axis.xaxis.get_major_locator().set_params(integer=True)
    phase_axis.grid(alpha=.2)
    figure.savefig(output/"brightness.png",dpi=160)
    figure.clear()




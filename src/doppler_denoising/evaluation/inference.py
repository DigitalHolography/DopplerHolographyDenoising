"""Restore checkpoints and export denoised NPY/AVI result bundles."""
import json
import os
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ..common import get_device, seed_all, sha256, write_json
from ..config import Config
from ..training.model import Noise2Time
from ..training.sampling import Record

def export_denoised(model, record, cfg, device, npy_path=None, avi_path=None, original_avi_path=None, frame_callback=None):
    """One inference pass; stream float NPY and/or fixed-scale MJPG AVI."""
    paths = [Path(p) for p in (npy_path, avi_path, original_avi_path) if p is not None]
    if not paths and frame_callback is None:
        raise ValueError("At least one output is required")
    if len(record.frames) <= cfg.history:
        raise ValueError("Record shorter than history")
    if any(p.exists() for p in paths):
        raise FileExistsError(next(p for p in paths if p.exists()))
    if paths and len({p.parent.resolve() for p in paths}) != 1:
        raise ValueError("AVI and NPY outputs must share a directory")
    fps = float(record.metadata["fps"])
    if (avi_path is not None or original_avi_path is not None) and (not np.isfinite(fps) or fps <= 0):
        raise ValueError("AVI output requires a finite positive acquisition FPS")
    parent = paths[0].parent.resolve() if paths else Path(tempfile.gettempdir()).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    previous_mode = model.training
    model.eval()
    try:
        with tempfile.TemporaryDirectory(prefix=".denoise-", dir=parent) as temporary:
            staging = Path(temporary).resolve()
            if staging.parent != parent:
                raise ValueError("Temporary output escaped output directory")
            restored, writer, original_writer = None, None, None
            try:
                if npy_path is not None:
                    restored = np.lib.format.open_memmap(staging/"frames.npy", mode="w+",
                                                        dtype=np.float32, shape=record.frames.shape)
                if avi_path is not None:
                    height, width = record.frames.shape[1:]
                    writer = cv2.VideoWriter(str(staging/"preview.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                                             fps, (width,height), True)
                    if not writer.isOpened():
                        raise RuntimeError("Cannot initialize MJPG AVI writer")
                if original_avi_path is not None:
                    height, width = record.frames.shape[1:]
                    original_writer = cv2.VideoWriter(str(staging/"original.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                                                     fps, (width,height), True)
                    if not original_writer.isOpened():
                        raise RuntimeError("Cannot initialize original MJPG AVI writer")
                with torch.inference_mode():
                    for t in tqdm(range(len(record.frames)), desc=f"Denoising {record.name}",
                                  unit="frame", file=sys.stdout, dynamic_ncols=True, mininterval=1.0,
                                  disable=False):
                        if t < cfg.history:
                            prediction = record.frames[t]
                        else:
                            stop = t if cfg.input_mode == "history_only" else t+1
                            sequence = torch.from_numpy(np.array(record.frames[t-cfg.history:stop], copy=True))[None].to(device)
                            prediction = model(sequence)[0,0].cpu().numpy()
                        if not np.isfinite(prediction).all():
                            raise FloatingPointError(f"Non-finite output at frame {t}")
                        if record.metadata.get("diaphragm_mask_applied", False):
                            # Convolutions may predict nonzero values outside the aperture.
                            prediction = np.where(record.roi, prediction, 0)
                        if frame_callback is not None:
                            frame_callback(t, prediction)
                        if restored is not None:
                            restored[t] = prediction  # Preserve unclipped scientific intensities.
                        if writer is not None:
                            view = np.rint(np.clip(prediction,0,1)*255).astype(np.uint8)
                            writer.write(cv2.cvtColor(view,cv2.COLOR_GRAY2BGR))
                        if original_writer is not None:
                            view = np.rint(np.clip(record.frames[t],0,1)*255).astype(np.uint8)
                            original_writer.write(cv2.cvtColor(view,cv2.COLOR_GRAY2BGR))
            finally:
                if writer is not None:
                    writer.release()
                if original_writer is not None:
                    original_writer.release()
                if restored is not None:
                    try:
                        restored.flush()
                    finally:
                        restored._mmap.close()  # Windows requires closure before rename.
            if npy_path is not None:
                (staging/"frames.npy").rename(npy_path)
            if avi_path is not None:
                (staging/"preview.avi").rename(avi_path)
            if original_avi_path is not None:
                (staging/"original.avi").rename(original_avi_path)
    finally:
        model.train(previous_mode)


def denoise(args):
    requested = Path(args.output)
    folder_mode = requested.is_dir() or requested.suffix.lower() not in (".npy", ".avi")
    record = Record(args.record)
    if folder_mode:
        measure = record.name if record.metadata.get("schema") == "noise2time.dataset.v1" else record.name.removesuffix("_HD_M0")
        if not measure or measure in (".", ".."):
            raise ValueError("Invalid measurement folder name")
        destination = requested.resolve() / measure
        if destination.exists():
            raise FileExistsError(destination)
        output, avi_path = destination/"denoised.npy", destination/"denoised.avi"
    else:
        output = requested.with_suffix(".npy") if requested.suffix.lower() == ".avi" else requested
        avi_path = requested if requested.suffix.lower() == ".avi" else None
    for path in (output, output.with_suffix(".json"), avi_path):
        if path is not None and path.exists():
            raise FileExistsError(path)
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = Config(**checkpoint["config"])
    cfg.validate()
    seed_all(cfg.seed)
    if len(record.frames) <= cfg.history:
        raise ValueError("Record shorter than history")
    model = Noise2Time(cfg.base_channels, cfg.convlstm).to(device)
    model.load_state_dict(checkpoint["model"])
    metadata = dict(record=str(record.path), record_sha256=sha256(record.path/"frames.npy"),
               checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=sha256(args.checkpoint),
               first_original_frame=record.metadata["first_original_frame"], fps=record.metadata["fps"],
               copied_prefix=cfg.history, epoch=checkpoint["epoch"],
               inference="previous_frames_only" if cfg.input_mode == "history_only" else "fully_visible_sliding_window",
               intensities="float32, unclipped",
               intensity_scale=record.metadata.get("intensity_scale",1.),
               input_mode=record.metadata.get("input_mode","legacy"),
               diaphragm_mask_applied=record.metadata.get("diaphragm_mask_applied",False),
               avi=str(avi_path.resolve()) if avi_path is not None else None,
               avi_conversion="MJPG; clip [0,1], round to uint8; no contrast normalization" if avi_path is not None else None)
    if folder_mode:
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata.update(original_avi=str(destination/"original.avi"),
                        original_definition=("Full-length selected input, fixed-scale preview; diaphragm masking follows preparation metadata; no trimming" if record.metadata.get("schema") == "noise2time.dataset.v1" else
                        "Prepared input before denoising; circularly masked and trimmed to first peak; aligned with denoised output"))
        with tempfile.TemporaryDirectory(prefix=".denoise-record-", dir=destination.parent) as temporary:
            staging = Path(temporary).resolve()
            if staging.parent != destination.parent.resolve() or destination.resolve().parent != staging.parent:
                raise ValueError("Output escaped destination parent")
            bundle = staging/"record"
            export_denoised(model, record, cfg, device, npy_path=bundle/"denoised.npy",
                            avi_path=bundle/"denoised.avi", original_avi_path=bundle/"original.avi")
            write_json(bundle/"denoised.json", metadata)
            bundle.rename(destination)
    else:
        export_denoised(model, record, cfg, device, npy_path=output, avi_path=avi_path)
        write_json(output.with_suffix(".json"), metadata)
    print(f"Saved {output}" + (f" and {avi_path}" if avi_path is not None else "")
          + f"; omit first {cfg.history} copied frames from metrics")




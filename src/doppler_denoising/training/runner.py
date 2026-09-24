"""Training loop, checkpointing, progress and epoch diagnostics."""
from dataclasses import asdict
import json
import os
import platform
from pathlib import Path
import random
import sys
import time
import warnings

import cv2
import numpy as np
import scipy
import torch
from torch import nn
from tqdm import tqdm

from ..common import get_device, load_sibling, seed_all, sha256, write_json
from ..config import Config, SAMPLING_SCHEMA
from .losses import loss_terms
from .model import Noise2Time
from .sampling import Record, replacement, resolve_training_records, select_train_samples_for_epoch, split_samples

def train(args):
    from .. import noise2time as api
    from ..evaluation.inference import export_denoised
    support = load_sibling("training_monitor")
    output = Path(args.output)
    resume = getattr(args, "resume", False)
    saved = torch.load(output/"last.pt", map_location="cpu", weights_only=True) if resume else None
    if saved is not None and saved.get("sampling_schema") != SAMPLING_SCHEMA:
        raise ValueError("This checkpoint used the older integer-phase sampler and cannot be resumed with fractional-cycle matching")
    cfg = (Config(**json.loads(Path(args.config).read_text(encoding="utf-8"))) if args.config
           else Config(**saved["config"]) if saved else Config())
    if getattr(args, "epochs", None) is not None:
        cfg.epochs = args.epochs
    requested_config = json.loads(json.dumps(asdict(cfg)))
    if saved and any(requested_config[key] != value for key,value in
                     json.loads(json.dumps(saved["config"])).items() if key != "epochs"):
        raise ValueError("Resume must keep the saved configuration; only total --epochs may change")
    cfg.validate()
    seed_all(cfg.seed)
    device = get_device(args.device)
    print(f"Training device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    record_paths = resolve_training_records(args.records)
    records = []
    for index, path in enumerate(record_paths, 1):
        print(f"Loading recording {index}/{len(record_paths)}: {path}", flush=True)
        records.append(Record(path))
    if len({r.name for r in records}) != len(records):
        raise ValueError("Record folder names must be unique")
    if len({r.frames.shape[1:] for r in records}) != 1:
        raise ValueError("All training records must have the same spatial dimensions")
    if cfg.effective_patch_mode() == "vessel_patches":
        for record in records:
            record.training_vessel_mask()
    training, validation = split_samples(records, cfg)
    if len({i for i, _ in training}) > cfg.samples_per_epoch:
        raise ValueError("samples_per_epoch must be at least the number of training records")
    print(f"Split: {len(training)} training targets, {len(validation)} validation targets. "
          f"Each epoch: {cfg.samples_per_epoch} training samples, "
          f"{(cfg.samples_per_epoch + cfg.batch_size - 1) // cfg.batch_size} batches.", flush=True)
    print("Checking training sample construction...", flush=True)
    # Validate pairing and patch placement before creating outputs.
    for stage, samples in (("train", training[:1]), ("valid", validation)):
        for i,t in samples:
            replacement(records[i], t, cfg, np.random.default_rng(cfg.seed), stage)
    split = dict(training=training, validation=validation,
               policy="record" if cfg.validation_records else "overlapping_sequences",
               phase_matching=SAMPLING_SCHEMA,
               records=[str(r.path) for r in records])
    if cfg.split_mode == "temporal":
        split.update(policy="disjoint_temporal_blocks", ranges={r.name:r.split_ranges for r in records},
                     partition_signals={r.name:r.split_signals for r in records})
    manifest = []
    for index, record in enumerate(records, 1):
        print(f"Hashing input files {index}/{len(records)}: {record.name}", flush=True)
        files = ["frames.npy","roi.npy","phase.npy","brightness.npy","metadata.json"]
        if (record.path/"valid_frames.npy").exists():
            files.append("valid_frames.npy")
        files += [name for name in ("cycle.npy", "fractional_phase.npy")
                  if (record.path/name).exists()]
        entry=dict(path=str(record.path),metadata=record.metadata,
                   hashes={name:sha256(record.path/name) for name in files})
        if cfg.effective_patch_mode()=="vessel_patches":
            entry["vessel_mask"]=record.vessel_mask_provenance
        manifest.append(entry)
    if resume:
        if json.loads((output/"provenance.json").read_text())["records"] != manifest:
            raise ValueError("Prepared records changed since training; cannot resume")
        if json.loads((output/"split.json").read_text()) != json.loads(json.dumps(split)):
            raise ValueError("Training/validation split changed; cannot resume")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "split.json", split)
        write_json(output / "provenance.json", dict(records=manifest, script_sha256=sha256(__file__),
               sampling_schema=SAMPLING_SCHEMA,
               python=sys.version, platform=platform.platform(), torch=torch.__version__,
               numpy=np.__version__, scipy=scipy.__version__, opencv=cv2.__version__, device=str(device)))
    write_json(output / "config.json", asdict(cfg))
    monitors = {}
    if not getattr(args, "no_epoch_metrics", False):
        for record in records[:1]:
            if "dataset_measure" in record.metadata:
                monitors[record.name] = support.Monitor(record, cfg.inference_prefix(), api,
                                                        getattr(args, "background_dilation_radius", 2))
    monitor_provenance = {name: monitor.provenance for name,monitor in monitors.items()}
    monitor_path = output/"monitor_masks.json"
    if resume and monitor_path.exists():
        previous_monitors = json.loads(monitor_path.read_text())
        if any(name in previous_monitors and previous_monitors[name] != value
               for name,value in monitor_provenance.items()):
            raise ValueError("Monitoring masks/settings changed; keep them unchanged when resuming")
        # Retain provenance for old curves when reducing monitoring to one video.
        monitor_provenance = dict(previous_monitors, **monitor_provenance)
    write_json(monitor_path, monitor_provenance)
    if not cfg.validation_records and cfg.split_mode == "mixed":
        print("Validation uses overlapping sequences and shared donor pools; it is not an independent test.", flush=True)
    print("Initializing model and optimizer...", flush=True)
    model = Noise2Time(cfg.base_channels, cfg.convlstm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    best, stale = float("inf"), 0
    rows, first_epoch = [], 1
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        first_epoch = saved["epoch"] + 1
        best = saved["best_validation"]
        history_path = output/"metrics.jsonl"
        if history_path.exists():
            for line in history_path.read_text().splitlines():
                try: row = json.loads(line)
                except json.JSONDecodeError: continue  # Old interrupted append may be incomplete.
                if row["epoch"] <= saved["epoch"]: rows.append(row)
        by_epoch = {row["epoch"]:row for row in rows}
        rows = [by_epoch[epoch] for epoch in sorted(by_epoch)]
        if not rows or rows[-1]["epoch"] != saved["epoch"]:
            rows.append(dict(epoch=saved["epoch"], **saved["metrics"]))
        stale = saved.get("stale", 0)
        if "stale" not in saved:
            for row in reversed(rows):
                if row["valid"]["total"] <= best: break
                stale += 1
        if "rng" in saved:
            rng.bit_generator.state = saved["rng"]
            torch.set_rng_state(saved["torch_rng"])
            random.setstate(saved["python_rng"])
            if device.type == "cuda" and saved.get("cuda_rng"):
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
        else:
            warnings.warn("Older checkpoint: model/optimizer restored; random sampling cannot be restored exactly")
        support.save_history(output, rows)
        print(f"Resuming after epoch {saved['epoch']}; target total: {cfg.epochs} epochs", flush=True)
    for epoch in range(first_epoch, cfg.epochs+1):
        if stale >= cfg.patience:
            print("Saved run already reached early stopping.", flush=True)
            break
        epoch_started = time.monotonic()
        summaries = {}
        for stage, pool in (("train", training), ("valid", validation)):
            model.train(stage == "train")
            choices = (select_train_samples_for_epoch(pool, cfg.samples_per_epoch, rng)
                       if stage == "train" else np.arange(len(pool)))
            sums = {key:0. for key in ("total","reconstruction","gradient","hessian")}
            seen = 0
            batch_size = cfg.batch_size if stage == "train" else 1
            batches = tqdm(range(0, len(choices), batch_size),
                           desc=f"Epoch {epoch}/{cfg.epochs} {stage}", unit="batch",
                           file=sys.stdout, dynamic_ncols=True, mininterval=1.0,
                           disable=getattr(args, "no_progress", False))
            for start in batches:
                samples = []
                for index in choices[start:start+batch_size]:
                    i,t = pool[int(index)]
                    # Fixed target, donor and patch choices on every validation pass.
                    sampler = rng if stage == "train" else np.random.default_rng(10000+int(index))
                    samples.append(replacement(records[i],t,cfg,sampler,stage))
                sequence, target, mask = [torch.from_numpy(np.stack(items)).to(device) for items in zip(*samples)]
                with torch.set_grad_enabled(stage == "train"):
                    parts = loss_terms(model(sequence), target, mask, cfg.objective)
                    loss = parts["total"].sum()  # Paper specifies sum across batch.
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite {stage} loss at epoch {epoch}")
                    if stage == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                for key in sums:
                    sums[key] += float(parts[key].detach().sum())
                seen += len(samples)
                batches.set_postfix(loss=f"{sums['total']/seen:.6g}",
                                    samples=f"{seen}/{len(choices)}", refresh=False)
            if seen == 0:
                raise ValueError(f"No {stage} samples evaluated")
            summaries[stage] = {key:value/seen for key,value in sums.items()}
        value = summaries["valid"]["total"]
        improved = value < best
        best, stale = (value, 0) if improved else (best, stale+1)
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=asdict(cfg),
                          epoch=epoch, best_validation=best, metrics=summaries,
                          sampling_schema=SAMPLING_SCHEMA,
                          stale=stale, rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                          python_rng=random.getstate(),
                          cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                          script_sha256=sha256(__file__))
        print(f"Epoch {epoch}/{cfg.epochs}: saving checkpoints...", flush=True)
        if improved:
            support.atomic_checkpoint(torch, checkpoint, output / "best.pt")
        support.atomic_checkpoint(torch, checkpoint, output / "last.pt")
        # Retain inference weights so new measurements can be scored at every
        # epoch later, without saving another copy of the optimizer state.
        epoch_folder = output / "checkpoints"
        epoch_folder.mkdir(exist_ok=True)
        support.atomic_checkpoint(torch, dict(model=checkpoint['model'], config=checkpoint['config'],
                                             epoch=epoch, metrics=summaries),
                                  epoch_folder / f"epoch_{epoch:03d}.pt")
        rows.append(dict(epoch=epoch, **summaries))
        support.save_history(output, rows)
        if monitors or not getattr(args, "no_epoch_previews", False):
            preview_dir = output / "previews" / f"epoch_{epoch:03d}"
            print(f"Epoch {epoch}/{cfg.epochs}: computing diagnostics/exporting previews...", flush=True)
            diagnostics = {}
            for index, record in enumerate(records[:1]):
                monitor = monitors.get(record.name)
                if monitor: monitor.reset()
                preview_path = None if getattr(args, "no_epoch_previews", False) else preview_dir / f"{record.name}.avi"
                if preview_path is None and monitor is None: continue
                export_denoised(model, record, cfg, device, avi_path=preview_path,
                                frame_callback=monitor.update if monitor else None)
                if monitor: diagnostics[record.name] = monitor.result()
                if preview_path is not None: write_json(preview_path.with_suffix(".json"), dict(
                    epoch=epoch, record=str(record.path), config=asdict(cfg),
                    record_sha256=manifest[index]["hashes"]["frames.npy"],
                    first_original_frame=record.metadata["first_original_frame"],
                    fps=record.metadata["fps"], copied_prefix=cfg.inference_prefix(),
                    intensities="MJPG preview: clip [0,1], round to uint8; no contrast normalization",
                    weights_source="Current epoch model; inference weights retained in runs/checkpoints"))
            rows[-1]["diagnostics"] = diagnostics
            support.save_history(output, rows)
        print(f"Epoch {epoch}/{cfg.epochs}: train={summaries['train']['total']:.6g}, "
              f"valid={value:.6g}, best={best:.6g}, "
              f"no improvement={stale}/{cfg.patience}, "
              f"elapsed={time.monotonic()-epoch_started:.1f}s", flush=True)
        if stale >= cfg.patience:
            print(f"Early stopping after {stale} epochs without improvement.", flush=True)
            break
    print(f"Training finished. Best checkpoint: {output / 'best.pt'}", flush=True)




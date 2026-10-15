"""Step-based pretraining and fine-tuning.

"Virtual epoch" is only a reporting unit.  Optimization is controlled by
global steps, so adding videos does not increase the training budget.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from ..common import get_device, seed_all, sha256, write_json
from ..config import Config, SAMPLING_SCHEMA
from ..training.losses import loss_terms
from ..training.model import Noise2Time
from ..training.monitor import atomic_checkpoint
from ..training.sampling import Record, replacement, split_samples
from .sampling import PatientBalancedSampler, deterministic_validation, eligible_pools


def _set_scope(model, scope):
    if scope not in ("full", "decoder_temporal"):
        raise ValueError("trainable scope must be full or decoder_temporal")
    for parameter in model.parameters():
        parameter.requires_grad = scope == "full"
    if scope == "decoder_temporal":
        for name in ("bottleneck", "memory", "temporal_attention", "decoders", "up", "final"):
            module = getattr(model, name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = True


def _batch(records, samples, cfg, device, rng, stage):
    replacement_stage = stage if cfg.split_mode == "temporal" else None
    built = [replacement(records[i], t, cfg, rng if stage == "train"
                         else np.random.default_rng(10_000 + i * 1_000_003 + t),
                         stage=replacement_stage)
             for i, t in samples]
    sequence, target, mask = [torch.from_numpy(np.stack(items)).to(device)
                              for items in zip(*built)]
    return sequence, target, mask


@torch.no_grad()
def _validate(model, records, samples, cfg, device):
    model.eval()
    totals = {key: 0.0 for key in ("total", "reconstruction", "gradient", "hessian")}
    for sample in samples:
        sequence, target, mask = _batch(records, [sample], cfg, device, None, "valid")
        parts = loss_terms(model(sequence), target, mask, cfg.objective)
        for key in totals:
            totals[key] += float(parts[key].sum())
    return {key: value / len(samples) for key, value in totals.items()}


def train_steps(*, train_records, train_patients, validation_records, config,
                output, max_steps, eval_every_steps, checkpoint_every_steps,
                validation_anchors_per_video=20, device="auto", initialization=None,
                learning_rate=None, trainable_scope="full", resume=False,
                early_stopping_evaluations=10, run_metadata=None,
                temporal_finetune=False, validation_fraction=0.25):
    """Train with fixed optimizer work and patient-balanced online samples."""
    output = Path(output).resolve()
    cfg = Config(**json.loads(Path(config).read_text(encoding="utf-8")))
    if learning_rate is not None:
        cfg.learning_rate = learning_rate
    cfg.split_mode, cfg.validation_records = ("temporal" if temporal_finetune else "mixed"), ()
    cfg.validation_fraction = validation_fraction
    cfg.validate()
    seed_all(cfg.seed)
    device = get_device(device)
    train_data = [Record(path) for path in train_records]
    valid_data = [Record(path) for path in validation_records]
    if len(train_data) != len(train_patients):
        raise ValueError("A patient label is required for every training record")
    geometries = {record.frames.shape[1:] for record in train_data + valid_data}
    if len(geometries) != 1:
        raise ValueError(f"All records in one run need the same geometry; found {sorted(geometries)}")
    for record in train_data + valid_data:
        if cfg.effective_patch_mode() in ("vessel_patches", "black_patches"):
            record.training_vessel_mask()
    rng = np.random.default_rng(cfg.seed)
    if temporal_finetune:
        if len(train_data) != 1 or valid_data:
            raise ValueError("Temporal fine-tuning needs exactly one adaptation record and no external validation records")
        training_pairs, validation = split_samples(train_data, cfg)
        train_pool = [[t for i, t in training_pairs if i == 0]]
        sampler = PatientBalancedSampler(train_pool, train_patients, rng)
        valid_data = train_data
    else:
        sampler = PatientBalancedSampler(eligible_pools(train_data, cfg), train_patients, rng)
        validation = deterministic_validation(valid_data, cfg, validation_anchors_per_video)
    model = Noise2Time(cfg.base_channels, cfg.convlstm, cfg.effective_model()).to(device)
    if initialization:
        source = torch.load(initialization, map_location="cpu", weights_only=True)
        model.load_state_dict(source["model"])
    _set_scope(model, trainable_scope)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    state_path = output / "last.pt"
    history, step, best, stale = [], 0, float("inf"), 0
    if resume:
        saved = torch.load(state_path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        step, best, stale = saved["global_step"], saved["best_validation"], saved["stale"]
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"]); random.setstate(saved["python_rng"])
        history = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines() if line.strip()]
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "config.json", asdict(cfg))
        write_json(output / "run.json", {"sampling_unit": "patient", "max_steps": max_steps,
                   "eval_every_steps": eval_every_steps, "checkpoint_every_steps": checkpoint_every_steps,
                   "virtual_epoch_steps": eval_every_steps, "trainable_scope": trainable_scope,
                   "initialization": str(Path(initialization).resolve()) if initialization else "random",
                   "training_records": [str(Path(p).resolve()) for p in train_records],
                   "training_patients": list(train_patients),
                   "validation_records": [str(Path(p).resolve()) for p in validation_records],
                   "metadata": run_metadata or {}})
    started, running = time.monotonic(), {key: 0.0 for key in ("total", "reconstruction", "gradient", "hessian")}
    running_count = 0
    print(f"Device={device}; records={len(train_data)}; patients={len(set(train_patients))}; "
          f"steps={step}/{max_steps}; batch={cfg.batch_size}", flush=True)
    while step < max_steps and stale < early_stopping_evaluations:
        model.train()
        samples = sampler.draw(cfg.batch_size)
        sequence, target, mask = _batch(train_data, samples, cfg, device, rng, "train")
        parts = loss_terms(model(sequence), target, mask, cfg.objective)
        loss = parts["total"].sum()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {step + 1}")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
        step += 1; running_count += len(samples)
        for key in running:
            running[key] += float(parts[key].detach().sum())
        should_evaluate = step % eval_every_steps == 0 or step == max_steps
        if should_evaluate:
            train_mean = {key: value / running_count for key, value in running.items()}
            valid_mean = _validate(model, valid_data, validation, cfg, device)
            improved = valid_mean["total"] < best
            best, stale = (valid_mean["total"], 0) if improved else (best, stale + 1)
            row = {"global_step": step, "virtual_epoch": step / eval_every_steps,
                   "train": train_mean, "valid": valid_mean,
                   "elapsed_seconds": time.monotonic() - started,
                   "best_validation": best, "stale_evaluations": stale}
            history.append(row)
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            running = {key: 0.0 for key in running}; running_count = 0
            print(f"step {step}/{max_steps} (virtual epoch {row['virtual_epoch']:.2f}): "
                  f"train={train_mean['total']:.6g}, valid={valid_mean['total']:.6g}, "
                  f"best={best:.6g}, stale={stale}/{early_stopping_evaluations}", flush=True)
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "config": asdict(cfg), "epoch": int(np.ceil(step / eval_every_steps)),
                      "global_step": step, "best_validation": best, "stale": stale,
                      "sampling_schema": SAMPLING_SCHEMA, "numpy_rng": rng.bit_generator.state,
                      "torch_rng": torch.get_rng_state(), "python_rng": random.getstate()}
        if should_evaluate and improved:
            atomic_checkpoint(torch, checkpoint, output / "best.pt")
        if should_evaluate or step % checkpoint_every_steps == 0 or step == max_steps:
            atomic_checkpoint(torch, checkpoint, state_path)
    write_json(output / "summary.json", {"completed_steps": step, "requested_steps": max_steps,
               "best_validation": best, "early_stopped": stale >= early_stopping_evaluations})
    return output / "best.pt"

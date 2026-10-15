"""Discover repeated-eye measurements and freeze the experimental design.

The manifest is the source of truth for every later command. Repeat A/B/C
means the stable order recorded here; directory suffixes are not trusted.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np

from ..common import sha256, write_json
from ..preparation.dataset import choroidal_masks, manual_mask, resolve_m0_key, single_h5

SCHEMA = "doppler-denoising.scaling-study.v1"
NAME = re.compile(r"^(?P<date>\d{6})_(?P<patient>[^_]+)_(?P<eye>L|R)(?:_(?P<suffix>\d+))?$", re.I)


def parse_measure_name(name: str) -> dict:
    match = NAME.fullmatch(name)
    if not match:
        raise ValueError(f"Cannot infer patient and eye from {name!r}; expected DATE_PATIENT_L/R[_REPEAT]")
    values = match.groupdict()
    values["eye"] = values["eye"].upper()
    return values


def _file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def discover(dataset: Path) -> list[dict]:
    """Audit HDF5 geometry and masks without loading video arrays into RAM."""
    dataset = dataset.resolve()
    folders = sorted((p for p in dataset.iterdir() if p.is_dir() and list(p.glob("*.h5"))),
                     key=lambda p: p.name.casefold())
    if not folders:
        raise ValueError(f"No measurement folders containing one HDF5 file in {dataset}")
    rows = []
    for folder in folders:
        identity = parse_measure_name(folder.name)
        h5_path = single_h5(folder)
        with h5py.File(h5_path, "r") as handle:
            key = resolve_m0_key(handle)
            data = handle[key]
            if data.ndim != 3:
                raise ValueError(f"{folder.name}: expected a (time,height,width) M0 sequence")
            shape, dtype = list(data.shape), str(data.dtype)
        artery, vein = manual_mask(folder, "artery"), manual_mask(folder, "vein")
        choroid = choroidal_masks(folder)[0][0]
        small = folder / "manual" / "small_vessels.png"
        rows.append({
            "measure": folder.name, "date": identity["date"], "patient": identity["patient"],
            "eye": identity["eye"], "name_suffix": identity["suffix"],
            "measure_path": str(folder.resolve()), "h5": _file_identity(h5_path),
            "h5_key": key, "shape": shape, "dtype": dtype,
            "artery_mask": str(artery.resolve()), "artery_sha256": sha256(artery),
            "vein_mask": str(vein.resolve()), "vein_sha256": sha256(vein),
            "choroidal_mask": str(choroid.resolve()), "choroidal_sha256": sha256(choroid),
            "small_vessels_mask": str(small.resolve()) if small.exists() else None,
            "small_vessels_sha256": sha256(small) if small.exists() else None,
        })
    groups = {}
    for row in rows:
        groups.setdefault((row["patient"], row["eye"]), []).append(row)
    for key, group in groups.items():
        if len(group) != 3:
            raise ValueError(f"{key[0]} {key[1]}: expected exactly three repeated measurements; found {len(group)}")
        for rank, row in enumerate(sorted(group, key=lambda x: (x["date"], x["measure"].casefold()))):
            row["repeat_rank"], row["repeat_role"] = rank, "ABC"[rank]
    return sorted(rows, key=lambda x: (x["patient"], x["eye"], x["repeat_rank"]))


def resolve_dataset_root(dataset: Path) -> Path:
    """Accept either the measurement root or its conventional ``retina`` parent."""
    dataset = dataset.resolve()
    direct = [p for p in dataset.iterdir() if p.is_dir() and list(p.glob("*.h5"))]
    if direct:
        return dataset
    retina = dataset / "retina"
    if retina.is_dir() and any(p.is_dir() and list(p.glob("*.h5")) for p in retina.iterdir()):
        return retina.resolve()
    raise ValueError(f"No measurement folders found directly in {dataset} or {retina}")


def default_protocol() -> dict:
    return {
        "schema": SCHEMA,
        "patient_split": {"pretraining": 13, "validation": 3, "test": 5},
        "subset_patient_counts": [4, 7, 10, 13], "subset_orders": 3,
        "diversity_ablation_patient_counts": [4, 8, 13],
        "pretraining": {"max_steps": 17500, "eval_every_steps": 1750,
                        "checkpoint_every_steps": 1750, "validation_anchors_per_video": 20,
                        "early_stopping_evaluations": 10},
        "fine_tuning": {"learning_rates": [5e-6, 1e-5],
                        "step_budgets": [0, 250, 500, 1000, 1750],
                        "trainable_scopes": ["full", "decoder_temporal"],
                        "adaptation_repeat": "A", "evaluation_repeats": ["B", "C"],
                        "validation_fraction": 0.25},
        "baselines": ["zero_shot", "pretrained_finetuned", "random_init_same_budget",
                      "target_only_convergence", "temporal_average"],
        "primary_metrics": ["off_harmonic_noise_reduction_db", "NRR",
                            "retinal_artery_temporal_correlation", "retinal_artery_amplitude_ratio"],
        "small_vessel_policy": "optional; report missingness, never synthesize a manual mask",
    }


def _split_patients(patients: list[str], seed: int, counts: dict) -> dict:
    if sum(counts.values()) != len(patients):
        raise ValueError(f"Split counts sum to {sum(counts.values())}, but dataset has {len(patients)} patients")
    order = np.random.default_rng(seed).permutation(patients).tolist()
    a, b = counts["pretraining"], counts["pretraining"] + counts["validation"]
    return {"pretraining": order[:a], "validation": order[a:b], "test": order[b:]}


def _jobs(rows: list[dict], split: dict, protocol: dict, seed: int) -> list[dict]:
    by_patient = {}
    for row in rows:
        by_patient.setdefault(row["patient"], []).append(row)
    jobs, base = [], split["pretraining"]
    for order_index in range(protocol["subset_orders"]):
        order = np.random.default_rng(seed + 1009 * order_index).permutation(base).tolist()
        for count in protocol["subset_patient_counts"]:
            selected = order[:count]
            measures = sorted(r["measure"] for p in selected for r in by_patient[p])
            jobs.append({"name": f"scale_o{order_index+1}_p{count:02d}_r3", "kind": "scaling",
                         "order": order_index + 1, "patient_count": count, "repeats_per_eye": 3,
                         "patients": selected, "measures": measures})
    first_order = np.random.default_rng(seed).permutation(base).tolist()
    for count in protocol["diversity_ablation_patient_counts"]:
        selected = first_order[:count]
        measures = sorted(r["measure"] for p in selected for r in by_patient[p] if r["repeat_role"] == "A")
        jobs.append({"name": f"diversity_p{count:02d}_r1", "kind": "diversity_ablation",
                     "order": 1, "patient_count": count, "repeats_per_eye": 1,
                     "patients": selected, "measures": measures})
    return jobs


def build_study(dataset, output, seed=2026, protocol_overrides=None) -> dict:
    dataset, output = resolve_dataset_root(Path(dataset)), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Study folder must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = discover(dataset)
    protocol = default_protocol()
    if protocol_overrides:
        protocol.update(protocol_overrides)
    patients = sorted({row["patient"] for row in rows})
    split = _split_patients(patients, seed, protocol["patient_split"])
    jobs = _jobs(rows, split, protocol, seed)
    manifest = {"schema": SCHEMA, "dataset": str(dataset), "seed": seed,
                "counts": {"measures": len(rows), "patients": len(patients),
                           "eyes": len({(r['patient'], r['eye']) for r in rows}),
                           "small_vessel_masks": sum(r["small_vessels_mask"] is not None for r in rows)},
                "patient_split": split, "measurements": rows, "jobs": jobs}
    write_json(output / "protocol.json", protocol)
    write_json(output / "manifest.json", manifest)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["measure", "patient", "eye", "repeat_role", "repeat_rank", "date",
                  "name_suffix", "measure_path", "h5_key", "shape", "dtype", "small_vessels_mask"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader()
        for row in rows:
            writer.writerow({**row, "shape": "x".join(map(str, row["shape"]))})
    with (output / "jobs.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["name", "kind", "order", "patient_count", "repeats_per_eye", "measure_count", "patients"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for job in jobs:
            writer.writerow({"name": job["name"], "kind": job["kind"], "order": job["order"],
                             "patient_count": job["patient_count"], "repeats_per_eye": job["repeats_per_eye"],
                             "measure_count": len(job["measures"]), "patients": ";".join(job["patients"])})
    digest = hashlib.sha256((output / "manifest.json").read_bytes() +
                            (output / "protocol.json").read_bytes()).hexdigest()
    (output / "STUDY_SHA256.txt").write_text(digest + "\n", encoding="ascii")
    return manifest


def load_study(study) -> tuple[Path, dict, dict]:
    root = Path(study).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or protocol.get("schema") != SCHEMA:
        raise ValueError("Unsupported or incomplete scaling-study manifest")
    return root, manifest, protocol

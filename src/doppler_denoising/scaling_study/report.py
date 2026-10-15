"""Consolidate learning curves, held-out metrics and repeatability."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from ..common import write_json
from .evaluation import flatten_metrics, repeatability

DISPLAY_METRICS = (
    "NRR", "background_off_harmonic_noise_reduction_db",
    "retinal_artery_temporal_correlation", "retinal_artery_amplitude_ratio",
    "retinal_artery_off_harmonic_noise_reduction_db",
    "choroidal_off_harmonic_noise_reduction_db",
)


def _training_rows(root):
    rows = []
    for path in sorted((root / "pretraining").glob("*/metrics.jsonl")):
        run = path.parent.name
        details = json.loads((path.parent / "run.json").read_text())
        metadata = details.get("metadata", {})
        for line in path.read_text().splitlines():
            item = json.loads(line)
            rows.append({"run": run, "patient_count": metadata.get("patient_count"),
                         "order": metadata.get("order"), "global_step": item["global_step"],
                         "train_loss": item["train"]["total"], "valid_loss": item["valid"]["total"]})
    return rows


def _evaluation_rows(root):
    rows = []
    for path in sorted((root / "evaluation").glob("**/metrics.json")):
        data = json.loads(path.read_text())
        row = flatten_metrics(data)
        relative = path.relative_to(root / "evaluation").parts
        row["model"] = relative[0] if len(relative) > 1 else "unknown"
        row["cohort"] = relative[1] if len(relative) > 2 else "unknown"
        rows.append(row)
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _training_plot(path, rows):
    if not rows:
        return None
    figure = Figure(figsize=(10, 6), layout="constrained"); FigureCanvasAgg(figure); ax = figure.subplots()
    grouped = {}
    for row in rows:
        if row["patient_count"] is not None:
            grouped.setdefault(row["patient_count"], {}).setdefault(row["global_step"], []).append(row["valid_loss"])
    for count in sorted(grouped):
        steps = sorted(grouped[count]); values = [grouped[count][step] for step in steps]
        mean = np.asarray([np.mean(x) for x in values]); std = np.asarray([np.std(x, ddof=1) if len(x)>1 else 0 for x in values])
        ax.plot(steps, mean, label=f"{count} patients / {count*6} videos")
        ax.fill_between(steps, mean-std, mean+std, alpha=.18)
    ax.set(xlabel="Optimizer step", ylabel="Validation loss", title="Pretraining scaling curves (mean ± SD across subset orders)")
    ax.grid(alpha=.2); ax.legend(fontsize=8); figure.savefig(path, dpi=150); figure.clear(); return path.name


def _boxplot(path, rows):
    if not rows:
        return None
    metric = "retinal_artery_off_harmonic_noise_reduction_db"
    models = sorted({r["model"] for r in rows})
    cohorts = sorted({r["cohort"] for r in rows})
    figure = Figure(figsize=(max(10, len(models)*1.4), 6), layout="constrained"); FigureCanvasAgg(figure); ax = figure.subplots()
    values, labels = [], []
    for model in models:
        for cohort in cohorts:
            group = [r.get(metric) for r in rows if r["model"] == model and r["cohort"] == cohort and r.get(metric) is not None]
            if group:
                values.append(group); labels.append(f"{model}\n{cohort}")
    if not values:
        figure.clear(); return None
    ax.boxplot(values, tick_labels=labels, showfliers=True)
    ax.set(ylabel="Reduction (dB)", title="Off-harmonic arterial noise reduction; points beyond whiskers are outliers")
    ax.tick_params(axis="x", labelrotation=35); ax.grid(axis="y", alpha=.2)
    figure.savefig(path, dpi=150); figure.clear(); return path.name


def build_report(study):
    root = Path(study).resolve(); destination = root / "report"; destination.mkdir(exist_ok=True)
    training, evaluations = _training_rows(root), _evaluation_rows(root)
    _write_csv(destination / "training_curves.csv", training)
    _write_csv(destination / "evaluation_metrics.csv", evaluations)
    training_image = _training_plot(destination / "training_curves.png", training)
    box_image = _boxplot(destination / "evaluation_boxplots.png", evaluations)
    repeat = {}
    for model in sorted({row["model"] for row in evaluations}):
        subset = [row for row in evaluations if row["model"] == model]
        repeat[model] = {metric: repeatability(subset, metric) for metric in DISPLAY_METRICS}
    write_json(destination / "repeatability.json", repeat)
    sections = ["<h1>Denoising dataset-scaling study</h1>",
                "<p>Patient-disjoint results. Training work is expressed in optimizer steps; virtual epochs are plotting labels only.</p>"]
    if training_image: sections.append(f'<h2>Learning curves</h2><img src="{training_image}" alt="training curves">')
    if box_image: sections.append(f'<h2>Held-out distributions</h2><img src="{box_image}" alt="metric boxplots">')
    sections.append("<h2>Repeatability</h2><p>ICC(2,1), within-subject CV, and Bland–Altman B−C limits are in <code>repeatability.json</code>.</p>")
    if not evaluations:
        sections.append("<p><strong>No evaluation rows yet.</strong> Run the evaluate command after training.</p>")
    missing_small = sum(row.get("small_vessels_off_harmonic_noise_reduction_db") is None for row in evaluations)
    if missing_small:
        sections.append(f"<p>Small-vessel metrics are missing for {missing_small} result(s); no mask was fabricated.</p>")
    document = "<!doctype html><meta charset='utf-8'><title>Scaling study</title><style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}img{max-width:100%}code{background:#eee;padding:.1rem .3rem}</style>" + "".join(sections)
    (destination / "report.html").write_text(document, encoding="utf-8")
    return destination / "report.html"

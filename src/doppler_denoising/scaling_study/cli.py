"""Command-line workflow for patient-disjoint dataset-scaling experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from .. import noise2time as api
from ..preparation.dataset import prepare_dataset
from .evaluation import score_measure
from .manifest import build_study, load_study
from .report import build_report
from .training import train_steps


def _rows(manifest):
    return {row["measure"]: row for row in manifest["measurements"]}


def _prepared(root, measure):
    path = root / "prepared" / measure
    if not (path / "metadata.json").exists():
        raise ValueError(f"Missing preparation: {path}")
    return path


def command_plan(args):
    manifest = build_study(args.dataset, args.output, args.seed)
    print(f"Created study: {Path(args.output).resolve()}")
    print(json.dumps(manifest["counts"], indent=2)); return 0


def command_prepare(args):
    root, manifest, _ = load_study(args.study)
    options = SimpleNamespace(input=manifest["dataset"], output=root,
        measures=[row["measure"] for row in manifest["measurements"]], avi=args.avi,
        fps=args.fps, cx=args.cx, cy=args.cy, radius=args.radius,
        peak_min_hz=args.peak_min_hz, peak_max_hz=args.peak_max_hz,
        peak_method="arterial", artery_mask=None, brightness_mask=None,
        skip_existing=args.skip_existing)
    return prepare_dataset(options, api)


def command_pretrain(args):
    root, manifest, protocol = load_study(args.study); by_name = _rows(manifest)
    jobs = list(manifest["jobs"])
    if args.pilot:
        jobs = [j for j in jobs if j["order"] == 1 and j["patient_count"] in (7, 10, 13)]
    if args.jobs:
        wanted = set(args.jobs); jobs = [j for j in jobs if j["name"] in wanted]
        missing = wanted - {j["name"] for j in jobs}
        if missing: raise ValueError(f"Unknown jobs: {sorted(missing)}")
    validation = [row for row in manifest["measurements"] if row["patient"] in manifest["patient_split"]["validation"]]
    settings = protocol["pretraining"]
    for index, job in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {job['name']}", flush=True)
        output = root / "pretraining" / job["name"]
        train_steps(train_records=[_prepared(root, name) for name in job["measures"]],
                    train_patients=[by_name[name]["patient"] for name in job["measures"]],
                    validation_records=[_prepared(root, row["measure"]) for row in validation],
                    config=args.config, output=output, max_steps=settings["max_steps"],
                    eval_every_steps=settings["eval_every_steps"],
                    checkpoint_every_steps=settings["checkpoint_every_steps"],
                    validation_anchors_per_video=settings["validation_anchors_per_video"],
                    early_stopping_evaluations=settings["early_stopping_evaluations"],
                    device=args.device, resume=args.resume and output.exists(),
                    run_metadata={"job": job["name"], "kind": job["kind"],
                                  "patient_count": job["patient_count"], "video_count": len(job["measures"]),
                                  "order": job["order"]})
    return 0


def command_finetune(args):
    root, manifest, protocol = load_study(args.study)
    rows = [row for row in manifest["measurements"]
            if row["patient"] in manifest["patient_split"][args.cohort] and row["repeat_role"] == "A"]
    settings = protocol["fine_tuning"]
    scopes = args.scopes or settings["trainable_scopes"]
    rates = args.learning_rates or settings["learning_rates"]
    budgets = args.step_budgets or [step for step in settings["step_budgets"] if step]
    checkpoints = list(args.checkpoints) if args.checkpoints else sorted((root / "pretraining").glob("*/best.pt"))
    for checkpoint in map(Path, checkpoints):
        model_name = checkpoint.parent.name
        for row in rows:
            for scope in scopes:
                for lr in rates:
                    for steps in budgets:
                        name = f"{row['patient']}_{row['eye']}/{scope}_lr{lr:g}_s{steps}"
                        output = root / "fine_tuning" / args.cohort / model_name / name
                        print(f"Fine-tuning {model_name}: {name}", flush=True)
                        train_steps(train_records=[_prepared(root, row["measure"])], train_patients=[row["patient"]],
                            validation_records=[], config=args.config, output=output, max_steps=steps,
                            eval_every_steps=min(args.eval_every_steps, steps),
                            checkpoint_every_steps=min(args.eval_every_steps, steps),
                            validation_anchors_per_video=1, device=args.device, initialization=checkpoint,
                            learning_rate=lr, trainable_scope=scope,
                            early_stopping_evaluations=max(2, len(settings["step_budgets"])),
                            temporal_finetune=True, validation_fraction=settings["validation_fraction"],
                            resume=args.resume and output.exists(),
                            run_metadata={"source_model": model_name, "patient": row["patient"],
                                          "eye": row["eye"], "adaptation_measure": row["measure"]})
                        if args.random_init_control:
                            control = root / "fine_tuning" / args.cohort / (model_name + "_random_init") / name
                            print(f"Random-init control: {name}", flush=True)
                            train_steps(train_records=[_prepared(root, row["measure"])], train_patients=[row["patient"]],
                                validation_records=[], config=args.config, output=control, max_steps=steps,
                                eval_every_steps=min(args.eval_every_steps, steps),
                                checkpoint_every_steps=min(args.eval_every_steps, steps),
                                validation_anchors_per_video=1, device=args.device, initialization=None,
                                learning_rate=lr, trainable_scope=scope,
                                early_stopping_evaluations=max(2, len(settings["step_budgets"])),
                                temporal_finetune=True, validation_fraction=settings["validation_fraction"],
                                resume=args.resume and control.exists(),
                                run_metadata={"control": "random_init_same_budget", "patient": row["patient"],
                                              "eye": row["eye"], "adaptation_measure": row["measure"]})
    return 0


def command_evaluate(args):
    root, manifest, _ = load_study(args.study)
    patients = set(manifest["patient_split"][args.cohort])
    roles = set(args.repeats)
    selected = [row for row in manifest["measurements"] if row["patient"] in patients and row["repeat_role"] in roles]
    checkpoint = Path(args.checkpoint).resolve()
    if args.name:
        model_name = args.name
    else:
        try:
            relative = checkpoint.parent.relative_to(root / "fine_tuning")
            model_name = "__".join(relative.parts)
        except ValueError:
            model_name = checkpoint.parent.name
    patient, eye = args.patient, args.eye
    run_path = checkpoint.parent / "run.json"
    if run_path.exists() and (patient is None or eye is None):
        run_metadata = json.loads(run_path.read_text()).get("metadata", {})
        patient = patient or run_metadata.get("patient")
        eye = eye or run_metadata.get("eye")
    if patient is not None:
        selected = [row for row in selected if row["patient"] == patient]
    if eye is not None:
        selected = [row for row in selected if row["eye"] == eye]
    if not selected:
        raise ValueError("No measurements match the cohort, repeat, patient and eye selection")
    failures = 0
    for index, row in enumerate(selected, 1):
        output = root / "evaluation" / model_name / args.cohort / row["measure"]
        if output.exists() and args.skip_existing:
            continue
        print(f"[{index}/{len(selected)}] Evaluating {row['measure']}", flush=True)
        try:
            score_measure(checkpoint, _prepared(root, row["measure"]), row, output, args.device, args.background_dilation_radius)
        except Exception as error:
            failures += 1; print(f"ERROR {row['measure']}: {error}", flush=True)
    return int(failures > 0)


def command_report(args):
    path = build_report(args.study); print(f"Saved {path}"); return 0


def parser():
    root = argparse.ArgumentParser(description=__doc__); commands = root.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Audit the dataset and freeze patient splits/subsets")
    plan.add_argument("--dataset", required=True); plan.add_argument("--output", required=True)
    plan.add_argument("--seed", type=int, default=2026); plan.set_defaults(func=command_plan)
    prepare = commands.add_parser("prepare", help="Prepare all manifest videos once")
    prepare.add_argument("--study", required=True); prepare.add_argument("--avi", action="store_true")
    prepare.add_argument("--skip-existing", action="store_true"); prepare.add_argument("--fps", type=float)
    prepare.add_argument("--cx", type=int, default=255); prepare.add_argument("--cy", type=int, default=255)
    prepare.add_argument("--radius", type=int, default=260); prepare.add_argument("--peak-min-hz", type=float, default=.5)
    prepare.add_argument("--peak-max-hz", type=float, default=2.5); prepare.set_defaults(func=command_prepare)
    pre = commands.add_parser("pretrain", help="Run fixed-step nested-subset pretraining")
    pre.add_argument("--study", required=True); pre.add_argument("--config", required=True); pre.add_argument("--device", default="auto")
    pre.add_argument("--jobs", nargs="+"); pre.add_argument("--pilot", action="store_true"); pre.add_argument("--resume", action="store_true")
    pre.set_defaults(func=command_pretrain)
    fine = commands.add_parser("finetune", help="Tune pretrained models on validation repeat A")
    fine.add_argument("--study", required=True); fine.add_argument("--config", required=True); fine.add_argument("--device", default="auto")
    fine.add_argument("--checkpoints", nargs="+"); fine.add_argument("--eval-every-steps", type=int, default=250)
    fine.add_argument("--cohort", choices=("validation", "test"), default="validation",
                      help="Use test only after one policy has been selected on validation patients")
    fine.add_argument("--learning-rates", nargs="+", type=float); fine.add_argument("--step-budgets", nargs="+", type=int)
    fine.add_argument("--scopes", nargs="+", choices=("full", "decoder_temporal"))
    fine.add_argument("--random-init-control", action="store_true",
                      help="Also train a random-initialized model for the identical target-video budget")
    fine.add_argument("--resume", action="store_true"); fine.set_defaults(func=command_finetune)
    evaluate = commands.add_parser("evaluate", help="Denoise and score a patient-disjoint cohort")
    evaluate.add_argument("--study", required=True); evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--name"); evaluate.add_argument("--cohort", choices=("pretraining", "validation", "test"), required=True)
    evaluate.add_argument("--repeats", nargs="+", choices=("A", "B", "C"), default=("B", "C"))
    evaluate.add_argument("--patient"); evaluate.add_argument("--eye", choices=("L", "R"),
                      help="Fine-tuned run metadata selects patient/eye automatically when omitted")
    evaluate.add_argument("--device", default="auto"); evaluate.add_argument("--background-dilation-radius", type=int, default=2)
    evaluate.add_argument("--skip-existing", action="store_true"); evaluate.set_defaults(func=command_evaluate)
    report = commands.add_parser("report", help="Aggregate curves, distributions and repeatability")
    report.add_argument("--study", required=True); report.set_defaults(func=command_report)
    return root


def main(argv=None):
    args = parser().parse_args(argv); return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

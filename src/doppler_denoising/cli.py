"""Command-line interface for preparation, training, inference and evaluation."""

import argparse

from .common import load_sibling
from .evaluation.inference import denoise
from .evaluation.metrics import evaluate, summarize
from .preparation.core import prepare
from .training.runner import train


def _add_prepare_command(commands):
    """Define arguments used to locate and prepare recordings."""
    parser = commands.add_parser(
        "prepare",
        help="Find arterial peaks and brightness tables from an HDF5 dataset; cache selected input and preview",
    )
    parser.add_argument("--input", required=True, help="Dataset folder containing measurement folders; legacy AVI/NPY input also accepted")
    parser.add_argument("--output", required=True, help="Dataset mode: workflow root, writes prepared/; legacy mode: prepared destination")
    parser.add_argument("--measures", nargs="+", help="Dataset mode: selected measurement folder names; default all")
    parser.add_argument("--avi", action="store_true", help="Dataset mode: encode/decode MJPG before peak detection and training; default raw HDF5")
    parser.add_argument("--pattern", default="*", help="Folder mode: filename glob, e.g. '*.avi' or '*.npy'")
    parser.add_argument("--skip-existing", action="store_true", help="Folder mode: leave existing outputs untouched (not verified)")
    parser.add_argument("--fps", type=float, help="Effective M0 FPS; dataset default 37000/256 = 144.53125")
    parser.add_argument("--cx", type=int, default=255)
    parser.add_argument("--cy", type=int, default=255)
    parser.add_argument("--radius", type=int, default=260)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--peak-distance", type=int, default=15)
    parser.add_argument("--prominence", type=float, default=.10)
    parser.add_argument("--brightness-mask", help="Optional NPY mask for legacy vessel-mean brightness")
    parser.add_argument("--artery-mask", help="Manual retinal artery PNG/NPY used for robust peak timing; same geometry as input")
    parser.add_argument("--dataset-measure", help="For one AVI/NPY: measurement folder supplying manual/pseudo masks and report provenance")
    parser.add_argument(
        "--peak-method", choices=("auto", "arterial", "legacy"), default="auto",
        help="Auto uses arterial detection when --artery-mask is supplied, otherwise legacy detection",
    )
    parser.add_argument("--peak-min-hz", type=float, default=.5)
    parser.add_argument("--peak-max-hz", type=float, default=2.5)
    parser.set_defaults(func=prepare)


def _add_train_command(commands):
    """Define training and checkpoint-resume arguments."""
    parser = commands.add_parser("train", help="Train on one or multiple prepared records")
    parser.add_argument("--records", nargs="+", help="Prepared record directories and/or a folder containing prepared records")
    parser.add_argument("--config", help="JSON configuration; unspecified keys use defaults")
    parser.add_argument("--output", required=True, help="Dataset mode: workflow root; legacy mode: new experiment directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-epoch-previews", action="store_true", help="Skip the default end-of-epoch AVI export for every supplied record")
    parser.add_argument("--resume", action="store_true", help="Resume output/runs/last.pt (legacy: output/last.pt), from the last completed epoch")
    parser.add_argument("--epochs", type=int, help="Total epoch target, including already completed epochs")
    parser.add_argument("--no-epoch-metrics", action="store_true", help="Skip full-video regional diagnostics; loss history is always saved")
    parser.add_argument("--no-progress", action="store_true", help="Disable interactive batch progress bars (used by benchmark subprocesses)")
    parser.add_argument("--background-dilation-radius", type=int, default=2)
    parser.add_argument("--input", help="Dataset folder; use --output as workflow root (training writes runs/)")
    parser.add_argument("--measures", nargs="+")
    parser.set_defaults(func=train)


def _add_denoise_command(commands):
    """Define inference arguments."""
    parser = commands.add_parser("denoise", help="Infer from a checkpoint, without retraining")
    parser.add_argument("--checkpoint", help="Dataset mode default: output/runs/best.pt")
    parser.add_argument("--record")
    parser.add_argument("--input", help="Dataset folder; use --output as workflow root")
    parser.add_argument("--measures", nargs="+")
    parser.add_argument("--output", required=True, help="Parent folder for a measurement bundle; alternatively .npy or .avi output filename")
    parser.add_argument("--device", default="auto")
    parser.set_defaults(func=denoise)


def _add_evaluation_commands(commands):
    """Define per-recording evaluation and multi-recording summary arguments."""
    parser = commands.add_parser("evaluate", help="Calculate the article's per-recording metrics")
    parser.add_argument("--record")
    parser.add_argument("--denoised")
    parser.add_argument("--input", help="Dataset folder: auto-load manual retinal/pseudo choroidal masks; writes evaluation/")
    parser.add_argument("--measures", nargs="+")
    parser.add_argument("--checkpoint", help="Dataset mode default: output/runs/best.pt; infer missing denoised outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vessel-mask", help="Legacy single-region NPY or PNG mask")
    parser.add_argument("--background-mask", help="Background NPY or PNG for legacy single-region evaluation only")
    parser.add_argument("--retinal-artery-mask", "--retinal-artery", nargs="+", help="One or more artery masks, combined by union")
    parser.add_argument("--retinal-vein-mask", "--retinal-vein", nargs="+", help="One or more vein masks, combined by union")
    parser.add_argument("--choroidal-masks", "--choroidal-mask", nargs="+", help="One or more choroidal masks")
    parser.add_argument("--background-masks", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument(
        "--background-dilation-radius", type=int, default=2,
        help="Regional background: retinal dilation disk radius in prepared-image pixels (default: 2; 0 disables dilation)",
    )
    parser.add_argument("--cardiac-hz", type=float, help="Regional report: override common cardiac frequency")
    parser.add_argument("--local-size", type=int, default=32, help="Regional report: local grid patch width in pixels")
    parser.add_argument("--local-count", type=int, default=3, help="Regional report: maximum local patches per vessel group")
    parser.add_argument("--max-lag-seconds", type=float, default=.25)
    parser.add_argument("--profiles", help="Regional report: JSON mapping region to start/end [x,y] profile coordinates")
    parser.add_argument("--min-hz", type=float, default=.5)
    parser.add_argument("--max-hz", type=float, default=3.)
    parser.add_argument("--output", required=True, help="New report folder for regional masks; JSON file for legacy evaluation")
    parser.set_defaults(func=evaluate)

    parser = commands.add_parser("summarize", help="Aggregate metrics across recordings (mean and sample SD)")
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.set_defaults(func=summarize)


def _build_parser():
    """Build the complete parser from one small function per workflow stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    _add_prepare_command(commands)
    _add_train_command(commands)
    _add_denoise_command(commands)
    _add_evaluation_commands(commands)
    return parser


def _uses_dataset_mode(args):
    return args.command in ("train", "denoise", "evaluate") and args.input


def _run_dataset_stage(args, parser):
    """Delegate dataset layout discovery while keeping algorithms in stage modules."""
    legacy_inputs = (
        "record", "records", "denoised", "vessel_mask", "retinal_artery_mask",
        "retinal_vein_mask", "choroidal_masks", "background_mask", "background_masks",
    )
    if any(getattr(args, name, None) for name in legacy_inputs):
        parser.error("Dataset --input mode automatically resolves records and masks; do not mix legacy input flags")

    # Dataset orchestration receives the compatibility API because it also
    # supports older callers that inject this module in tests or scripts.
    from . import noise2time as api
    return load_sibling("dataset_workflow").run_stage(args, api) or 0


def _validate_legacy_inputs(args, parser):
    """Report missing arguments for the direct record-based interface."""
    if args.command == "train" and not args.records:
        parser.error("train requires --input dataset or --records")
    if args.command == "denoise" and (not args.record or not args.checkpoint):
        parser.error("denoise requires --input dataset or both --record and --checkpoint")
    if args.command == "evaluate" and (not args.record or not args.denoised):
        parser.error("evaluate requires --input dataset or both --record and --denoised")


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if _uses_dataset_mode(args):
        return _run_dataset_stage(args, parser)
    _validate_legacy_inputs(args, parser)
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())

# DopplerHolographyDenoising

Tools to prepare Doppler holography videos, train and benchmark Noise2Time
denoisers, export denoised videos, and evaluate retinal and choroidal signal
preservation.

This README is the project documentation. The maintained implementation is in
`src/doppler_denoising/`. Historical research scripts are preserved unchanged
in `legacy/` and are not imported by the maintained workflow.

## Installation

Python 3.10 or later is required. Install PyTorch for the CUDA version supported
by your machine, then install this repository:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

The commands installed by the package are:

| Command | Purpose |
|---|---|
| `dh-denoise` | Prepare, train, resume, denoise, evaluate, and summarize |
| `dh-benchmark` | Run one-factor-at-a-time training comparisons |
| `dh-collect` | Find and collect requested measurements from large directory trees |

Run `COMMAND --help` or `dh-denoise SUBCOMMAND --help` for every option.

## Repository layout

```text
configs/                     training and benchmark configurations
legacy/                      previous standalone research scripts
src/doppler_denoising/
  preparation/               HDF5/AVI loading, arterial peaks, previews, collection
  training/                  sampling, temporal splits, model, losses, training loop
  evaluation/                inference, regional metrics, plots, HTML reports
  benchmark/                 one-factor-at-a-time experiment runner and comparisons
  cli.py                     command definitions and workflow dispatch
  config.py                  validated training configuration
  common.py                  small shared utilities
  noise2time.py              compatibility imports for older Python callers
tests/                       synthetic and CPU integration tests
README.md                    complete project documentation
```

The stage packages follow the order of the workflow. Most changes should go in
the package that owns that stage. `noise2time.py` contains no algorithms; it
keeps the previous import surface working while scripts migrate to the focused
modules.

Generated datasets, checkpoints, videos, and reports are intentionally ignored
by Git.

## Dataset layout

The normal workflow expects one directory per measurement:

```text
dataset/
  MEASURE/
    MEASURE_DV.h5
    manual/
      retina_artery_mask.png
      retina_vein_mask.png
    pseudo/
      ...choroidal...vessel_mask_raw.png
```

Accepted retinal filenames use either `retina_*` or `retinal_*`. Exactly one
HDF5 file and one pseudo choroidal vessel mask are required. Manual choroidal
masks are deliberately ignored.

The HDF5 input must contain `doppler_signal/M0_ff`, or the older
`moment0ff`, in `(time, height, width)` order. Spatial dimensions must be
divisible by 32. Masks must already match the video geometry; the program never
resizes or registers them.

The default effective sampling rate is `37000 / 256 = 144.53125 Hz`. Supply
`--fps` when another rate applies.

## Collect measurements

Write exact measurement directory names in a UTF-8 text file, one per line.
Blank lines and lines starting with `#` are ignored.

```bash
dh-collect \
  --measures measures.txt \
  --folders Y:/folder1 Y:/folder2 \
  --output D:/collected \
  --max-depth 2
```

The collector uses bounded directory listings instead of an unrestricted
recursive traversal. `--max-depth 1` searches direct children only. Add
`--dry-run` to inspect matches without copying.

For every selected measurement, the collector copies:

- `MEASURE_HD_M0.avi`
- `MEASURE_version_holodoppler.txt`
- `MEASURE_parameters_holodoppler.json`

The metadata comes from `MEASURE/MEASURE_HD/version_holodoppler.txt` and
`MEASURE/MEASURE_HD/json/parameters_holodoppler.json`. The measurement prefix
keeps filenames unique in the flat collection folder. A missing video or
metadata file is reported and that measurement is not copied.

To extract the HDF5 `moment0ff` array losslessly instead of copying the AVI:

```bash
dh-collect --measures measures.txt --folders Y:/folder1 \
  --output D:/collected_h5 --h5 --frame-axis 0
```

## Prepare raw HDF5 measurements

```bash
dh-denoise prepare \
  --input D:/dataset_choroid2 \
  --output D:/N2T \
  --measures 251031_ALA_L_1 260310_AUZ0752_16
```

Preparation:

1. Reads `M0_ff` in bounded blocks.
2. Applies the circular diaphragm mask.
3. Divides the complete recording by one fixed maximum inside the diaphragm.
4. Extracts the manual retinal arterial mean.
5. Detects arterial peaks and artifacts.
6. Identifies complete peak-to-peak cycles.
7. Saves lossless float32 frames and viewable AVI/PNG/CSV diagnostics.

The prepared output is:

```text
OUTPUT/
  prepared/
    MEASURE/
      frames.npy
      prepared.avi
      roi.npy
      artery_mask.npy
      brightness.npy
      brightness_raw.npy
      brightness.csv
      brightness.png
      phase.npy
      cycle.npy
      fractional_phase.npy
      valid_frames.npy
      arterial_peaks.json
      arterial_peak_diagnostics.npz
      metadata.json
```

`prepared.avi` is a visualization. Training uses `frames.npy`.

### Compression experiment

Add `--avi` during dataset preparation to encode the normalized raw sequence
as MJPG and decode it before peak detection and training:

```bash
dh-denoise prepare --input D:/dataset_choroid2 --output D:/N2T_AVI --avi
```

This tests 8-bit quantization and MJPG compression together. Use a separate
output root so raw and compressed preparations cannot be mixed.

### Use one existing AVI

To denoise a particular existing AVI, link it to its measurement directory:

```bash
dh-denoise prepare \
  --input D:/dataset/MEASURE/specific.avi \
  --output D:/N2T_AVI/prepared/MEASURE \
  --dataset-measure D:/dataset/MEASURE
```

The AVI remains the actual input. All its frames are retained, and its container
FPS is used unless `--fps` is supplied. The linked directory supplies the
retinal and choroidal masks and the manual arterial signal. Metadata labels this
representation `source_avi`, distinct from the HDF5 compression experiment.

## Cardiac-phase matching

Each complete cycle is assigned a normalized phase from 0 to 1. For a target
frame at phase `u`, donor data are sampled at the same `u` in another cycle.
If that coordinate lies between two measured frames, only the required donor
pixels are linearly interpolated. The stored video and the target history are
never temporally resampled.

Both interpolation endpoints must:

- belong to the donor cycle;
- be valid and outside detected artifacts;
- obey the active temporal split;
- remain outside the input window for the no-patch strategy.

The input modes differ as follows. With the default `history = 9`, both
`patched` and `no_patch` receive ten frames, `t-9` through `t`:

- `patched` replaces blocks in input frame `t` and reconstructs the original
  values only at those block locations;
- `no_patch` leaves all ten input frames untouched and predicts a complete
  frame from another cardiac cycle at the same fractional phase as frame `t`;
- the donor used by `no_patch` must lie outside the ten-frame input window;
- `history_only` is the separate legacy ablation that excludes frame `t` and
  receives only the nine preceding frames.

`cycle.npy`, `fractional_phase.npy`, `brightness.csv`, and
`metadata.json` make this alignment auditable. Online generation avoids
storing every possible donor and patch combination.

## Train

Dataset mode automatically resolves prepared measurements and writes into
`OUTPUT/runs`:

```bash
dh-denoise train \
  --input D:/dataset_choroid2 \
  --output D:/N2T \
  --config configs/l2.json \
  --device cuda
```

Alternatively, train directly from one record or a prepared-record parent:

```bash
dh-denoise train \
  --records D:/N2T/prepared \
  --output D:/N2T/runs \
  --config configs/l2.json \
  --device cuda
```

Each epoch records training and validation loss components, saves `last.pt`,
updates `best.pt`, keeps inference weights under `checkpoints/`, and creates
metrics and a preview for the first measurement.

Resume an interrupted run with the same data and output:

```bash
dh-denoise train --input D:/dataset_choroid2 --output D:/N2T \
  --resume --epochs 100 --device cuda
```

`--epochs` is the total target, including completed epochs. The checkpoint
restores model, optimizer, early-stopping state, and random generators.
Prepared-data hashes and the split must remain unchanged. Checkpoints made with
the older integer-phase sampler can be used for inference but cannot be resumed
with the fractional-cycle sampler.

## Denoise

Dataset mode uses `runs/best.pt` unless another checkpoint is supplied:

```bash
dh-denoise denoise --input D:/dataset_choroid2 --output D:/N2T --device cuda
```

For a single prepared record:

```bash
dh-denoise denoise \
  --record D:/N2T/prepared/MEASURE \
  --checkpoint D:/N2T/runs/best.pt \
  --output D:/results \
  --device cuda
```

The result bundle contains:

```text
results/
  MEASURE/
    original.avi
    denoised.avi
    denoised.npy
    denoised.json
```

The NPY file is the scientific output. AVI files use a fixed display mapping.

## Evaluate

```bash
dh-denoise evaluate --input D:/dataset_choroid2 --output D:/N2T --device cuda
```

Evaluation automatically uses:

- the manual retinal artery mask;
- the manual retinal vein mask;
- the pseudo choroidal vessel mask;
- background defined inside the diaphragm as the complement of the union of
  the dilated retinal masks and choroidal mask.

The choroidal region is removed from retinal masks, and retinal regions are
removed from the choroidal mask. Reports include mask overlays, waveform plots,
frequency spectra, residual maps, local patches, HTML interpretation, JSON
metrics, and CSV tables.

With no clean reference, the results are diagnostics rather than accuracy
scores. Interpret them together:

- positive NRR means lower temporal background variability;
- vessel waveform correlation near 1 means shape is preserved;
- amplitude and waveform standard-deviation ratios near 1 mean pulsatility is
  preserved;
- residual pulsatility near 0 means little cardiac signal was removed. It is
  the RMS fitted amplitude at `f0`, `2*f0`, and `3*f0` in
  `original - denoised`, divided by the corresponding original RMS amplitude;
- mean ratios near 1 mean regional intensity is preserved;
- lag near zero argues against temporal displacement;
- residual cardiac energy or structured residual maps can reveal removed
  physiological signal.

Each regional report includes a compact `metrics_overview.png`. It shows the
background NRR beside waveform correlation, amplitude preservation, and
residual pulsatility for the three vessel groups. Per-measure benchmark
comparison folders contain the same dashboard with one series per strategy.

A constant output can produce excellent background NRR while destroying all
physiology, so NRR must never be interpreted alone.

## Benchmark strategies

The benchmark changes one factor at a time:

- baseline patched input;
- full-frame no-patch target: ten untouched inputs (`history + 1`), with a
  complete same-phase frame from another cycle as the target;
- temporal train/validation split;
- video-disjoint validation;
- U-Net without ConvLSTM;
- no brightness correction;
- L1 and composite loss variants.

```bash
dh-benchmark \
  --prepared D:/N2T/prepared \
  --output D:/N2T_benchmark \
  --config configs/l2.json \
  --experiments configs/benchmark_full.json \
  --device cuda
```

Use `configs/benchmark_single_video.json` when only one development recording
is available; it omits video-disjoint validation.

Resume after interruption:

```bash
dh-benchmark --output D:/N2T_benchmark --resume --device cuda
```

Evaluate saved checkpoints on a separate dataset without retraining:

```bash
dh-benchmark --output D:/N2T_benchmark --evaluate-only \
  --evaluation-input D:/independent_dataset --device cuda
```

Each strategy receives its own run, previews, reports, status, and logs.
`index.html` links the comparisons. Each measurement has a separate comparison
folder. Validation losses from different split strategies do not use the same
data and should not be compared as if they were identical tests.

## Configuration

The supplied training files are:

- `configs/l2.json`: practical L2 baseline;
- `configs/l1.json`: L1 reconstruction;
- `configs/article.json`: L1 reconstruction with gradient and Hessian terms.

Important fields include history length, ConvLSTM toggle, patch size/count,
objective, epoch/sample budget, batch size, learning rate, early-stopping
patience, split mode, input mode, and brightness correction. Benchmark variants
start from the selected baseline and alter one conceptual factor.

## Tests

The test suite uses synthetic data and does not require the private dataset:

```bash
python -m pytest
```

It checks peak detection, fractional phase matching, artifact exclusion,
temporal splits, losses, CPU training/resume, AVI linkage, mask rules, report
generation, benchmark orchestration, and bounded collection.

## Legacy scripts

`legacy/` contains the earlier standalone preprocessing and model scripts,
including `pretraitement_complet.py`, the original N2N variants, UDVD,
Blind2Unblind, and Neighbor2Neighbor. They are kept for traceability and may
contain hard-coded paths, duplicated code, older pairing logic, and assumptions
that differ from the maintained workflow. New experiments should use the
installed commands documented above.

## Reproducibility limits

- The implementation is deterministic for a fixed compatible PyTorch/CUDA
  environment, seed, preparation, and split.
- Preparation and run manifests hash scientific inputs.
- AVI is lossy; use NPY/HDF5 data for quantitative work unless compression is
  the variable under study.
- Peak detection is derived from the retinal arterial signal and is not an ECG
  reference. Inspect `brightness.png` and peak diagnostics before long runs.
- Report conclusions remain limited by the absence of clean ground truth.

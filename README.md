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
| `dh-scaling-study` | Patient-disjoint dataset-scaling and target fine-tuning study |

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

## Dataset-size and target-adaptation study

The `scaling_study/` package estimates how much patient diversity is needed
before target-video fine-tuning becomes useful. It treats the patient as the
independent experimental unit. Both eyes and all repeated acquisitions from a
patient always remain in the same pretraining, validation, or locked-test
partition.

For the 132-video repeatability dataset (22 patients, two eyes, three repeats),
the default protocol reserves 14 patients for nested pretraining subsets,
three for choosing fine-tuning settings, and five for the final locked test.
The nested sizes are 1, 2, 4, 8, 12, and 14 patients (6 to 84 videos), repeated
with three patient orders. Additional runs compare one versus three repeats
per eye at fixed patient counts. Repeat roles A/B/C come from the frozen
manifest order, not from inconsistent filename suffixes.

Create and audit the study before preparing any arrays:

```bash
dh-scaling-study plan \
  --dataset D:/dataset_DV_repeatable \
  --output D:/DV_denoising_scaling

dh-scaling-study prepare \
  --study D:/DV_denoising_scaling \
  --skip-existing
```

`plan` accepts either the folder containing measurement folders or its parent
when the measurements are under `retina/`. It writes `manifest.json`,
`manifest.csv`, `protocol.json`, `jobs.csv`, and `STUDY_SHA256.txt`. The source
videos are referenced in place and never copied. Review these files before the
first long run. The manifest records HDF5 identity, geometry, masks, patient,
eye, and the repeat assignment.

Run the short pilot first. It uses one subset order at 1, 4, and 14 patients:

```bash
dh-scaling-study pretrain \
  --study D:/DV_denoising_scaling \
  --config configs/l2.json \
  --device cuda \
  --pilot
```

Remove `--pilot` for the complete nested-subset experiment. Add `--resume` to
continue every incomplete run from `last.pt`. Training uses a fixed number of
optimizer steps at every dataset size. Each sample is drawn as patient, then
video within patient, then eligible frame, which prevents long recordings or
patients with more usable frames from receiving greater weight. Validation
uses 20 fixed anchors from every validation video. The default 1,750-step
reporting interval corresponds to the old 3,500-sample, batch-size-two epoch;
reports call it a *virtual epoch* and always retain the real global step.

Fine-tuning uses repeat A of each validation eye, with a disjoint temporal
block inside A for early stopping. Repeats B and C remain independent:

```bash
dh-scaling-study finetune \
  --study D:/DV_denoising_scaling \
  --config configs/l2.json \
  --device cuda
```

The stored grid compares learning rates 5e-6 and 1e-5, budgets 250, 500,
1,000, and 1,750 steps, and full-model versus decoder/temporal-only updates.
The zero-step case is the original pretrained checkpoint. Choose one policy
from validation patients before evaluating the locked test patients. Also
retain random-initialized same-budget, target-only, and temporal-average
controls listed in `protocol.json`; they distinguish transfer from simply
fitting the target recording.

Use `--random-init-control` to run the same target-video budgets without
pretrained weights. Once one learning rate, budget, and scope has been selected
using validation B/C results, apply only that policy to locked test patients:

```bash
dh-scaling-study finetune \
  --study D:/DV_denoising_scaling \
  --config configs/l2.json \
  --cohort test \
  --learning-rates 1e-5 \
  --step-budgets 500 \
  --scopes decoder_temporal \
  --device cuda
```

An evaluation command given a fine-tuned `best.pt` reads its `run.json` and
automatically limits B/C evaluation to the corresponding patient and eye. This
prevents a target-adapted model from being incorrectly scored as a general
model on other patients.

Evaluate a selected checkpoint on B/C repeats and then aggregate:

```bash
dh-scaling-study evaluate \
  --study D:/DV_denoising_scaling \
  --checkpoint D:/DV_denoising_scaling/pretraining/scale_o1_p14_r3/best.pt \
  --name scale_o1_p14_r3 \
  --cohort validation \
  --device cuda

dh-scaling-study evaluate \
  --study D:/DV_denoising_scaling \
  --checkpoint D:/DV_denoising_scaling/pretraining/scale_o1_p14_r3/best.pt \
  --name scale_o1_p14_r3 \
  --cohort test \
  --device cuda

dh-scaling-study report --study D:/DV_denoising_scaling
```

Evaluation exports the original and denoised AVI, lossless denoised NPY, and
compact JSON/CSV metrics per measurement. The consolidated report includes
training curves with variation across subset orders, held-out box plots with
visible outliers, ICC(2,1), within-subject CV, and Bland–Altman repeatability
limits for B versus C. It prioritizes off-harmonic temporal noise reduction,
background NRR, arterial waveform correlation, and arterial amplitude ratio.

The repeatability dataset currently has no `manual/small_vessels.png` masks.
The study records this as missing and evaluates artery, vein, choroid, and
background without inventing a mask. Add manually checked masks to validation
and test measurements before drawing conclusions about small-vessel
preservation, then create a new manifest so mask hashes remain auditable.

For publication, report optimizer steps, batch size, number of sampled windows,
patient-balanced sampling, evaluation cadence, and patient-level split. An
"epoch" alone is ambiguous here because the online sampler has no finite pass
through all possible windows.

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

## Published self-supervised baselines

The benchmark can also run UDVD, Blind2Unblind, and Neighbor2Neighbor on the
same prepared recordings and evaluate them with the same regional reports.
Their objectives, temporal context, provenance, configuration and limitations
are documented in [METHOD_BASELINES.md](METHOD_BASELINES.md). The ready-to-run
selection is `configs/benchmark_methods.json`.

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
- the handmade `manual/small_vessels.png` mask as an independent signal region;
- background defined inside the diaphragm as the complement of the union of
  the dilated retinal/small-vessel masks and choroidal mask.

The choroidal region is removed from retinal masks, and retinal regions are
removed from the choroidal mask. The small-vessel mask may overlap these
anatomical groups because it measures vessel size rather than vessel type.
Reports include mask overlays, waveform plots,
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
- positive off-harmonic noise reduction in dB means less temporal power away
  from `f0`, `2*f0`, and `3*f0`. Curves are linearly detrended and Hann
  windowed; frequencies below `f0/2` and bands of ±2 FFT bins around each
  harmonic are excluded. This is a fluctuation metric because non-periodic
  physiology can also occupy the retained frequencies;
- mean ratios near 1 mean regional intensity is preserved;
- lag near zero argues against temporal displacement;
- residual cardiac energy or structured residual maps can reveal removed
  physiological signal.

Each regional report includes a compact `metrics_overview.png`. It shows the
background NRR beside waveform correlation, amplitude preservation, and
residual pulsatility for the four signal regions. Per-measure benchmark
comparison folders contain the same dashboard with one series per strategy.

A constant output can produce excellent background NRR while destroying all
physiology, so NRR must never be interpreted alone.

## Benchmark combinations

The benchmark accepts explicit combinations of six independent categories:

- objective: `l1`, `l2`, `l1_grad_hessian`, or `l2_grad_hessian`;
- split: `random`, `per_cycle`, or `external_video`;
- frame pairing: `random`, `next`, or `cycle_phase`;
- patch policy: `none`, `random_patches`, `vessel_patches`, `black_patches`, or `patch_mean`;
- brightness correction: `none` or `on`;
- model: `unet`, `unet_convlstm`, or `unet_transformer`.

All maintained modes receive frames `t-history` through `t`. Without patches,
the paired frame is the full target and loss covers the diaphragm ROI.
`random_patches` places donor patches uniformly at random inside the diaphragm
and restricts loss to those squares. With `vessel_patches`, square patches are
centred on randomly selected vessel-mask
pixels, their contents come from the paired frame, and loss is restricted to
the selected squares. The vessel mask is the union of handmade retinal artery
and vein masks and the pseudo-choroidal mask, intersected with the diaphragm.
`black_patches` uses the same vessel-centred placement and restricted loss, but
sets the selected pixels to zero. It reconstructs the unmodified anchor frame,
so frame pairing and brightness correction do not alter its patch contents.
`patch_mean` places patches randomly inside the diaphragm, replaces each patch
with its own original mean, and also reconstructs the unmodified anchor only
inside those patches. It needs neither a donor nor vessel masks, so frame
pairing and brightness correction do not affect its contents.

`unet_transformer` reproduces the temporal-attention model from
`noisetotrans.py`: the current-frame bottleneck is the query and the preceding
frame bottlenecks are keys and values. Attention is computed independently at
each spatial bottleneck location with 64 query/key channels, and its softmax
weights are multiplied by 1.1 before fusion. It is a single temporal
cross-attention layer rather than a full multi-layer Transformer encoder.

`random` pairing chooses a valid frame outside the input window. `next` uses
frame `t+1`. `cycle_phase` uses linear interpolation at the same fractional
phase in another complete cardiac cycle. Brightness correction scales paired
content to the arterial brightness of frame `t`.

```bash
dh-benchmark \
  --prepared D:/N2T/prepared \
  --output D:/N2T_benchmark \
  --config configs/l2.json \
  --experiments configs/benchmark_combinations.json \
  --device cuda
```

The combination file has complete defaults and a list of named experiments.
Every experiment inherits the defaults and overrides any categories it needs.
Runs are explicit; the benchmark never expands a Cartesian product.

An external-video validation entry names complete prepared measurements:

```json
{
  "name": "external_validation",
  "split": {
    "strategy": "external_video",
    "validation_records": ["260622_LEC0430_L_4"]
  }
}
```

These recordings are excluded from training but used to select checkpoints.
They are distinct from a final independent dataset supplied with
`--evaluation-input`. If `validation_records` is omitted, `--validation-video`
is used; without that option, the last selected prepared recording is used.

Resume after interruption:

```bash
dh-benchmark --output D:/N2T_benchmark --resume --device cuda
```

Extend a completed benchmark to a new total epoch count:

```bash
dh-benchmark --output D:/N2T_benchmark/benchmark --resume --epochs 7 --device cuda
```

`--epochs` is the total, so a checkpoint that completed epoch 6 runs only epoch
7. Strategies already at the requested total are skipped. If the benchmark and
its `prepared` folder were moved together, missing stored paths are relinked to
the sibling `prepared` folder and verified against the hashes in `plan.json`.
Reports generated from an older best checkpoint are archived and refreshed.

Add newly named combinations to an existing benchmark while leaving completed
strategies untouched:

```bash
dh-benchmark --output D:/N2T_benchmark/benchmark --resume \
  --experiments configs/benchmark_combinations.json --device cuda
```

Existing names must still describe exactly the saved configurations. New
strategies inherit the benchmark's current total epoch count.

`configs/benchmark_methods.json` also includes `temporal_average`, a
training-free centered temporal mean. Its default window is 11 frames. Every
output frame uses the same number of samples; the time axis is reflected at
the two boundaries. Odd windows are exactly centered. For an even window, the
same past-heavy convention as the 12-frame structural reference is used (for
example, 12 means offsets -6 through +5). Change the window explicitly in the
method configuration:

```json
{
  "experiments": ["temporal_average"],
  "configs": {"temporal_average": {"window_size": 21}}
}
```

Add and evaluate it in an existing benchmark without training:

```bash
dh-benchmark --output D:/N2T_benchmark/benchmark --resume \
  --experiments configs/benchmark_methods.json \
  --strategies temporal_average --all-development \
  --evaluation-input D:/independent_dataset --device cuda
```

The analytic strategy creates an epoch-zero provenance checkpoint so its exact
configuration and implementation hash participate in the same cache and report
integrity checks as trained strategies. It has no loss curve or per-epoch
diagnostics.

Use `--exclude-measures NAME ...` when extending an existing benchmark to
persistently omit recordings from newly trained strategies, development
evaluation, comparisons, and aggregate reports. Existing checkpoints remain
unchanged and can still document the datasets on which they were trained.

Evaluate saved checkpoints on a separate dataset without retraining:

```bash
dh-benchmark --output D:/N2T_benchmark --evaluate-only \
  --evaluation-input D:/independent_dataset --device cuda
```

Evaluate one strategy on every development measurement and every measurement
in an external dataset:

```bash
dh-benchmark --output D:/N2T_benchmark/benchmark --evaluate-only \
  --strategies l2_black_patch_random_frames_unet_conv_lstm \
  --all-development --evaluation-input D:/independent_dataset --device cuda
```

Omitting `--evaluation-measures` selects every external measurement. Existing
compatible preparations and reports are reused.

Aggregate all completed development and unseen evaluations:

```bash
dh-aggregate --benchmark D:/N2T_benchmark/benchmark
```

This writes `aggregate/report.html` plus reusable CSV tables. The report has
per-variant box plots with every measurement and labeled Tukey outliers,
development-versus-unseen panels, diagnostic progression curves showing the
mean ± SD across measurements, and the training/validation loss curves. The SD
bands describe variation between measurements; estimating variation between
training runs requires repeated runs with different seeds.

Each strategy receives its own run, previews, reports, status, and logs.
`index.html` links the comparisons. Each measurement has a separate comparison
folder. Validation losses from different split strategies do not use the same
data and should not be compared as if they were identical tests.

## Configuration

The supplied training files are:

- `configs/l2.json`: practical L2 baseline;
- `configs/l1.json`: L1 reconstruction;
- `configs/article.json`: L1 reconstruction with gradient and Hessian terms;
- `configs/benchmark_combinations.json`: editable combination benchmark;
- `configs/benchmark_methods.json`: published learned baselines and the
  configurable temporal-average baseline.

Important shared fields include history length, channel count, patch size/count,
epoch/sample budget, batch size, learning rate, and early-stopping patience.
The combination file owns objective, split,
pairing, patch policy, brightness correction, and model selection. Legacy
experiment-list files remain readable for reproducing old benchmarks.

## Cycle-aligned pulse-shape preservation

Regional reports (schema `noise2time.regional.v6`) split every vessel waveform
into complete detector peak-to-peak cycles, resample each cycle to 128 phase
samples, and form a pointwise median. At least two complete cycles are required.
The median cycle is limited to harmonics 1 through 10 before shape and
derivative metrics are computed. Reports include:

- baseline- and amplitude-independent shape correlation and normalized RMSE;
- peak-to-peak amplitude, maximum periodic upstroke and total-variation ratios;
- preservation of power in harmonics 4 through 10;
- per-harmonic amplitude ratios for harmonics 1 through 10.

Each anatomical region writes `plots/<region>_pulse_shape.png` and a CSV with
the original/denoised median, interquartile ranges, normalized shapes and phase
derivatives. These endpoints are intended to expose temporal smoothing that can
retain the cardiac fundamental while removing a sharp systolic upstroke.

## Local contrast and 12-frame profile consistency

Regional reports include local vessel contrast
and cross-section agreement with a **12-frame uniform sliding average of the
original video**. No Gaussian averaging is used. For scored frame `t`, the
reference uses `t-6` through `t+5` inclusive, with centroid `t-0.5` and five future
frames. Only complete windows inside the scored sequence are evaluated; no
boundary padding or excluded startup frames enter the reference.

Local contrast is vessel mean minus nearby background mean. For each selected
local tile, nearby background is a disk dilation of its vessel mask (radius
`max(3, local_size // 2)`) intersected with the cleaned background mask. Thus other
annotated vessels and pixels outside the diaphragm are excluded. Empty local
backgrounds produce an explicit unavailable status. Contrast is measured in
the original input intensity units, without per-profile normalization. Regional
contrast summaries weight valid local tiles equally, not individual pixels.

For each automatic or `--profiles` cross-section, reports provide per-frame RMSE
against the same original-video average for both the original and denoised
profiles, plus mean/median RMSE and a displayed frame selected with seed 2026.
Coordinates and eligible timestamps are identical across methods. Bilinear
sample support must lie entirely inside the ROI; vessel/background trace
samples must additionally lie entirely inside their respective masks. The
vessel portion of each cross-section also receives the existing correlation,
cardiac-amplitude, residual-pulsatility and off-harmonic metrics. Cross-sections
without vessel or background samples remain explicitly unavailable where needed.

Outputs include regional `metrics.json`/`metrics.csv`, local contrast plots and
CSVs, saved local-background masks, and `plots/*_profile.npz` with reference
profiles and sample masks. `*_profile_metrics.csv` stores per-frame errors;
`*_profile_contrast.csv` and `*_profile_waveform.csv` store cross-section traces
when available. Benchmark comparison CSVs include only strategy-dependent
summary fields, and `dh-aggregate` summarizes regional contrast, pulse shape
and profile consistency. Input-only constants such as original local contrast,
12-frame local contrast and original-to-reference profile RMSE remain available
in detailed plot data when needed, but are omitted from comparison reports and
aggregate rankings. Existing reports need regeneration (use the existing
`--force-evaluation` option, which
archives older reports); training is not required to recompute these metrics.

The reference shares noise with the original and smooths physiological dynamics.
Lower profile error is agreement with this reference, not evidence of clean-signal
accuracy. Automatic local tiles favor mask occupancy; inspect manually specified
cross-sections when evaluating particular small vessels. These additions do not
turn the combined small-vessel mask into individually annotated vessel instances.

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

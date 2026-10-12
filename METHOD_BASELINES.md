# Published self-supervised method baselines

The benchmark supports `udvd`, `blind2unblind`, and `neighbor2neighbor` beside
Noise2Time. They use the same prepared float32 input, development/external
recordings, masks, report generator and checkpoint-selection rule. They change
the architecture and self-supervised target together, so they are method
comparisons rather than one-factor Noise2Time ablations.

## UDVD

[Unsupervised Deep Video Denoising](https://openaccess.thecvf.com/content/ICCV2021/html/Sheth_Unsupervised_Deep_Video_Denoising_ICCV_2021_paper.html)
uses a directional blind-spot network. The benchmark maps five frames
`(t-2, t-1, t, t+1, t+2)` to the noisy middle frame. The target pixel from the
middle frame cannot enter its own prediction through the spatial receptive
field. It is a non-causal method because it needs two future frames.

The architecture comes from the authors' [MIT-licensed repository](https://github.com/sreyas-mohan/udvd)
at commit `7374e5bafde865c3a7ef11c5543d5c0b19da40af`, adapted to one channel and the
common benchmark interface.

## Blind2Unblind

[Blind2Unblind](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Blind2Unblind_Self-Supervised_Image_Denoising_With_Visible_Blind_Spots_CVPR_2022_paper.html)
is framewise. A width-four periodic mapper evaluates 16 hidden pixel positions,
then assembles their predictions. Its loss combines the blind residual with a
detached fully visible residual. Beta stays at 2 for the first 40 percent of
training and rises linearly to 20 at the final epoch. Inference combines the
blind and visible estimates with the final beta.

The authors' [public repository](https://github.com/zejinwang/Blind2Unblind)
has no license file. This project therefore contains a clean-room implementation
of the paper equations and mask procedure rather than copied repository code.
Its grayscale U-Net backbone comes from the separately BSD-licensed
Neighbor2Neighbor repository.

## Neighbor2Neighbor

[Neighbor2Neighbor](https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Neighbor2Neighbor_Self-Supervised_Denoising_From_Single_Noisy_Images_CVPR_2021_paper.html)
selects a random directed pair of horizontal or vertical neighbors inside each
2 x 2 cell. One half-resolution image predicts the other. A detached
full-resolution prediction supplies the published consistency regularizer,
whose weight increases linearly to 2.

The network and objective are adapted from the authors'
[BSD-licensed repository](https://github.com/TaoHuang2018/Neighbor2Neighbor) at
commit `e66389ebef1e64d306d8fcb95f096cf427243452`.

## Run and resume

Use a new output directory for a new plan:

```bash
dh-benchmark \
  --prepared D:/N2T_benchmark_2/prepared \
  --output D:/N2T_method_benchmark \
  --config configs/l2.json \
  --experiments configs/benchmark_methods.json \
  --device cuda
```

To add the methods to a saved benchmark and keep its original strategies:

```bash
dh-benchmark \
  --output D:/existing_benchmark \
  --resume \
  --experiments configs/benchmark_methods.json \
  --strategies udvd blind2unblind neighbor2neighbor \
  --device cuda
```

Resume an interrupted plan without repeating completed strategies:

```bash
dh-benchmark --output D:/N2T_method_benchmark --resume --device cuda
```

An optional `configs` object overrides strict method settings:

```json
{
  "experiments": ["baseline", "udvd", "blind2unblind", "neighbor2neighbor"],
  "configs": {
    "udvd": {"epochs": 20, "batch_size": 2},
    "blind2unblind": {"epochs": 20, "batch_size": 2},
    "neighbor2neighbor": {"epochs": 20, "batch_size": 2}
  }
}
```

Unknown settings are rejected. Overriding `epochs` also extends the
Neighbor2Neighbor regularizer and Blind2Unblind beta schedules unless their
schedule duration is explicitly overridden.

## Interpretation

Do not compare the numerical training losses between methods: the objectives
and scales differ. Compare their common regional reports instead. Inspect
background variability together with arterial, venous, choroidal and
small-vessel waveform correlation, amplitude, lag and mean intensity.

Equal epochs and samples are also not equal computation. Blind2Unblind performs
16 masked views, UDVD processes five frames, and Neighbor2Neighbor trains on
half-resolution pairs. Record wall time and GPU memory if efficiency matters.

Doppler speckle may be spatially or temporally correlated, violating the
independence assumptions of every blind-spot method. Neighbor2Neighbor may also
smooth vessel boundaries. Freeze settings using development recordings, then
evaluate the selected configurations once on unseen recordings.


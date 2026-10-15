"""Regional Noise2Time evaluation and a self-contained local HTML report.

All numerical analysis uses unclipped arrays. Display clipping is explicitly
separate from analysis. No model, training, or network access is required here.
"""
from pathlib import Path
import csv
import html
import json
import hashlib

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import colormaps

PRIMARY_REGIONS = ("retinal_artery", "retinal_vein", "choroidal")
REGIONS = (*PRIMARY_REGIONS, "small_vessels")
LABELS = {"retinal_artery":"Retinal arteries", "retinal_vein":"Retinal veins",
          "choroidal":"Choroidal vessels", "small_vessels":"Small vessels",
          "background":"Background"}
COLORS = {"retinal_artery":"#e34a33", "retinal_vein":"#3182bd",
          "choroidal":"#a661c2", "small_vessels":"#e6ab02", "background":"#31a354"}
EPS = 1e-12
AVERAGE_FRAMES = 12
PROFILE_SEED = 2026
PULSE_PHASE_SAMPLES = 128
PULSE_MAX_HARMONIC = 10
PULSE_HIGH_HARMONIC_START = 4


def sliding_average_reference(values):
    """Return complete 12-frame means and their scored-frame indices.

    At t use [t-6, t+5], inclusive: centroid t-0.5, five future frames.
    No padding, clipping, or use of excluded startup frames. Averaging traces
    or fixed bilinear profiles is equivalent to averaging the image first.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or not np.isfinite(values).all():
        raise ValueError("Sliding average requires finite time-first values")
    indices = np.arange(6, len(values)-5, dtype=int)
    if len(values) < AVERAGE_FRAMES:
        return indices, np.empty((0, *values.shape[1:]), dtype=np.float64)
    sums = np.concatenate((np.zeros_like(values[:1]), np.cumsum(values, axis=0)), axis=0)
    return indices, (sums[AVERAGE_FRAMES:]-sums[:-AVERAGE_FRAMES])/AVERAGE_FRAMES


def local_background(mask, background, radius):
    """Nearby background only; the supplied background excludes all vessels."""
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    kernel = (xx*xx+yy*yy <= radius*radius).astype(np.uint8)
    nearby = cv2.dilate(mask.astype(np.uint8), kernel,
                        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return nearby & background


def contrast_metrics(vessel, background):
    """Local intensity difference on common, complete-window timestamps."""
    if background is None:
        return dict(status="unavailable", reason="no_local_background", frames_scored=0), None
    contrast = np.asarray(vessel, float)-np.asarray(background, float)
    indices, reference = sliding_average_reference(contrast[0])
    if not len(indices):
        return dict(status="unavailable", reason="fewer_than_12_scored_frames", frames_scored=0), None
    original, denoised = contrast[:, indices]
    means = [float(x.mean()) for x in (original, denoised, reference)]
    result = dict(status="ok", frames_scored=len(indices),
                  original_mean=means[0], denoised_mean=means[1], sliding_average_mean=means[2],
                  mean_change=means[1]-means[0], denoised_to_original_ratio=ratio(means[1], means[0]),
                  rmse_original_vs_sliding_average=float(np.sqrt(np.mean((original-reference)**2))),
                  rmse_denoised_vs_sliding_average=float(np.sqrt(np.mean((denoised-reference)**2))))
    return result, (indices, original, denoised, reference)


def save_contrast_csv(path, data, fps, first_original_frame):
    if data is None:
        return
    indices, original, denoised, reference = data
    np.savetxt(path, np.column_stack((indices, indices+first_original_frame, indices/fps,
                                     original, denoised, reference)), delimiter=",",
               header="scored_frame,original_frame,seconds_since_scored_start,original_contrast,denoised_contrast,sliding_average_contrast",
               comments="")


def profile_consistency(original, denoised, valid_samples):
    """Unnormalized profile RMSE against the same original-video reference."""
    indices, reference = sliding_average_reference(original)
    if not np.any(valid_samples):
        return dict(status="unavailable", reason="no_eligible_profile_samples", frames_scored=0), None
    if not len(indices):
        return dict(status="unavailable", reason="fewer_than_12_scored_frames", frames_scored=0), None
    errors = [np.sqrt(np.mean((x[indices][:, valid_samples]-reference[:, valid_samples])**2, axis=1))
              for x in (original, denoised)]
    selected = int(np.random.default_rng(PROFILE_SEED).choice(len(indices)))
    return dict(status="ok", frames_scored=len(indices), samples=int(np.sum(valid_samples)),
                rmse_original_mean=float(errors[0].mean()), rmse_denoised_mean=float(errors[1].mean()),
                rmse_original_median=float(np.median(errors[0])), rmse_denoised_median=float(np.median(errors[1])),
                selected_scored_frame=int(indices[selected]),
                selected_rmse_original=float(errors[0][selected]), selected_rmse_denoised=float(errors[1][selected]),
                selection_seed=PROFILE_SEED), (indices, reference, *errors)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def source_sha256():
    """Hash this evaluator so reports cannot silently mix metric definitions."""
    with Path(__file__).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def derive_background(raw, roi, radius):
    """Dilate retinal/small-vessel masks, then negate them with choroid."""
    if not isinstance(radius, int) or radius < 0:
        raise ValueError("Background dilation radius must be a nonnegative integer")
    retinal = raw["retinal_artery"] | raw["retinal_vein"] | raw["small_vessels"]
    if radius:
        yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
        kernel = (xx*xx + yy*yy <= radius*radius).astype(np.uint8)
        retinal = cv2.dilate(retinal.astype(np.uint8), kernel,
                             borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return roi & ~(retinal | raw["choroidal"])


def exclusive_masks(raw, roi):
    """Clean anatomical groups while retaining small-vessel overlap."""
    clipped = {name:np.asarray(mask, bool) & roi for name,mask in raw.items()}
    vessel_union = np.logical_or.reduce([clipped[name] for name in REGIONS])
    cleaned, counts = {}, {}
    for name in PRIMARY_REGIONS:
        others = np.logical_or.reduce([clipped[other] for other in PRIMARY_REGIONS if other != name])
        cleaned[name] = clipped[name] & ~others
    # Small vessels are a size-based analysis region. Their overlap with an
    # artery, vein, or choroidal label is meaningful and remains selected.
    cleaned["small_vessels"] = clipped["small_vessels"].copy()
    cleaned["background"] = clipped["background"] & ~vessel_union
    for name in (*REGIONS, "background"):
        counts[name] = dict(input_pixels=int(raw[name].sum()),
                            outside_roi_pixels=int((raw[name] & ~roi).sum()),
                            excluded_overlap_pixels=int((clipped[name] & ~cleaned[name]).sum()),
                            evaluated_pixels=int(cleaned[name].sum()))
        if not cleaned[name].any():
            raise ValueError(f"{name} has no pixels left after ROI/overlap exclusion: {counts[name]}")
    primary_union = np.logical_or.reduce([clipped[name] for name in PRIMARY_REGIONS])
    excluded = primary_union & ~np.logical_or.reduce([cleaned[name] for name in PRIMARY_REGIONS])
    return cleaned, counts, excluded


def ratio(a, b):
    return float(a/b) if abs(b) > EPS else None


def correlation(a, b):
    # Float32 std can be nonzero for a constant trace due to mean rounding,
    # while corrcoef's float64 covariance is exactly zero (and returns NaN).
    a, b = np.asarray(a,dtype=np.float64), np.asarray(b,dtype=np.float64)
    if min(np.std(a), np.std(b)) <= EPS:
        return None
    return float(np.clip(np.corrcoef(a,b)[0,1], -1, 1))


def sinusoid(curve, fps, frequency):
    t = np.arange(len(curve))/fps
    design = np.column_stack((np.ones(len(t)),np.cos(2*np.pi*frequency*t),np.sin(2*np.pi*frequency*t)))
    fit = np.linalg.lstsq(design,curve,rcond=None)[0]
    return float(np.hypot(fit[1],fit[2]))


def residual_pulsatility(before, after, fps, frequency, orders=(1, 2, 3)):
    """Measure cardiac harmonics left in ``original - denoised``.

    A sinusoid is fitted at f0 and each requested harmonic. The aggregate
    ratio is the RMS harmonic amplitude in the residual divided by the RMS
    harmonic amplitude in the original signal. It is phase-sensitive because
    the residual is formed before fitting. Zero is ideal; values above one are
    possible when denoising strongly distorts or amplifies pulsatility.
    """
    components = []
    for order in orders:
        harmonic_frequency = order * frequency
        if harmonic_frequency >= fps / 2:
            continue
        original_amplitude = sinusoid(before, fps, harmonic_frequency)
        denoised_amplitude = sinusoid(after, fps, harmonic_frequency)
        residual_amplitude = sinusoid(before-after, fps, harmonic_frequency)
        components.append(dict(
            order=order,
            frequency_hz=harmonic_frequency,
            amplitude_original=original_amplitude,
            amplitude_denoised=denoised_amplitude,
            amplitude_ratio=ratio(denoised_amplitude, original_amplitude),
            residual_amplitude=residual_amplitude,
            residual_amplitude_ratio=ratio(residual_amplitude, original_amplitude),
        ))
    original_power = sum(item["amplitude_original"]**2 for item in components)
    residual_power = sum(item["residual_amplitude"]**2 for item in components)
    power_ratio = ratio(residual_power, original_power)
    return dict(
        residual_pulsatility_ratio=None if power_ratio is None else float(np.sqrt(power_ratio)),
        residual_pulsatility_power_ratio=power_ratio,
        harmonics=components,
        definition="sqrt(sum(A_residual,k^2) / sum(A_original,k^2)), k=1..3 below Nyquist",
    )


def off_harmonic_noise(before, after, fps, frequency, orders=(1,2,3), half_width_bins=2):
    """Measure temporal power outside the cardiac fundamental and harmonics.

    A linear trend is removed, both curves receive the same Hann window, and
    bins within ``half_width_bins`` FFT bins of f0, 2f0 and 3f0 are excluded.
    Positive dB means the denoised curve contains less off-harmonic power.
    This is a fluctuation diagnostic: non-periodic physiology may also occur
    outside the excluded bands and must not automatically be called noise.
    """
    before,after=np.asarray(before,float),np.asarray(after,float)
    if (before.shape!=after.shape or before.ndim!=1 or len(before)<4
            or not np.isfinite([fps,frequency]).all() or fps<=0 or not 0<frequency<fps/2
            or not isinstance(half_width_bins,int) or half_width_bins<1):
        raise ValueError("Invalid curves, sampling rate, cardiac frequency or band width")
    n=len(before);coordinate=np.linspace(-1,1,n)
    design=np.column_stack((np.ones(n),coordinate));window=np.hanning(n)
    powers=[]
    for curve in (before,after):
        detrended=curve-design@np.linalg.lstsq(design,curve,rcond=None)[0]
        powers.append(np.abs(np.fft.rfft(detrended*window))**2)
    frequencies=np.fft.rfftfreq(n,1/fps);resolution=float(fps/n)
    # Very slow changes are drift rather than the frame-to-frame fluctuations
    # this metric is meant to summarize.  Start halfway below the cardiac
    # fundamental, then remove the cardiac bands themselves below.
    minimum_frequency=max(resolution,frequency/2)
    selected=frequencies>=minimum_frequency
    bands=[];half_width=half_width_bins*resolution
    for order in orders:
        center=order*frequency
        if center>=fps/2: continue
        low=max(0.,center-half_width);high=min(fps/2,center+half_width)
        selected &= ~((frequencies>=low)&(frequencies<=high))
        bands.append(dict(order=order,center_hz=float(center),low_hz=float(low),high_hz=float(high)))
    original=float(powers[0][selected].sum());denoised=float(powers[1][selected].sum())
    power_ratio=ratio(denoised,original)
    reduction=None if power_ratio is None else float(1-power_ratio)
    reduction_db=(None if original<=EPS
                  else float(10*np.log10(original/max(denoised,EPS))))
    return dict(off_harmonic_power_original=original,
                off_harmonic_power_denoised=denoised,
                off_harmonic_power_ratio=power_ratio,
                off_harmonic_noise_reduction=reduction,
                off_harmonic_noise_reduction_db=reduction_db,
                off_harmonic_frequency_resolution_hz=resolution,
                off_harmonic_minimum_frequency_hz=float(minimum_frequency),
                off_harmonic_excluded_half_width_hz=float(half_width),
                off_harmonic_excluded_bands=bands,
                off_harmonic_bin_count=int(selected.sum()),
                off_harmonic_definition=(
                    "10*log10(P_original/P_denoised) after linear detrending and a Hann window; "
                    "P sums bins >=f0/2 outside +/-2 FFT bins of f0, 2f0 and 3f0"))


def peak_frequency(peaks, fps, minimum_intervals=2):
    """Estimate cardiac frequency from the arterial peak detector.

    Peak spacing is in the original frame coordinate system.  Using the median
    interval is robust to one missed or extra peak and avoids the coarse FFT
    grid that can select a subharmonic on short recordings.
    """
    values = np.asarray(peaks, dtype=float)
    intervals = np.diff(values)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if len(intervals) < minimum_intervals:
        return None, dict(intervals_frames=intervals.tolist(), reason="too_few_peak_intervals")
    period = float(np.median(intervals))
    frequency = float(fps / period)
    return frequency, dict(intervals_frames=intervals.tolist(), median_period_frames=period,
                          period_mad_frames=float(np.median(np.abs(intervals-period))),
                          frequency_hz=frequency)


def frequency_crosscheck(curve, fps, frequency, minimum_hz=.5, maximum_hz=3.):
    """Report the strongest FFT bin near the detector-derived frequency."""
    freq, amplitude = spectrum(curve, fps)
    candidates = np.flatnonzero((freq >= minimum_hz) & (freq <= maximum_hz))
    if not len(candidates):
        return dict(strongest_fft_hz=None, strongest_fft_amplitude=None, fft_resolution_hz=fps/len(curve))
    index = int(candidates[np.argmax(amplitude[candidates])])
    return dict(strongest_fft_hz=float(freq[index]), strongest_fft_amplitude=float(amplitude[index]),
                fft_resolution_hz=float(fps/len(curve)),
                detector_frequency_bin_offset_hz=float(freq[index]-frequency))


def waveform_metrics(before, after, fps, frequency, max_lag_seconds):
    mean_before, mean_after = float(before.mean()), float(after.mean())
    a, b = sinusoid(before,fps,frequency), sinusoid(after,fps,frequency)
    limit = min(int(round(max_lag_seconds*fps)),len(before)//4)
    candidates = []
    for lag in range(-limit,limit+1):
        x,y = (before[:-lag],after[lag:]) if lag>0 else ((before[-lag:],after[:lag]) if lag<0 else (before,after))
        value = correlation(x,y)
        if value is not None:
            candidates.append((value,lag))
    best = max(candidates,key=lambda item:(item[0],-abs(item[1]))) if candidates else None
    pulsatility = residual_pulsatility(before, after, fps, frequency)
    off_harmonic = off_harmonic_noise(before, after, fps, frequency)
    return dict(temporal_correlation=correlation(before,after),
                vessel_mean_original=mean_before,vessel_mean_denoised=mean_after,
                mean_change=mean_after-mean_before,
                mean_change_percent=None if abs(mean_before)<=EPS else 100*(mean_after/mean_before-1),
                cardiac_frequency_hz=frequency,amplitude_original=a,amplitude_denoised=b,
                amplitude_ratio=ratio(b,a),
                residual_cardiac_amplitude=sinusoid(before-after,fps,frequency),
                lag_frames=best[1] if best else None,
                lag_ms=1000*best[1]/fps if best else None,
                lag_corrected_correlation=best[0] if best else None,
                lag_limit_seconds=max_lag_seconds,
                **pulsatility,**off_harmonic)


def _phase_resampled_cycles(curve, peaks, scored_start, samples):
    """Return complete peak-to-peak cycles on one common phase grid.

    Peak coordinates use the prepared-video frame system. ``scored_start`` is
    therefore subtracted before indexing the scored waveform. Appending the
    first cycle sample at phase one makes interpolation explicitly periodic.
    """
    curve = np.asarray(curve, dtype=np.float64)
    target = np.arange(samples, dtype=np.float64) / samples
    cycles = []
    for start, stop in zip(peaks[:-1], peaks[1:]):
        start, stop = int(start)-scored_start, int(stop)-scored_start
        if start < 0 or stop > len(curve) or stop-start < 4:
            continue
        values = curve[start:stop]
        source = np.arange(len(values), dtype=np.float64) / len(values)
        cycles.append(np.interp(target, np.r_[source, 1.0], np.r_[values, values[0]]))
    return np.asarray(cycles, dtype=np.float64)


def _harmonic_limited_cycle(values, maximum_harmonic):
    """Remove cycle-average noise above the harmonics being assessed."""
    coefficients = np.fft.rfft(np.asarray(values, dtype=np.float64))
    coefficients[maximum_harmonic+1:] = 0
    return np.fft.irfft(coefficients, n=len(values))


def pulse_shape_metrics(before, after, peaks, scored_start, samples=PULSE_PHASE_SAMPLES,
                        maximum_harmonic=PULSE_MAX_HARMONIC):
    """Compare robust, phase-aligned cardiac pulse shapes.

    Complete peak-to-peak cycles are resampled to a common phase grid. Their
    pointwise median is then limited to harmonics 1..``maximum_harmonic`` so a
    few noisy cycles do not make derivative metrics unstable. Shape
    correlation and normalized RMSE ignore baseline and overall amplitude;
    the other ratios separately test amplitude, upstroke, total variation and
    higher-harmonic preservation.
    """
    before, after = np.asarray(before, float), np.asarray(after, float)
    if before.shape != after.shape or before.ndim != 1:
        raise ValueError("Pulse-shape curves must be matching one-dimensional arrays")
    if samples < 16 or maximum_harmonic < PULSE_HIGH_HARMONIC_START or maximum_harmonic >= samples//2:
        raise ValueError("Invalid pulse-shape phase samples or harmonic limit")
    original_cycles = _phase_resampled_cycles(before, peaks, scored_start, samples)
    denoised_cycles = _phase_resampled_cycles(after, peaks, scored_start, samples)
    if len(original_cycles) < 2 or len(original_cycles) != len(denoised_cycles):
        return dict(status="unavailable", reason="fewer_than_two_complete_cycles",
                    cycle_count=int(len(original_cycles))), None

    original_median = np.median(original_cycles, axis=0)
    denoised_median = np.median(denoised_cycles, axis=0)
    original_shape = _harmonic_limited_cycle(original_median, maximum_harmonic)
    denoised_shape = _harmonic_limited_cycle(denoised_median, maximum_harmonic)
    original_centered = original_shape-original_shape.mean()
    denoised_centered = denoised_shape-denoised_shape.mean()
    original_rms = float(np.sqrt(np.mean(original_centered**2)))
    denoised_rms = float(np.sqrt(np.mean(denoised_centered**2)))
    if min(original_rms, denoised_rms) <= EPS:
        return dict(status="unavailable", reason="constant_cycle_median",
                    cycle_count=int(len(original_cycles))), None
    original_normalized = original_centered/original_rms
    denoised_normalized = denoised_centered/denoised_rms
    normalized_rmse = float(np.sqrt(np.mean((original_normalized-denoised_normalized)**2)))

    # Periodic central derivatives include the upstroke that can cross the
    # peak-to-peak boundary. Multiplication by samples gives intensity per
    # unit cycle phase and cancels in the reported ratio.
    derivatives = [samples*(np.roll(values, -1)-np.roll(values, 1))/2
                   for values in (original_shape, denoised_shape)]
    peak_to_peak = [float(np.ptp(values)) for values in (original_shape, denoised_shape)]
    upstroke = [float(np.max(values)) for values in derivatives]
    total_variation = [float(np.sum(np.abs(np.roll(values, -1)-values)))
                       for values in (original_shape, denoised_shape)]

    spectra = [np.fft.rfft(values-values.mean()) for values in
               (original_median, denoised_median)]
    harmonic_stop = min(maximum_harmonic, len(spectra[0])-1)
    high_slice = slice(PULSE_HIGH_HARMONIC_START, harmonic_stop+1)
    high_power = [float(np.sum(np.abs(values[high_slice])**2)) for values in spectra]
    harmonic_ratios = []
    for order in range(1, harmonic_stop+1):
        amplitudes = [float(np.abs(values[order])) for values in spectra]
        harmonic_ratios.append(dict(order=order, ratio=ratio(amplitudes[1], amplitudes[0])))

    result = dict(
        status="ok", cycle_count=int(len(original_cycles)), phase_samples=int(samples),
        maximum_harmonic=int(maximum_harmonic),
        cycle_shape_correlation=correlation(original_shape, denoised_shape),
        cycle_normalized_rmse=normalized_rmse,
        cycle_peak_to_peak_ratio=ratio(peak_to_peak[1], peak_to_peak[0]),
        cycle_upstroke_slope_ratio=ratio(upstroke[1], upstroke[0]),
        cycle_total_variation_ratio=ratio(total_variation[1], total_variation[0]),
        cycle_high_harmonic_power_ratio=ratio(high_power[1], high_power[0]),
        harmonic_amplitude_ratios=harmonic_ratios,
        definition=("Median of complete peak-to-peak cycles resampled to a common phase grid; "
                    f"shape and derivative metrics use harmonics 1..{maximum_harmonic}; "
                    f"high-harmonic power uses {PULSE_HIGH_HARMONIC_START}..{maximum_harmonic}"),
    )
    data = dict(
        phase=np.arange(samples, dtype=float)/samples,
        original_cycles=original_cycles, denoised_cycles=denoised_cycles,
        original_median=original_median, denoised_median=denoised_median,
        original_shape=original_shape, denoised_shape=denoised_shape,
        original_normalized=original_normalized, denoised_normalized=denoised_normalized,
        original_derivative=derivatives[0], denoised_derivative=derivatives[1],
    )
    return result, data


def local_regions(mask, size, count):
    """Non-overlapping grid patches, ordered by original mask occupancy."""
    tiles = []
    for y in range(0,mask.shape[0],size):
        for x in range(0,mask.shape[1],size):
            n = int(mask[y:y+size,x:x+size].sum())
            if n:
                tiles.append((n,y,x))
    result = []
    for n,y,x in sorted(tiles,key=lambda z:(-z[0],z[1],z[2]))[:count]:
        selected = np.zeros_like(mask)
        selected[y:y+size,x:x+size] = mask[y:y+size,x:x+size]
        result.append((selected,dict(x=x,y=y,width=min(size,mask.shape[1]-x),
                                     height=min(size,mask.shape[0]-y),pixels=n)))
    return result


def phase_bins(n, copied_prefix, peaks):
    """Four fractional-cycle bins; discard intervals outside complete peak pairs."""
    labels = np.full(n,-1,dtype=np.int8)
    for start,end in zip(peaks[:-1],peaks[1:]):
        if end <= start:
            continue
        positions = np.arange(max(start,copied_prefix),min(end,copied_prefix+n))
        if len(positions):
            labels[positions-copied_prefix] = np.minimum(3,4*(positions-start)//(end-start))
    return labels


def accumulate(original, denoised, masks, labels, chunk=8):
    """Bounded frame buffers; accumulate pixel statistics in float64."""
    n,h,w = original.shape
    total = np.zeros((2,h,w),np.float64)
    squares = np.zeros_like(total)
    phases = np.zeros((4,h,w),np.float64)
    phase_count = np.zeros(4,np.int64)
    curves = {name:np.empty((2,n),np.float64) for name in masks}
    for start in range(0,n,chunk):
        stop = min(start+chunk,n)
        a = np.asarray(original[start:stop],dtype=np.float64)
        b = np.asarray(denoised[start:stop],dtype=np.float64)
        for i,block in enumerate((a,b)):
            # Merge centered batch moments to avoid cancellation for flat pixels.
            block_mean = block.mean(axis=0)
            delta = block_mean-total[i]
            squares[i] += np.square(block-block_mean).sum(axis=0) + delta**2*start*len(block)/stop
            total[i] += delta*len(block)/stop
            for name,mask in masks.items():
                curves[name][i,start:stop] = block[:,mask].mean(axis=1)
        for phase in range(4):
            selected = labels[start:stop] == phase
            if selected.any():
                phases[phase] += (a[selected]-b[selected]).sum(axis=0)
                phase_count[phase] += selected.sum()
    mean = total
    std = np.sqrt(np.maximum(0,squares/n))
    for phase in range(4):
        if phase_count[phase]:
            phases[phase] /= phase_count[phase]
    return mean,std,curves,phases,phase_count


def new_figure(*args,**kwargs):
    figure = Figure(*args,constrained_layout=True,**kwargs)
    FigureCanvasAgg(figure)
    return figure


def save_figure(figure, path):
    figure.savefig(path,dpi=130)
    figure.clear()


def save_metric_dashboard(path, entries, title):
    """Plot the small set of metrics needed to judge useful denoising.

    ``entries`` contains one item per denoising strategy. Keeping this helper
    independent of CSV column layout lets single reports and benchmark
    comparisons use exactly the same visual language.
    """
    if not entries:
        raise ValueError("Metric dashboard requires at least one entry")
    labels = [entry["label"] for entry in entries]
    colors = [colormaps["tab10"](i % 10) for i in range(len(entries))]
    regions = list(REGIONS)
    region_labels = [LABELS[name].replace("Retinal ", "") for name in regions]
    figure = new_figure(figsize=(16, 8))
    axes = figure.subplots(2, 3)

    def annotate(ax, bars, suffix=""):
        values = [bar.get_height() for bar in bars]
        text = ["" if not np.isfinite(value) else f"{value:.1f}{suffix}" for value in values]
        ax.bar_label(bars, labels=text, padding=3, fontsize=8)

    # Background noise removal is useful only when the three signal panels
    # remain close to their ideal values.
    nrr = np.asarray([np.nan if entry.get("background_nrr") is None else entry["background_nrr"]
                      for entry in entries], float) * 100
    bars = axes[0, 0].bar(labels, nrr, color=colors)
    annotate(axes[0, 0], bars, "%")
    axes[0, 0].set(title="Background noise reduction", ylabel="NRR (%)")
    axes[0, 0].axhline(0, color="#555555", lw=.8)

    x = np.arange(len(regions), dtype=float)
    width = .8 / len(entries)
    for index, (entry, color) in enumerate(zip(entries, colors)):
        offset = (index - (len(entries)-1)/2) * width
        correlations = [entry["regions"].get(name, {}).get("temporal_correlation") for name in regions]
        amplitudes = [entry["regions"].get(name, {}).get("amplitude_ratio") for name in regions]
        residuals = [entry["regions"].get(name, {}).get("residual_pulsatility_ratio") for name in regions]
        pulse_shape = [entry["regions"].get(name, {}).get("pulse_shape", {}).get(
            "cycle_shape_correlation") for name in regions]
        correlations = [np.nan if value is None else value for value in correlations]
        amplitudes = [np.nan if value is None else value for value in amplitudes]
        residuals = [np.nan if value is None else 100*value for value in residuals]
        pulse_shape = [np.nan if value is None else value for value in pulse_shape]
        axes[0, 1].bar(x+offset, correlations, width, label=entry["label"], color=color)
        axes[0, 2].bar(x+offset, amplitudes, width, color=color)
        axes[1, 0].bar(x+offset, residuals, width, color=color)
        axes[1, 2].bar(x+offset, pulse_shape, width, color=color)
        off_harmonic=[entry.get("background_off_harmonic_noise_reduction_db")]
        off_harmonic += [entry["regions"].get(name,{}).get("off_harmonic_noise_reduction_db")
                         for name in regions]
        off_harmonic=[np.nan if value is None else value for value in off_harmonic]
        axes[1, 1].bar(np.arange(len(off_harmonic))+offset,off_harmonic,width,color=color)

    for ax in (axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 2]):
        ax.set_xticks(x, region_labels)
    axes[0, 1].set(title="Waveform shape preservation", ylabel="Temporal correlation")
    axes[0, 1].set_ylim(-.05, 1.05)
    axes[0, 2].set(title="Pulsation amplitude preservation", ylabel="Denoised / original")
    axes[0, 2].axhline(1, color="#555555", ls="--", lw=1, label="Ideal")
    axes[1, 0].set(title="Pulsatility left in removed signal", ylabel="Residual pulsatility ratio (%)")
    axes[1, 0].axhline(0, color="#555555", lw=.8)
    axes[1, 1].set(title="Off-harmonic temporal-power reduction",ylabel="Reduction (dB)",
                   xticks=np.arange(len(regions)+1),xticklabels=("Background",*region_labels))
    axes[1, 1].axhline(0,color="#555555",lw=.8)
    axes[1, 2].set(title="Cycle-aligned pulse shape", ylabel="Shape correlation", ylim=(-.05,1.05))
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.2)
    if len(entries) > 1:
        axes[0, 1].legend(fontsize=8, frameon=False)
    figure.suptitle(title)
    save_figure(figure, path)


def limit(values):
    return max(float(np.percentile(np.abs(values),99)),1e-6)


def spectrum(curve,fps):
    window = np.hanning(len(curve))
    return np.fft.rfftfreq(len(curve),1/fps), 2*np.abs(np.fft.rfft((curve-curve.mean())*window))/window.sum()


def save_waveforms(folder,name,curves,fps,frequency,first,offset):
    before,after = curves
    t = np.arange(len(before))/fps
    table = np.column_stack((np.arange(len(t))+first,np.arange(len(t))+first+offset,
                            t,before,after,before-after))
    np.savetxt(folder/f"{name}_waveform.csv",table,delimiter=",",comments="",
               header="prepared_frame,original_frame,seconds_since_scored_start,original,denoised,residual")
    figure = new_figure(figsize=(11,8))
    axes = figure.subplots(3,1)
    for ax in axes[:2]:
        ax.plot(t,before,label="Original",color="#777777",lw=1)
        ax.plot(t,after,label="Denoised",color="#007c91",lw=1)
        ax.set_ylabel("Mean intensity")
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[0].set_title(LABELS.get(name,name))
    axes[1].set_xlim(0,min(t[-1],3/frequency))
    axes[1].set_title("First three nominal cardiac cycles; same intensity scale")
    axes[1].set_ylim(axes[0].get_ylim())
    axes[2].plot(t,before-after,color="#b34b35",lw=1)
    axes[2].axhline(0,color="black",lw=.5)
    axes[2].set_ylabel("Original - denoised")
    axes[2].set_xlabel("Time since first scored frame (s)")
    save_figure(figure,folder/f"{name}_waveform.png")
    figure = new_figure(figsize=(10,4))
    ax = figure.subplots()
    for curve,label,color in ((before,"Original","#777777"),(after,"Denoised","#007c91"),
                              (before-after,"Residual","#b34b35")):
        f,a = spectrum(curve,fps)
        ax.plot(f,a,label=label,color=color,lw=1)
    ax.axvspan(0,frequency/2,color="#777777",alpha=.08)
    for k in (1,2,3):
        if k*frequency < fps/2:
            ax.axvline(k*frequency,color="#333333",ls="--",alpha=.4)
            half_width=2*fps/len(before)
            ax.axvspan(max(0,k*frequency-half_width),min(fps/2,k*frequency+half_width),
                       color="#d9a441",alpha=.12)
    ax.set(xlim=(0,fps/2),xlabel="Frequency (Hz)",ylabel="Amplitude (Hann window)",
           title=f"{LABELS.get(name,name)}: spectrum; shaded cardiac bands excluded from off-harmonic power")
    ax.legend(); ax.grid(alpha=.2)
    save_figure(figure,folder/f"{name}_spectrum.png")


def save_pulse_shape(folder, name, curves, peaks, scored_start):
    """Save pulse-shape metrics, aligned-cycle samples and an explanatory plot."""
    metrics, data = pulse_shape_metrics(*curves, peaks, scored_start)
    figure = new_figure(figsize=(11, 10))
    axes = figure.subplots(3, 1)
    if data is None:
        for axis in axes:
            axis.text(.5, .5, metrics["reason"], ha="center", va="center",
                      transform=axis.transAxes)
            axis.set_axis_off()
        figure.suptitle(f"{LABELS.get(name,name)}: pulse-shape preservation")
        save_figure(figure, folder/f"{name}_pulse_shape.png")
        return metrics

    phase = data["phase"]
    quantiles = {}
    for source in ("original", "denoised"):
        cycles = data[f"{source}_cycles"]
        quantiles[source] = np.percentile(cycles, [25, 75], axis=0)
    np.savetxt(
        folder/f"{name}_pulse_shape.csv",
        np.column_stack((phase, data["original_median"], quantiles["original"].T,
                         data["denoised_median"], quantiles["denoised"].T,
                         data["original_normalized"], data["denoised_normalized"],
                         data["original_derivative"], data["denoised_derivative"])),
        delimiter=",", comments="",
        header=("cycle_phase,original_median,original_q25,original_q75,"
                "denoised_median,denoised_q25,denoised_q75,"
                "original_normalized,denoised_normalized,"
                "original_phase_derivative,denoised_phase_derivative"),
    )
    colors = {"original":"#777777", "denoised":"#007c91"}
    for source, label in (("original", "Original"), ("denoised", "Denoised")):
        axes[0].plot(phase, data[f"{source}_median"], color=colors[source], label=label)
        axes[0].fill_between(phase, *quantiles[source], color=colors[source], alpha=.18)
        axes[1].plot(phase, data[f"{source}_normalized"], color=colors[source], label=label)
        axes[2].plot(phase, data[f"{source}_derivative"], color=colors[source], label=label)
    axes[0].set(title=f"Median and interquartile range across {metrics['cycle_count']} complete cycles",
                ylabel="Mean intensity")
    axes[1].set(title="Baseline- and amplitude-normalized median pulse (harmonics 1-10)",
                ylabel="Normalized intensity")
    axes[2].set(title="Periodic phase derivative (harmonics 1-10)",
                xlabel="Peak-to-peak cycle phase", ylabel="Intensity / cycle")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.grid(alpha=.2)
        axis.legend(frameon=False)
    figure.suptitle(f"{LABELS.get(name,name)}: pulse-shape preservation")
    save_figure(figure, folder/f"{name}_pulse_shape.png")
    return metrics


def profile_coordinates(mask, size):
    y,x = np.nonzero(mask)
    center = np.array([x.mean(),y.mean()])
    normal = np.array([1.,0.])
    if len(x)>2:
        _,vectors = np.linalg.eigh(np.cov(np.stack((x,y))))
        normal = vectors[:,0]  # Perpendicular to major axis of selected mask patch.
    distance = np.arange(-size//2,size//2+1,dtype=float)
    points = center[:,None]+normal[:,None]*distance
    keep = (points[0]>=0)&(points[0]<=mask.shape[1]-1)&(points[1]>=0)&(points[1]<=mask.shape[0]-1)
    return distance[keep],points[:,keep]


def save_profile(folder,name,local_mask,original,denoised,fps,size,selected_frame,override=None,
                 *, vessel_mask=None, background_mask=None, roi=None, frequency=None,
                 max_lag_seconds=.25, first_original_frame=0):
    if override is None:
        distance,points = profile_coordinates(local_mask,size)
        origin = "Automatic perpendicular to mask-patch major axis; inspect placement"
    else:
        start,end = np.asarray(override["start"],float),np.asarray(override["end"],float)
        if start.shape != (2,) or end.shape != (2,) or not np.isfinite([start,end]).all():
            raise ValueError("Profile start/end must be finite [x,y] coordinates")
        length = float(np.linalg.norm(end-start))
        if length < 1 or length > 2*max(local_mask.shape):
            raise ValueError("Invalid profile length")
        points = np.linspace(start,end,int(np.ceil(length))+1).T
        if np.any(points<0) or np.any(points[0]>local_mask.shape[1]-1) or np.any(points[1]>local_mask.shape[0]-1):
            raise ValueError("Profile lies outside image")
        distance = np.linspace(0,length,points.shape[1])
        origin = "User-supplied line"
    samples = []
    for frames in (original,denoised):
        samples.append(np.stack([map_coordinates(frame,[points[1],points[0]],order=1,mode="nearest") for frame in frames]))
    a,b = (np.asarray(values,dtype=np.float64) for values in samples)
    # Require all bilinear support to belong to the relevant mask. Nearest-mask
    # selection could include outside-ROI zeros or neighboring vessel pixels.
    def on_mask(mask):
        if mask is None:
            return np.zeros(len(distance), bool)
        return map_coordinates(mask.astype(float), [points[1],points[0]],
                               order=1,mode="nearest") >= 1-1e-9
    eligible = np.ones(len(distance),bool) if roi is None else on_mask(roi)
    vessel_samples = on_mask(local_mask if vessel_mask is None else vessel_mask) & eligible
    background_samples = on_mask(background_mask) & eligible
    consistency, comparison = profile_consistency(a,b,eligible)
    vessel_consistency, _ = profile_consistency(a,b,vessel_samples)
    indices, average = sliding_average_reference(a)
    if comparison is not None:
        selected_frame = consistency["selected_scored_frame"]
        np.savetxt(folder/f"{name}_profile_metrics.csv",
                   np.column_stack((indices,indices+first_original_frame,indices/fps,*comparison[2:])),
                   delimiter=",",header="scored_frame,original_frame,seconds_since_scored_start,rmse_original_vs_sliding_average,rmse_denoised_vs_sliding_average",comments="")
    temporal = dict(status="unavailable",reason="no_vessel_samples")
    contrast = dict(status="unavailable",reason="no_vessel_samples",frames_scored=0)
    if vessel_samples.any():
        vessel_curves = np.stack((a[:,vessel_samples].mean(1),b[:,vessel_samples].mean(1)))
        if frequency is not None:
            temporal = dict(status="ok",**waveform_metrics(*vessel_curves,fps,frequency,max_lag_seconds))
        bg_curves = (np.stack((a[:,background_samples].mean(1),b[:,background_samples].mean(1)))
                     if background_samples.any() else None)
        contrast, contrast_data = contrast_metrics(vessel_curves,bg_curves)
        save_contrast_csv(folder/f"{name}_profile_contrast.csv",contrast_data,fps,first_original_frame)
        np.savetxt(folder/f"{name}_profile_waveform.csv",
                   np.column_stack((np.arange(len(a))/fps,*vessel_curves)),delimiter=",",
                   header="seconds_since_scored_start,original_vessel_mean,denoised_vessel_mean",comments="")
    np.savez(folder/f"{name}_profile.npz",distance_pixels=distance,x=points[0],y=points[1],
             original=a,denoised=b,sliding_average=average,sliding_average_scored_frames=indices,
             eligible_samples=eligible,vessel_samples=vessel_samples,background_samples=background_samples)
    figure = new_figure(figsize=(12,8))
    axes = figure.subplots(2,3)
    axes[0,0].imshow(original[selected_frame],cmap="gray",vmin=0,vmax=1)
    axes[0,0].plot(points[0],points[1],color="#ffb000")
    axes[0,0].set_title(f"{LABELS[name]}: profile location")
    axes[0,0].axis("off")
    for ax,curves,title in ((axes[0,1],(a[selected_frame],b[selected_frame]),"Seeded matched frame (complete window)" if comparison is not None else "Frame (12-frame reference unavailable)"),
                             (axes[0,2],(a.mean(0),b.mean(0)),"Temporal mean (structural reference)")):
        ax.plot(distance,curves[0],color="#777777",label="Original")
        ax.plot(distance,curves[1],color="#007c91",label="Denoised")
        ax.set(title=title,xlabel="Distance (pixels)",ylabel="Intensity")
        ax.legend()
    if comparison is not None:
        position = int(np.flatnonzero(indices==selected_frame)[0])
        axes[0,1].plot(distance,average[position],color="#d18b00",label="Original 12-frame average")
        axes[0,1].legend()
    extent = [distance[0],distance[-1],(len(a)-1)/fps,0]
    for ax,values,title in ((axes[1,0],a,"Original"),(axes[1,1],b,"Denoised")):
        im = ax.imshow(values,aspect="auto",extent=extent,cmap="gray",vmin=0,vmax=1)
        ax.set(title=title,xlabel="Distance (pixels)",ylabel="Time (s)")
        figure.colorbar(im,ax=ax)
    scale = limit(a-b)
    im = axes[1,2].imshow(a-b,aspect="auto",extent=extent,cmap="RdBu_r",vmin=-scale,vmax=scale)
    axes[1,2].set(title="Signed residual",xlabel="Distance (pixels)",ylabel="Time (s)")
    figure.colorbar(im,ax=axes[1,2])
    figure.suptitle(origin)
    save_figure(figure,folder/f"{name}_profile.png")
    return dict(start=points[:,0].tolist(),end=points[:,-1].tolist(),placement=origin,
                interpolation="bilinear, same coordinates for all videos; full mask support required",
                selected_scored_frame=selected_frame,selected_original_frame=selected_frame+first_original_frame,
                vessel_samples=int(vessel_samples.sum()),background_samples=int(background_samples.sum()),
                consistency=consistency,vessel_consistency=vessel_consistency,
                local_contrast=contrast,temporal_metrics=temporal)


def write_comparison(path,original,denoised,fps,scale):
    h,w = original.shape[1:]
    # Pad rather than crop if externally supplied arrays have odd dimensions.
    out_h,out_w = h+h%2,4*w+(4*w)%2
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),fps,(out_w,out_h),True)
    try:
        if not writer.isOpened():
            raise RuntimeError("Cannot create comparison AVI")
        cmap = colormaps["RdBu_r"]
        # Keep one rolling image rather than materializing a full float64
        # averaged video.  The average uses the same complete [t-6, t+5]
        # windows as the structural metrics.  Boundary panels stay blank so
        # the comparison retains one output frame per scored input frame.
        rolling = (np.asarray(original[:AVERAGE_FRAMES],dtype=np.float64).sum(axis=0)
                   if len(original)>=AVERAGE_FRAMES else None)
        for index,(before,after) in enumerate(zip(original,denoised)):
            average = None
            if rolling is not None and 6<=index<len(original)-5:
                average = rolling/AVERAGE_FRAMES
            frames = (before,np.zeros_like(before) if average is None else average,after)
            gray = [cv2.cvtColor(np.rint(np.clip(frame,0,1)*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
                    for frame in frames]
            residual = (cmap(np.clip((before-after)/(2*scale)+.5,0,1))[...,:3]*255).astype(np.uint8)[...,::-1]
            canvas = np.zeros((out_h,out_w,3),np.uint8)
            canvas[:h,:4*w] = np.concatenate((*gray,residual),axis=1)
            for panel,label in enumerate(("Original","Original 12-frame average",
                                          "Denoised","Original - denoised")):
                cv2.putText(canvas,label,(panel*w+5,18),cv2.FONT_HERSHEY_SIMPLEX,.4,
                            (255,255,255),1,cv2.LINE_AA)
            writer.write(canvas)
            if average is not None and index<len(original)-6:
                rolling += np.asarray(original[index+6],dtype=np.float64)
                rolling -= np.asarray(original[index-6],dtype=np.float64)
    finally:
        writer.release()


def build_report(record,restored,metadata,raw_masks,provenance,args,output):
    """Write into a new staging folder; caller publishes it on successful return."""
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    plots = output/"plots"; plots.mkdir()
    mask_dir = output/"masks"; mask_dir.mkdir()
    masks,counts,excluded = exclusive_masks(raw_masks,record.roi)
    for name,mask in masks.items():
        np.save(mask_dir/f"{name}.npy",mask)
        ok,encoded = cv2.imencode(".png",mask.astype(np.uint8)*255)
        if not ok: raise RuntimeError("Mask PNG encoding failed")
        encoded.tofile(mask_dir/f"{name}.png")
    first = int(metadata.get("excluded_prefix", metadata.get("copied_prefix", -1)))
    if first < 0 or first >= len(record.frames):
        raise ValueError("Invalid excluded prefix")
    original,denoised = record.frames[first:],restored[first:]
    n = len(original); fps = float(record.metadata["fps"])
    if n<4 or not np.isfinite(fps) or not 0<args.min_hz<args.max_hz<fps/2:
        raise ValueError("Need >=4 scored frames and 0 < min_hz < max_hz < Nyquist")
    if args.local_size<3 or args.local_count<1 or not np.isfinite(args.max_lag_seconds) or args.max_lag_seconds<0:
        raise ValueError("Invalid local-region size/count or lag bound")
    locals_by_region = {name:local_regions(masks[name],args.local_size,args.local_count) for name in REGIONS}
    all_masks = dict(masks)
    local_backgrounds = {}
    contrast_radius = max(3,args.local_size//2)
    for name,items in locals_by_region.items():
        for i,(mask,_) in enumerate(items):
            key = f"{name}_local_{i+1}"
            all_masks[key] = mask
            nearby = local_background(mask,masks["background"],contrast_radius)
            local_backgrounds[key] = nearby
            if nearby.any():
                all_masks[key+"_background"] = nearby
            np.save(mask_dir/f"{key}_background.npy",nearby)
    labels = phase_bins(n,first,record.metadata.get("peaks",[]))
    print("Computing regional waveforms and spatial statistics...",flush=True)
    mean,std,curves,phases,phase_count = accumulate(original,denoised,all_masks,labels)
    reference = curves["retinal_artery"][0]
    detected_frequency, peak_diagnostics = peak_frequency(record.metadata.get("peaks", []), fps)
    if args.cardiac_hz is not None:
        frequency = float(args.cardiac_hz)
        if not np.isfinite(frequency) or not 0<frequency<fps/2:
            raise ValueError("--cardiac-hz must be finite, positive and below Nyquist")
        frequency_source = "User-supplied; detector-derived frequency retained as cross-check"
    else:
        if detected_frequency is None:
            raise ValueError("Cannot estimate cardiac frequency from at least two arterial peak intervals; supply --cardiac-hz")
        if not args.min_hz <= detected_frequency <= args.max_hz:
            raise ValueError(f"Detector-derived cardiac frequency {detected_frequency:.5g} Hz falls outside [{args.min_hz}, {args.max_hz}] Hz")
        frequency = detected_frequency
        frequency_source = "Median interval of arterial detector peaks"
    frequency_diagnostics = dict(peak_detector=peak_diagnostics,
                                 arterial_fft=frequency_crosscheck(reference, fps, frequency, args.min_hz, args.max_hz),
                                 pooled_vessel_fft=frequency_crosscheck(sum(curves[name][0]*masks[name].sum() for name in PRIMARY_REGIONS)/sum(masks[name].sum() for name in PRIMARY_REGIONS), fps, frequency, args.min_hz, args.max_hz))
    bg = [float(s[masks["background"]].mean()) for s in std]
    background = dict(pixels=int(masks["background"].sum()),background_std_original=bg[0],
                      background_std_denoised=bg[1],NRR=1-bg[1]/bg[0] if bg[0]>EPS else None,
                      pixel_sd_original_median=float(np.median(std[0][masks["background"]])),
                      pixel_sd_denoised_median=float(np.median(std[1][masks["background"]])),
                      pixel_sd_original_p95=float(np.percentile(std[0][masks["background"]],95)),
                      pixel_sd_denoised_p95=float(np.percentile(std[1][masks["background"]],95)))
    background.update(off_harmonic_noise(*curves["background"],fps,frequency))
    results = {}
    for name in REGIONS:
        print(f"Creating {LABELS[name]} metrics and plots...",flush=True)
        before,after = curves[name]
        metrics = waveform_metrics(before,after,fps,frequency,args.max_lag_seconds)
        metrics.update(pixels=int(masks[name].sum()),
                       local_regions=[],**{k:v for k,v in background.items()
                                           if k!="pixels" and not k.startswith("off_harmonic")})
        for key,frame in (("original",mean[0]),("denoised",mean[1])):
            metrics[f"vessel_background_contrast_{key}"] = float(frame[masks[name]].mean()-frame[masks["background"]].mean())
        for i,(_,bounds) in enumerate(locals_by_region[name]):
            key = f"{name}_local_{i+1}"
            local = waveform_metrics(*curves[key],fps,frequency,args.max_lag_seconds)
            contrast, contrast_data = contrast_metrics(curves[key],curves.get(key+"_background"))
            contrast.update(background_pixels=int(local_backgrounds[key].sum()),
                            background_radius_pixels=contrast_radius)
            save_contrast_csv(plots/f"{key}_contrast.csv",contrast_data,fps,
                              record.metadata["first_original_frame"]+first)
            metrics["local_regions"].append(dict(name=key,bounds=bounds,local_contrast=contrast,**local))
        available = [item["local_contrast"] for item in metrics["local_regions"]
                     if item["local_contrast"]["status"]=="ok"]
        metrics["local_contrast_valid_regions"] = len(available)
        metrics["local_contrast_denoised"] = (float(np.mean([item["denoised_mean"] for item in available]))
                                               if available else None)
        metrics["local_contrast_change"] = (float(np.mean([item["mean_change"] for item in available]))
                                             if available else None)
        metrics["pulse_shape"] = save_pulse_shape(
            plots, name, curves[name], record.metadata.get("peaks", []), first)
        results[name] = metrics
        save_waveforms(plots,name,curves[name],fps,frequency,first,record.metadata["first_original_frame"])
        figure = new_figure(figsize=(11,3*len(locals_by_region[name])))
        axes = np.atleast_1d(figure.subplots(len(locals_by_region[name]),1))
        for i,ax in enumerate(axes):
            key = f"{name}_local_{i+1}"
            for curve,label,color in zip(curves[key],("Original","Denoised"),("#777777","#007c91")):
                ax.plot(np.arange(n)/fps,curve,label=label,color=color,lw=1)
            ax.set(title=f"Local {i+1}: {locals_by_region[name][i][1]}",xlabel="Time (s)",ylabel="Mean intensity")
            ax.legend()
            np.savetxt(plots/f"{key}.csv",np.column_stack((np.arange(n)/fps,*curves[key])),delimiter=",",
                       header="seconds_since_scored_start,original,denoised",comments="")
        save_figure(figure,plots/f"{name}_local.png")
        figure = new_figure(figsize=(11,3*len(locals_by_region[name])))
        axes = np.atleast_1d(figure.subplots(len(locals_by_region[name]),1))
        for i,ax in enumerate(axes):
            key = f"{name}_local_{i+1}"
            contrast, data = contrast_metrics(curves[key],curves.get(key+"_background"))
            if data is None:
                ax.text(.5,.5,contrast["reason"],ha="center",transform=ax.transAxes)
            else:
                for trace,label,color in zip(data[1:],("Original","Denoised","Original 12-frame average"),
                                             ("#777777","#007c91","#d18b00")):
                    ax.plot(data[0]/fps,trace,label=label,color=color,lw=1)
                ax.legend()
            ax.set(title=f"Local {i+1}: vessel minus nearby background",xlabel="Time (s)",ylabel="Contrast (input units)")
        save_figure(figure,plots/f"{name}_contrast.png")
    # Masks and automatic local patches, using a shared background image.
    figure = new_figure(figsize=(15,8)); axes = figure.subplots(2,3)
    mask_names=(*REGIONS,"background")
    for ax,name in zip(axes.flat,mask_names):
        ax.imshow(mean[0],cmap="gray",vmin=0,vmax=1)
        overlay = np.zeros((*record.roi.shape,4))
        from matplotlib.colors import to_rgba
        overlay[masks[name]] = to_rgba(COLORS[name],.6)
        ax.imshow(overlay)
        for i,(_,bounds) in enumerate(locals_by_region.get(name,[])):
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((bounds["x"],bounds["y"]),bounds["width"],bounds["height"],fill=False,edgecolor="yellow"))
            ax.text(bounds["x"],bounds["y"],str(i+1),color="yellow")
        ax.set_title(f"{LABELS[name]}: {masks[name].sum()} pixels"); ax.axis("off")
    axes[1,2].imshow(excluded,cmap="gray",vmin=0,vmax=1)
    axes[1,2].set_title("Ambiguous anatomical pixels excluded"); axes[1,2].axis("off")
    save_figure(figure,plots/"masks.png")
    # Spatial metrics use full time series; color limits are for display only.
    mean_residual = mean[0]-mean[1]
    reduction = np.full(record.roi.shape,np.nan)
    usable = record.roi & (std[0]>EPS)
    reduction[usable] = 1-std[1][usable]/std[0][usable]
    np.savez(output/"spatial_maps.npz",mean_original=mean[0],mean_denoised=mean[1],mean_residual=mean_residual,
             std_original=std[0],std_denoised=std[1],std_reduction= reduction,
             phase_residual=phases,phase_frame_counts=phase_count)
    figure = new_figure(figsize=(12,8)); axes = figure.subplots(2,3)
    residual_scale = limit(mean_residual[record.roi]); std_max = max(float(std[:,record.roi].max()),1e-6)
    for ax,values,title,cmap,low,high in zip(axes.flat,
            (mean[0],mean[1],mean_residual,std[0],std[1],reduction),
            ("Original mean","Denoised mean","Mean signed residual","Original temporal SD","Denoised temporal SD","SD reduction (display clipped [-1,1])"),
            ("gray","gray","RdBu_r","magma","magma","RdBu"),
            (0,0,-residual_scale,0,0,-1),(1,1,residual_scale,std_max,std_max,1)):
        im = ax.imshow(np.ma.masked_where(~record.roi,values),cmap=cmap,vmin=low,vmax=high)
        ax.set_title(title); ax.axis("off"); figure.colorbar(im,ax=ax,shrink=.7)
    save_figure(figure,plots/"spatial_maps.png")
    phase_scale = limit(phases[:,record.roi])
    figure = new_figure(figsize=(12,4)); axes = figure.subplots(1,4)
    for i,ax in enumerate(axes):
        if phase_count[i]:
            im = ax.imshow(np.ma.masked_where(~record.roi,phases[i]),cmap="RdBu_r",vmin=-phase_scale,vmax=phase_scale)
            figure.colorbar(im,ax=ax,shrink=.65)
        else:
            ax.text(.5,.5,"No complete-cycle frames",ha="center",transform=ax.transAxes)
        ax.set_title(f"Cycle phase {i/4:.2f}-{(i+1)/4:.2f}\nn={phase_count[i]}"); ax.axis("off")
    figure.suptitle("Mean signed residual by fractional peak-to-peak phase; common symmetric scale")
    save_figure(figure,plots/"phase_residuals.png")
    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8")) if args.profiles else {}
    if set(profiles)-set(REGIONS): raise ValueError("Unknown region in profiles JSON")
    for name in REGIONS:
        selected = int(np.argmax(curves[name][0]))
        results[name]["profile"] = save_profile(plots,name,locals_by_region[name][0][0],original,denoised,
                                               fps,args.local_size,selected,profiles.get(name),
                                               vessel_mask=masks[name],background_mask=masks["background"],
                                               roi=record.roi,frequency=frequency,max_lag_seconds=args.max_lag_seconds,
                                               first_original_frame=record.metadata["first_original_frame"]+first)
        results[name]["profile_rmse_denoised_vs_sliding_average"] = results[name]["profile"]["consistency"].get("rmse_denoised_mean")
    sampled = np.linspace(0,n-1,min(20,n),dtype=int)
    video_scale = limit((np.asarray(original[sampled])-np.asarray(denoised[sampled]))[:,record.roi])
    print("Writing synchronized comparison video...",flush=True)
    write_comparison(output/"comparison.avi",original,denoised,fps,video_scale)
    result = dict(schema="noise2time.regional.v6",metric_protocol="noise2time_metric_audit_v6",
                  evaluator_sha256=source_sha256(),record=record.name,frames_scored=n,excluded_prefix=first,
                  fps=fps,first_original_scored_frame=int(record.metadata["first_original_frame"]+first),
                  cardiac_frequency_hz=frequency,frequency_source=frequency_source,
                  frequency_resolution_hz=fps/n,frequency_diagnostics=frequency_diagnostics,
                  background=background,regions=results,mask_counts=counts,
                  mask_sources=provenance,denoised_provenance=metadata,
                  phase_frame_counts=phase_count.tolist(),
                  phase_frames_excluded=int((labels<0).sum()),comparison_residual_display_limit=video_scale,
                  mask_policy="Artery/vein/choroid are mutually exclusive; small vessels are an independent overlapping region; background = ROI & ~(dilated original retinal/small-vessel union | original choroidal); choroidal not dilated",
                  lag_convention="Positive lag means denoised waveform is delayed; integer-frame search within configured bound",
                  local_selection="Highest-occupancy disjoint mask grid tiles; inspect and do not infer small-vessel performance from these alone",
                  structural_protocol=dict(sliding_average_frames=AVERAGE_FRAMES,
                      offsets_inclusive=[-6,5],centroid_offset_frames=-.5,uses_future_frames=True,
                      boundary_policy="Only complete windows within scored frames; no padding",
                      contrast="Vessel mean minus nearby background mean, shared masks and timestamps",
                      local_background="Disk dilation of local vessel mask intersected with cleaned global background",
                      local_background_radius_pixels=contrast_radius,
                      region_contrast_summary="Equal-weight mean of valid local-tile mean contrasts, complete-window times only",
                      profile_error="RMSE on ROI-supported cross-section samples versus original 12-frame average; mean over complete-window times",
                      profile_selection_seed=PROFILE_SEED,
                      reference_caveat="Temporal average shares input noise and blurs dynamics; consistency is not accuracy"),
                   pulse_shape_protocol=dict(
                       alignment="Complete detector peak-to-peak cycles",
                       aggregation="Pointwise median after resampling every cycle to 128 phase samples",
                       shape_filter="Fourier harmonics 1 through 10",
                       high_harmonic_band="Harmonics 4 through 10",
                       minimum_complete_cycles=2,
                       interpretation="Correlation and normalized RMSE test shape independently of baseline and amplitude; ratios should remain near one"),
                  caveat="No clean reference. Excluding mask overlaps does not unmix retinal/choroidal signals. Metrics describe changes, not denoising accuracy. Cardiac frequency is detector-derived; FFT values are cross-checks.")
    save_metric_dashboard(
        plots/"metrics_overview.png",
        [dict(label="Denoised", background_nrr=background["NRR"],
              background_off_harmonic_noise_reduction_db=background["off_harmonic_noise_reduction_db"],
              regions=results)],
        f"{record.name}: denoising summary",
    )
    save_json(output/"metrics.json",result)
    flat_keys = ("pixels","NRR","temporal_correlation","amplitude_ratio","residual_pulsatility_ratio",
                 "off_harmonic_noise_reduction_db",
                 "mean_change_percent","lag_ms",
                 "vessel_background_contrast_denoised",
                 "local_contrast_valid_regions","local_contrast_denoised","local_contrast_change",
                 "profile_rmse_denoised_vs_sliding_average")
    pulse_keys = ("cycle_count","cycle_shape_correlation","cycle_normalized_rmse",
                  "cycle_peak_to_peak_ratio","cycle_upstroke_slope_ratio",
                  "cycle_total_variation_ratio","cycle_high_harmonic_power_ratio")
    with (output/"metrics.csv").open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=("region",*flat_keys,*(f"pulse_shape.{key}" for key in pulse_keys)))
        writer.writeheader()
        for name,row in results.items():
            writer.writerow(dict(region=name,**{key:row[key] for key in flat_keys},
                                 **{f"pulse_shape.{key}":row["pulse_shape"].get(key) for key in pulse_keys}))
    write_html(output,result)
    return result


def write_html(output,result):
    esc = html.escape
    def number(value): return "undefined" if value is None else f"{value:.5g}"
    rows = "".join("<tr><th>"+esc(LABELS[name])+"</th>"+"".join(f"<td>{number(row[key])}</td>" for key in
                   ("pixels","temporal_correlation","amplitude_ratio","residual_pulsatility_ratio",
                    "off_harmonic_noise_reduction_db","mean_change_percent","lag_ms"))+"</tr>"
                   for name,row in result["regions"].items())
    counts = "".join(f"<tr><th>{esc(LABELS[name])}</th>"+"".join(f"<td>{row[key]}</td>" for key in
                     ("input_pixels","outside_roi_pixels","excluded_overlap_pixels","evaluated_pixels"))+"</tr>"
                     for name,row in result["mask_counts"].items())
    sections = []
    structural_rows = "".join("<tr><th>"+esc(LABELS[name])+"</th>"+"".join(
        f"<td>{number(row[key])}</td>" for key in
        ("local_contrast_valid_regions","local_contrast_denoised","local_contrast_change",
         "profile_rmse_denoised_vs_sliding_average"))+"</tr>" for name,row in result["regions"].items())
    pulse_keys = ("cycle_count","cycle_shape_correlation","cycle_normalized_rmse",
                  "cycle_peak_to_peak_ratio","cycle_upstroke_slope_ratio",
                  "cycle_total_variation_ratio","cycle_high_harmonic_power_ratio")
    pulse_rows = "".join("<tr><th>"+esc(LABELS[name])+"</th>"+"".join(
        f"<td>{number(row['pulse_shape'].get(key))}</td>" for key in pulse_keys)+"</tr>"
        for name,row in result["regions"].items())
    for name in REGIONS:
        profile = result["regions"][name]["profile"]
        temporal = profile["temporal_metrics"]
        detail = (f'<p>Cross-section vessel samples: {profile["vessel_samples"]}; background samples: {profile["background_samples"]}. '
                  f'Vessel-trace correlation: {number(temporal.get("temporal_correlation"))}; '
                  f'cardiac amplitude ratio: {number(temporal.get("amplitude_ratio"))}; '
                  f'off-harmonic reduction: {number(temporal.get("off_harmonic_noise_reduction_db"))} dB. '
                  f'Profile comparison status: {esc(profile["consistency"]["status"])}; '
                  f'cross-section contrast status: {esc(profile["local_contrast"]["status"])}. '
                  f'<a href="plots/{name}_profile.npz">Profile arrays and sample masks</a>.</p>')
        sections.append(f"<h2>{LABELS[name]}</h2>"+
                        "".join(f'<img src="plots/{name}_{kind}.png" alt="{LABELS[name]} {kind}">' for kind in
                                ("waveform","pulse_shape","spectrum","local","contrast","profile"))+detail)
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Noise2Time: {esc(result['record'])}</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1150px;margin:32px auto;padding:0 20px;color:#243747}}h1,h2{{color:#126782}}img{{width:100%;margin:12px 0}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{padding:10px;border-bottom:1px solid #ccd6dd;text-align:left}}.note{{background:#edf5f7;padding:16px;line-height:1.6}}p{{line-height:1.6}}</style>
<h1>Noise2Time evaluation: {esc(result['record'])}</h1>
<p>{result['frames_scored']} scored frames; {result['excluded_prefix']} leading frames excluded. FPS: {number(result['fps'])}.
Selected frequency: {number(result['cardiac_frequency_hz'])} Hz. FFT resolution: {number(result['frequency_resolution_hz'])} Hz.<br>{esc(result['frequency_source'])}.<br>
Detector median period: {number(result['frequency_diagnostics']['peak_detector'].get('median_period_frames'))} frames;
arterial FFT strongest bin: {number(result['frequency_diagnostics']['arterial_fft'].get('strongest_fft_hz'))} Hz.</p>
<div class="note">{esc(result['caveat'])}<br>Background NRR: {number(result['background']['NRR'])} (fractional decrease in average pixel temporal SD). Background off-harmonic reduction: {number(result['background']['off_harmonic_noise_reduction_db'])} dB. Positive dB means less off-harmonic temporal power.</div>
<img src="plots/metrics_overview.png" alt="Compact denoising metric dashboard">
<table><tr><th>Region</th><th>Pixels</th><th>Correlation</th><th>Amplitude ratio</th><th>Residual pulsatility</th><th>Off-harmonic reduction dB</th><th>Mean change %</th><th>Lag ms</th></tr>{rows}</table>
<p>Amplitude ratios compare the same detector-derived frequency and are fitted sinusoid amplitudes, not peak-to-peak ranges. Residual pulsatility is the RMS Fourier amplitude at f0, 2f0 and 3f0 in <em>original − denoised</em>, divided by the corresponding original RMS amplitude; zero is ideal and values can exceed one. Off-harmonic reduction is 10 log10(original power / denoised power) after linear detrending and a Hann window. It uses frequencies at or above f0/2 and excludes bands of ±2 FFT bins around f0, 2f0 and 3f0. Positive values mean fewer off-harmonic fluctuations. Non-periodic physiology can also lie outside those bands, so this is not an accuracy score. Positive lag means delayed denoised output; check zero-lag correlation as well. Undefined values are not perfect scores. Harmonic metrics, local measurements and frequency cross-checks are in <a href="metrics.json">metrics.json</a>; summary: <a href="metrics.csv">metrics.csv</a>.</p>
<h2>Pulse-shape preservation</h2>
<table><tr><th>Region</th><th>Complete cycles</th><th>Shape correlation</th><th>Normalized shape RMSE</th><th>Peak-to-peak ratio</th><th>Upstroke-slope ratio</th><th>Total-variation ratio</th><th>Harmonics 4-10 power ratio</th></tr>{pulse_rows}</table>
<p>Complete detector peak-to-peak cycles are resampled to 128 phase samples and combined with a pointwise median. Shape correlation and normalized RMSE compare baseline- and amplitude-normalized pulses. Peak-to-peak amplitude, maximum periodic upstroke, total variation and higher-harmonic power should remain close to one. Derivative and shape calculations retain harmonics 1 through 10. At least two complete cycles are required. Per-phase values and cycle quartiles are saved in each region's pulse-shape CSV.</p>
<h2>Local contrast and 12-frame profile consistency</h2>
<table><tr><th>Region</th><th>Valid local tiles</th><th>Denoised contrast</th><th>Contrast change</th><th>Denoised profile RMSE</th></tr>{structural_rows}</table>
<p>Contrast is vessel mean minus nearby background mean, summarized equally across valid local tiles. The report shows only strategy-dependent values; input-only and 12-frame-reference constants are omitted. Background is the intersection of a local dilation (radius {result['structural_protocol']['local_background_radius_pixels']} pixels) with the cleaned background mask. Empty local backgrounds yield undefined contrast, not zero. Per-frame contrast CSVs and saved local-background masks document the samples used.</p>
<p>The reference averages 12 original frames, offsets −6 through +5 relative to frame t (centroid t−0.5). It uses future frames and only complete windows inside the scored sequence, without padding. Profile RMSE compares original and denoised profiles with this same reference on identical ROI-supported coordinates and timestamps. The displayed profile frame is selected with seed {PROFILE_SEED}; aggregate errors use all eligible frames. Values retain the input intensity scale. Lower distance means greater agreement with a noisy, temporally blurred reference, not proven accuracy. Fewer than 12 scored frames yield undefined structural comparisons.</p>
<h2>Mask selection and excluded overlaps</h2><table><tr><th>Mask</th><th>Input pixels</th><th>Outside ROI</th><th>Overlap removed</th><th>Used</th></tr>{counts}</table>
<p>Retinal/choroidal and artery/vein overlaps are removed simultaneously from the three anatomical groups. The handmade small-vessel mask is evaluated independently and may overlap those groups. Background is the ROI-restricted binary negation of the union of dilated original retinal and small-vessel masks with the original choroidal mask. Dilation disk radius: {result['mask_sources']['background']['retinal_dilation_radius_pixels']} pixels; choroidal mask is not dilated. Cleaned masks are saved in masks/. Automatic local patches favor high mask occupancy.</p>
<img src="plots/masks.png" alt="Cleaned masks, local regions, excluded overlaps">
<h2>Spatial changes and phase-resolved residuals</h2><img src="plots/spatial_maps.png" alt="Spatial change maps"><img src="plots/phase_residuals.png" alt="Residuals across cardiac phase">
<p>Phase bins use fractional intervals between preprocessing brightness peaks; incomplete intervals are excluded. These are peak-to-peak phase labels, not independently measured systole/diastole. Zero mean residual does not exclude removal of pulsation. Spatial arrays: <a href="spatial_maps.npz">spatial_maps.npz</a>.</p>
<p><a href="comparison.avi">Open synchronized original / 12-frame average / denoised / signed-residual video</a> (AVI requires a compatible player). The first three panels use [0,1]; residuals use a fixed symmetric scale of ±{number(result['comparison_residual_display_limit'])}, estimated from sampled frames. The average uses original frames t−6 through t+5 and is blank for the first six and last five scored frames, where no complete 12-frame window exists. Display clipping never affects metrics.</p>
{''.join(sections)}
<h2>Interpretation limits</h2><p>Inspect local curves, residual structures and profile placement before drawing conclusions. Automatic profile orientation is approximate; replace it with manually selected perpendicular lines using --profiles. Original noisy observations and temporal averages are not clean ground truth. No uncertainty estimates or independent generalization test are implied by this report.</p></html>'''
    (output/"report.html").write_text(text,encoding="utf-8")

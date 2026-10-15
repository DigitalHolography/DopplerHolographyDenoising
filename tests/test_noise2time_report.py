"""Regional masks, known signal changes and complete report generation."""
import json

import cv2
import numpy as np
import pytest

from doppler_denoising import noise2time as n2t
from doppler_denoising import noise2time_report as report


def masks():
    raw = {name: np.zeros((32,32),bool) for name in (*report.REGIONS,"background")}
    raw["retinal_artery"][2:14,3:8] = True
    raw["retinal_vein"][2:14,12:17] = True
    raw["choroidal"][10:24,5:23] = True
    raw["small_vessels"][5:8,5:7] = True  # Intentionally overlaps the artery.
    raw["small_vessels"][25:29,25:29] = True
    raw["background"][:] = True
    return raw


def test_simultaneous_exclusion():
    raw = masks()
    original = {k:v.copy() for k,v in raw.items()}
    clean,counts,excluded = report.exclusive_masks(raw,np.ones((32,32),bool))
    overlap = raw["choroidal"] & raw["retinal_artery"]
    assert overlap.any()
    assert not clean["choroidal"][overlap].any()
    assert not clean["retinal_artery"][overlap].any()
    assert excluded[overlap].all()
    assert np.stack([clean[name] for name in (*report.PRIMARY_REGIONS,"background")]).sum(0).max() == 1
    assert clean["small_vessels"][raw["small_vessels"]].all()
    for key in raw:
        np.testing.assert_array_equal(raw[key],original[key])
        assert counts[key]["evaluated_pixels"] == clean[key].sum()
    raw["retinal_vein"] = raw["retinal_artery"].copy()
    with pytest.raises(ValueError,match="no pixels left"):
        report.exclusive_masks(raw,np.ones((32,32),bool))


def test_known_gain_and_delay():
    t = np.arange(200)/20
    before = .3+.02*np.sin(2*np.pi*t)
    after = .32+.016*np.sin(2*np.pi*(t-.1))
    result = report.waveform_metrics(before,after,20,1,.25)
    assert result["amplitude_ratio"] == pytest.approx(.8)
    assert result["mean_change"] == pytest.approx(.02)
    assert result["lag_frames"] == 2
    assert result["lag_ms"] == 100
    assert result["lag_corrected_correlation"] == pytest.approx(1)
    constant = report.waveform_metrics(np.ones(40),np.ones(40),20,1,.25)
    assert constant["temporal_correlation"] is None
    assert constant["amplitude_ratio"] is None


def test_residual_pulsatility_combines_fundamental_and_harmonics():
    t = np.arange(400) / 40
    before = (.3 + .02*np.sin(2*np.pi*t)
              + .01*np.sin(4*np.pi*t+.2)
              + .005*np.sin(6*np.pi*t-.3))
    identical = report.residual_pulsatility(before, before, 40, 1)
    scaled = report.residual_pulsatility(before, .8*before, 40, 1)
    assert identical["residual_pulsatility_ratio"] == pytest.approx(0, abs=1e-12)
    assert scaled["residual_pulsatility_ratio"] == pytest.approx(.2)
    assert scaled["residual_pulsatility_power_ratio"] == pytest.approx(.04)
    assert [item["order"] for item in scaled["harmonics"]] == [1, 2, 3]


def test_off_harmonic_noise_excludes_cardiac_bands_and_reports_db_reduction():
    t=np.arange(800)/40
    cardiac=.04*np.sin(2*np.pi*t)+.02*np.sin(4*np.pi*t+.2)
    fluctuation=.03*np.sin(2*np.pi*7*t+.4)
    result=report.off_harmonic_noise(cardiac+fluctuation,cardiac+.5*fluctuation,40,1)
    assert result["off_harmonic_power_ratio"]==pytest.approx(.25,rel=.03)
    assert result["off_harmonic_noise_reduction"]==pytest.approx(.75,rel=.02)
    assert result["off_harmonic_noise_reduction_db"]==pytest.approx(6.0206,abs=.15)
    assert result["off_harmonic_bin_count"]>0
    assert [band["order"] for band in result["off_harmonic_excluded_bands"]]==[1,2,3]


def test_peak_frequency_uses_robust_arterial_peak_period():
    frequency, diagnostics = report.peak_frequency([10, 95, 182, 267], 144.53125)
    assert frequency == pytest.approx(144.53125 / 85)
    assert diagnostics["period_mad_frames"] == pytest.approx(0)
    assert report.peak_frequency([10, 95], 144.53125)[0] is None


def test_cycle_aligned_pulse_shape_detects_temporal_smoothing():
    frames = 600
    phase = (np.arange(frames) % 100) / 100
    # A periodic narrow pulse contains the higher harmonics and steep upstroke
    # that a moving average can remove while retaining the fundamental.
    pulse = np.exp(10*(np.cos(2*np.pi*(phase-.15))-1))
    original = .3+.08*pulse
    padded = np.r_[original[-5:], original, original[:5]]
    averaged = np.convolve(padded, np.ones(11)/11, mode="valid")
    peaks = list(range(0, frames+1, 100))

    identical, _ = report.pulse_shape_metrics(original, original, peaks, 0)
    smoothed, data = report.pulse_shape_metrics(original, averaged, peaks, 0)

    assert identical["cycle_shape_correlation"] == pytest.approx(1)
    assert identical["cycle_normalized_rmse"] == pytest.approx(0)
    assert identical["cycle_high_harmonic_power_ratio"] == pytest.approx(1)
    assert smoothed["cycle_count"] == 6
    assert smoothed["cycle_shape_correlation"] > .98  # Correlation alone is permissive.
    assert smoothed["cycle_normalized_rmse"] > .1
    assert smoothed["cycle_upstroke_slope_ratio"] < .8
    assert smoothed["cycle_high_harmonic_power_ratio"] < .5
    assert data["original_cycles"].shape == (6, report.PULSE_PHASE_SAMPLES)
    unavailable, _ = report.pulse_shape_metrics(original[:150], averaged[:150], peaks, 0)
    assert unavailable["status"] == "unavailable"


def test_12_frame_reference_alignment_and_no_boundary_padding():
    values = np.arange(20, dtype=float)
    indices, average = report.sliding_average_reference(values)
    np.testing.assert_array_equal(indices, np.arange(6,15))
    np.testing.assert_allclose(average,indices-.5)
    # Averaging fixed profiles commutes with bilinear spatial sampling.
    profiles = values[:,None]+np.array([1.,2.,4.])
    _, averaged_profiles = report.sliding_average_reference(profiles)
    np.testing.assert_allclose(averaged_profiles,average[:,None]+[1,2,4])
    assert report.sliding_average_reference(np.ones(11))[1].shape == (0,)
    np.testing.assert_allclose(report.sliding_average_reference(np.arange(12))[1],[5.5])


def test_comparison_video_places_complete_12_frame_average_between_inputs(tmp_path):
    original = np.ones((20,32,32),dtype=np.float32)
    original[:6] = 0
    denoised = np.zeros_like(original)
    path = tmp_path/"comparison.avi"

    report.write_comparison(path,original,denoised,20.,.5)

    cap = cv2.VideoCapture(str(path))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 20
    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 128
    frames = []
    while True:
        ok,frame = cap.read()
        if not ok: break
        frames.append(frame)
    cap.release()
    assert len(frames) == 20
    # Avoid the text labels when sampling panel intensities.  Frame 6 uses
    # original frames 0..11: six zeros and six ones, hence mean 0.5.
    area = np.s_[22:30,:]
    assert frames[0][area][:,32:64].mean() < 10
    assert frames[6][area][:,:32].mean() > 240
    assert 110 < frames[6][area][:,32:64].mean() < 145
    assert frames[6][area][:,64:96].mean() < 10
    assert frames[14][area][:,32:64].mean() > 240
    assert frames[-1][area][:,32:64].mean() < 10


def test_contrast_known_gain_and_missing_background():
    bg = np.full((2,30),.4)
    vessels = bg+np.array([.2,.1])[:,None]
    result, data = report.contrast_metrics(vessels,bg)
    assert result['original_mean'] == pytest.approx(.2)
    assert result['denoised_mean'] == pytest.approx(.1)
    assert result['sliding_average_mean'] == pytest.approx(.2)
    assert result['denoised_to_original_ratio'] == pytest.approx(.5)
    assert result['rmse_denoised_vs_sliding_average'] == pytest.approx(.1)
    assert len(data[0]) == 19
    assert report.contrast_metrics(vessels,None)[0]['reason'] == 'no_local_background'
    assert report.contrast_metrics(vessels[:,:5],bg[:,:5])[0]['frames_scored'] == 0
    assert report.contrast_metrics(bg,bg)[0]['denoised_to_original_ratio'] is None


def test_float32_constant_trace_is_undefined_not_nan():
    before = np.full(151,.2,dtype=np.float32)
    after = np.linspace(.2,.3,151,dtype=np.float32)
    assert report.correlation(before,after) is None
    result = report.waveform_metrics(before,after,20,1,.25)
    assert result['temporal_correlation'] is None
    assert result['lag_corrected_correlation'] is None
    json.dumps(result,allow_nan=False)


def test_local_background_excludes_other_vessels_and_outside_roi():
    vessel = np.zeros((9,9),bool); vessel[4,4] = True
    background = np.ones_like(vessel); background[4,4] = False
    background[:,5] = False  # Neighbor vessel.
    background[:2] = False  # Outside imaging ROI.
    selected = report.local_background(vessel,background,3)
    assert selected[4,3]
    assert not selected[:,5].any()
    assert not selected[:2].any()
    assert not selected[0,0]


def test_profile_error_uses_original_average_and_keeps_intensity_scale():
    rng = np.random.default_rng(11)
    original = 2+rng.normal(0,.1,(50,7))  # Deliberately outside display [0,1].
    denoised = original.copy()
    indices, reference = report.sliding_average_reference(original)
    denoised[indices] = reference
    result, data = report.profile_consistency(original,denoised,np.ones(7,bool))
    assert result['rmse_original_mean'] > .01
    assert result['rmse_denoised_mean'] == pytest.approx(0)
    assert result['selected_scored_frame'] in indices
    same, _ = report.profile_consistency(original,original,np.ones(7,bool))
    assert result['selected_scored_frame'] == same['selected_scored_frame']
    assert same['rmse_original_mean'] == same['rmse_denoised_mean']
    denoised[indices] += .05
    changed, _ = report.profile_consistency(original,denoised,np.ones(7,bool))
    assert changed['rmse_denoised_mean'] == pytest.approx(.05)
    assert report.profile_consistency(original,denoised,np.zeros(7,bool))[0]['status'] == 'unavailable'
    assert report.profile_consistency(original[:8],denoised[:8],np.ones(7,bool))[0]['status'] == 'unavailable'


def test_cross_section_temporal_gain_and_local_contrast(tmp_path):
    vessel = np.zeros((9,9),bool); vessel[:,4] = True
    background = np.zeros_like(vessel); background[:,[0,1,7,8]] = True
    t=np.arange(80)/20
    original=np.full((80,9,9),.3)
    original[:,vessel] += (.2+.04*np.sin(2*np.pi*t))[:,None]
    denoised=.3+.8*(original-.3)
    result=report.save_profile(tmp_path,'retinal_artery',vessel,original,denoised,20,8,0,
                               dict(start=[0,4],end=[8,4]),vessel_mask=vessel,
                               background_mask=background,roi=np.ones_like(vessel),frequency=1,
                               first_original_frame=12)
    assert result['temporal_metrics']['amplitude_ratio'] == pytest.approx(.8)
    assert result['temporal_metrics']['temporal_correlation'] == pytest.approx(1)
    assert result['local_contrast']['denoised_to_original_ratio'] == pytest.approx(.8)
    assert result['vessel_samples']==1 and result['background_samples']==4
    assert result['selected_original_frame']==result['selected_scored_frame']+12
    arrays=np.load(tmp_path/'retinal_artery_profile.npz')
    np.testing.assert_allclose(arrays['sliding_average'][0],arrays['original'][:12].mean(0))
    assert (tmp_path/'retinal_artery_profile_metrics.csv').exists()
    assert (tmp_path/'retinal_artery_profile_contrast.csv').exists()


def test_derived_background():
    raw = {name:np.zeros((9,9),bool) for name in report.REGIONS}
    raw["retinal_artery"][4,4] = True
    raw["retinal_vein"][1,1] = True
    raw["choroidal"][4,4] = True  # Even excluded retinal/choroidal overlap is dilated.
    raw["choroidal"][7,7] = True
    raw["small_vessels"][5,7] = True
    roi = np.ones((9,9),bool); roi[0] = False
    result = report.derive_background(raw,roi,1)
    expected = roi.copy()
    for y,x in ((4,4),(1,1),(5,7)):
        for dy,dx in ((0,0),(-1,0),(1,0),(0,-1),(0,1)):
            expected[y+dy,x+dx] = False
    expected[7,7] = False
    np.testing.assert_array_equal(result,expected)
    assert result[7,6]  # Choroidal mask is not dilated.
    np.testing.assert_array_equal(report.derive_background(raw,roi,0),
                                  roi & ~np.logical_or.reduce(list(raw.values())))
    with pytest.raises(ValueError,match="nonnegative"):
        report.derive_background(raw,roi,-1)


def test_accumulation_and_phase():
    rng = np.random.default_rng(17)
    original = rng.random((30,4,5))
    denoised = .8*original
    labels = report.phase_bins(30,5,[0,10,20,30])
    assert np.count_nonzero(labels<0) == 5
    mask = np.ones((4,5),bool)
    mean,std,curves,phases,counts = report.accumulate(original,denoised,{"all":mask},labels,chunk=7)
    np.testing.assert_allclose(mean[0],original.mean(0))
    np.testing.assert_allclose(std[1],denoised.std(0))
    np.testing.assert_allclose(curves["all"][0],original.mean((1,2)))
    for phase in range(4):
        np.testing.assert_allclose(phases[phase],(original-denoised)[labels==phase].mean(0))
        assert counts[phase] == (labels==phase).sum()


def test_complete_regional_report(tmp_path):
    record = tmp_path/"measure"; record.mkdir()
    raw = masks()
    t = np.arange(90)/20
    rng = np.random.default_rng(8)
    frames = .3+.003*rng.normal(size=(90,32,32))
    for index,name in enumerate(report.REGIONS):
        frames[:,raw[name]] += (.01*(index+1)*np.sin(2*np.pi*t))[:,None]
    frames = frames.astype(np.float32)
    np.save(record/"frames.npy",frames)
    np.save(record/"roi.npy",np.ones((32,32),bool))
    np.save(record/"brightness.npy",frames.mean((1,2)))
    np.save(record/"phase.npy",np.arange(90)%20)
    n2t.write_json(record/"metadata.json",dict(fps=20,first_original_frame=3,peaks=[5,25,45,65,85]))
    denoised = tmp_path/"denoised.npy"; np.save(denoised,frames)
    n2t.write_json(denoised.with_suffix(".json"),dict(record_sha256=n2t.sha256(record/"frames.npy"),copied_prefix=9))
    paths = {}
    for name,mask in raw.items():
        paths[name] = tmp_path/f"{name}.png"
        assert cv2.imwrite(str(paths[name]),mask.astype(np.uint8)*255)
    output = tmp_path/"evaluation"
    command = ["evaluate","--record",str(record),"--denoised",str(denoised),
               "--retinal-artery-mask",str(paths["retinal_artery"]),
               "--retinal-vein-mask",str(paths["retinal_vein"]),
               "--choroidal-masks",str(paths["choroidal"]),str(paths["choroidal"]),
               "--small-vessels-mask",str(paths["small_vessels"]),
               "--background-dilation-radius","2","--cardiac-hz","1",
               "--local-size","8","--output",str(output)]
    assert n2t.main(command) == 0
    data = json.loads((output/"metrics.json").read_text())
    assert data["frames_scored"] == 81
    assert data['schema']=='noise2time.regional.v6'
    assert data['structural_protocol']['offsets_inclusive']==[-6,5]
    assert data["first_original_scored_frame"] == 12
    assert data["background"]["NRR"] == pytest.approx(0)
    assert data["background"]["off_harmonic_noise_reduction_db"] == pytest.approx(0,abs=1e-6)
    np.testing.assert_array_equal(np.load(output/"masks"/"background.npy"),
                                  report.derive_background(raw,np.ones((32,32),bool),2))
    assert data["mask_sources"]["background"]["retinal_dilation_radius_pixels"] == 2
    for name in report.REGIONS:
        assert data["regions"][name]["amplitude_ratio"] == pytest.approx(1)
        assert data["regions"][name]["temporal_correlation"] == pytest.approx(1)
        assert data["regions"][name]["residual_pulsatility_ratio"] == pytest.approx(0, abs=1e-6)
        assert data["regions"][name]["off_harmonic_noise_reduction_db"] == pytest.approx(0,abs=1e-6)
        assert data["regions"][name]["lag_ms"] == 0
        row=data['regions'][name]
        assert 'profile_rmse_original_vs_sliding_average' not in row
        assert row['pulse_shape']['cycle_count'] == 3
        assert row['pulse_shape']['cycle_shape_correlation'] == pytest.approx(1)
        assert row['pulse_shape']['cycle_normalized_rmse'] == pytest.approx(0,abs=1e-7)
        assert row['pulse_shape']['cycle_upstroke_slope_ratio'] == pytest.approx(1)
        assert row['pulse_shape']['cycle_high_harmonic_power_ratio'] == pytest.approx(1)
        assert row['profile']['consistency']['frames_scored']==70
        assert row['profile']['selected_scored_frame']>=6
        if row['local_contrast_valid_regions']:
            assert row['local_contrast_change']==pytest.approx(0)
        assert (output/f'plots/{name}_contrast.png').exists()
        assert (output/f'plots/{name}_profile_metrics.csv').exists()
        assert (output/f'plots/{name}_pulse_shape.csv').exists()
        assert (output/f'plots/{name}_pulse_shape.png').exists()
    assert (output/"report.html").is_file()
    assert (output/"plots/metrics_overview.png").is_file()
    assert "residual_pulsatility_ratio" in (output/"metrics.csv").read_text()
    assert "off_harmonic_noise_reduction_db" in (output/"metrics.csv").read_text()
    summary = (output/"metrics.csv").read_text()
    assert "pulse_shape.cycle_upstroke_slope_ratio" in summary
    assert "local_contrast_original" not in summary
    assert "local_contrast_sliding_average" not in summary
    assert "profile_rmse_original_vs_sliding_average" not in summary
    for path in (output/"plots").glob("*.png"):
        assert cv2.imread(str(path)) is not None
    cap = cv2.VideoCapture(str(output/"comparison.avi"))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 81
    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 128
    cap.release()
    with pytest.raises(FileExistsError):
        n2t.main(command)
    # Failure must not publish a partial report or retain staging folders.
    assert cv2.imwrite(str(paths["retinal_vein"]),raw["retinal_artery"].astype(np.uint8)*255)
    command[-1] = str(tmp_path/"invalid_report")
    with pytest.raises(ValueError,match="no pixels left"):
        n2t.main(command)
    assert not (tmp_path/"invalid_report").exists()
    assert not list(tmp_path.glob(".evaluation-*"))

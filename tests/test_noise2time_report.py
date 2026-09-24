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
    assert np.stack(list(clean.values())).sum(0).max() == 1
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


def test_peak_frequency_uses_robust_arterial_peak_period():
    frequency, diagnostics = report.peak_frequency([10, 95, 182, 267], 144.53125)
    assert frequency == pytest.approx(144.53125 / 85)
    assert diagnostics["period_mad_frames"] == pytest.approx(0)
    assert report.peak_frequency([10, 95], 144.53125)[0] is None


def test_derived_background():
    raw = {name:np.zeros((9,9),bool) for name in report.REGIONS}
    raw["retinal_artery"][4,4] = True
    raw["retinal_vein"][1,1] = True
    raw["choroidal"][4,4] = True  # Even excluded retinal/choroidal overlap is dilated.
    raw["choroidal"][7,7] = True
    roi = np.ones((9,9),bool); roi[0] = False
    result = report.derive_background(raw,roi,1)
    expected = roi.copy()
    for y,x in ((4,4),(1,1)):
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
               "--background-dilation-radius","2","--cardiac-hz","1",
               "--local-size","8","--output",str(output)]
    assert n2t.main(command) == 0
    data = json.loads((output/"metrics.json").read_text())
    assert data["frames_scored"] == 81
    assert data["first_original_scored_frame"] == 12
    assert data["background"]["NRR"] == pytest.approx(0)
    np.testing.assert_array_equal(np.load(output/"masks"/"background.npy"),
                                  report.derive_background(raw,np.ones((32,32),bool),2))
    assert data["mask_sources"]["background"]["retinal_dilation_radius_pixels"] == 2
    for name in report.REGIONS:
        assert data["regions"][name]["amplitude_ratio"] == pytest.approx(1)
        assert data["regions"][name]["temporal_correlation"] == pytest.approx(1)
        assert data["regions"][name]["residual_pulsatility_ratio"] == pytest.approx(0, abs=1e-6)
        assert data["regions"][name]["lag_ms"] == 0
    assert (output/"report.html").is_file()
    assert (output/"plots/metrics_overview.png").is_file()
    assert "residual_pulsatility_ratio" in (output/"metrics.csv").read_text()
    for path in (output/"plots").glob("*.png"):
        assert cv2.imread(str(path)) is not None
    cap = cv2.VideoCapture(str(output/"comparison.avi"))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 81
    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 96
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

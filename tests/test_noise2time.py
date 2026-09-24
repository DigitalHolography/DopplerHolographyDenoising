"""Scientific invariants and a tiny CPU end-to-end experiment; no external data."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from doppler_denoising import noise2time as n2t


def test_article_loss_matches_analytic_derivatives():
    # e(x,y) = x^2 + 2xy + 3y^2: Dxx=2, Dyy=6, forward Dxy=2.
    yy, xx = torch.meshgrid(torch.arange(5.), torch.arange(6.), indexing="ij")
    error = (xx**2 + 2*xx*yy + 3*yy**2)[None,None]
    mask = torch.ones_like(error)
    parts = n2t.loss_terms(error, torch.zeros_like(error), mask)
    gx = error[...,1:] - error[...,:-1]
    gy = error[...,1:,:] - error[...,:-1,:]
    expected_grad = (gx.sum()+gy.sum())/(gx.numel()+gy.numel())
    expected_hessian = (2*5*4 + 6*3*6 + 2*2*4*5)/(5*4+3*6+2*4*5)
    assert parts["reconstruction"].item() == pytest.approx(error.mean().item())
    assert parts["gradient"].item() == pytest.approx(expected_grad.item())
    assert parts["hessian"].item() == pytest.approx(expected_hessian)
    assert parts["total"].item() == pytest.approx(error.mean().item()+.1*expected_grad.item()+.05*expected_hessian)


def test_l1_l2_composite_losses():
    yy,xx=torch.meshgrid(torch.arange(5.),torch.arange(6.),indexing='ij')
    prediction=(xx**2+2*xx*yy+3*yy**2)[None,None].requires_grad_()
    target=torch.zeros_like(prediction);mask=torch.ones_like(prediction)
    article=n2t.loss_terms(prediction,target,mask,'article')
    for objective in ('l1','l2','l1_grad_hessian','l2_grad_hessian'):
        parts=n2t.loss_terms(prediction,target,mask,objective)
        expected=prediction.square().mean() if objective.startswith('l2') else prediction.abs().mean()
        torch.testing.assert_close(parts['reconstruction'][0],expected)
        regularized=objective.endswith('grad_hessian')
        for term in ('gradient','hessian'):
            torch.testing.assert_close(parts[term],article[term] if regularized else torch.zeros(1))
        torch.testing.assert_close(parts['total'],parts['reconstruction']+.1*parts['gradient']+.05*parts['hessian'])
        parts['total'].sum().backward(retain_graph=True)
        assert torch.isfinite(prediction.grad).all()
    with pytest.raises(ValueError,match='Unknown objective'):
        n2t.loss_terms(prediction,target,mask,'typo')


def test_unmasked_pixels_cannot_affect_any_loss():
    torch.manual_seed(3)
    target = torch.rand(1,1,12,12)
    prediction = target.clone()
    mask = torch.zeros_like(target)
    mask[:,:,3:9,3:9] = 1
    prediction[mask == 0] += 5
    prediction.requires_grad_()
    parts = n2t.loss_terms(prediction,target,mask)
    assert parts["total"].item() == 0
    parts["total"].sum().backward()
    assert prediction.grad[mask == 0].abs().sum().item() == 0


def fake_record(name="r"):
    frames = np.stack([np.full((32,32), .1+.01*i, np.float32) for i in range(30)])
    phases = np.arange(30)%10
    record = SimpleNamespace(name=name,frames=frames,roi=np.ones((32,32),bool),
                             phase=phases,brightness=np.ones(30),
                             donors={i:np.flatnonzero(phases==i) for i in range(10)})
    record.eligible = lambda history:list(range(history,30))
    return record


@pytest.mark.parametrize("size,blocks", [(1,3),(8,2),(32,1)])
def test_replacement_exact_support_and_donor_exclusion(size,blocks):
    record = fake_record()
    cfg = n2t.Config(history=2,block_size=size,blocks=blocks,objective="l2")
    sequence,target,mask = n2t.replacement(record,12,cfg,np.random.default_rng(4))
    np.testing.assert_array_equal(sequence[:-1],record.frames[10:12])
    np.testing.assert_array_equal(target[0],record.frames[12])
    assert mask.sum() == size*size*blocks
    np.testing.assert_array_equal(sequence[-1][mask[0]==0],target[0][mask[0]==0])
    assert np.all(sequence[-1][mask[0]>0] != target[0][mask[0]>0])


def test_replacement_brightness_scaling_and_clipping():
    record = fake_record()
    record.brightness[12] = 100
    cfg = n2t.Config(history=2,block_size=8,blocks=1)
    sequence,_,mask = n2t.replacement(record,12,cfg,np.random.default_rng(4))
    assert np.all(sequence[-1][mask[0]>0] == 1)


def test_record_split_is_disjoint_and_fixed():
    records = [fake_record("a"),fake_record("b")]
    cfg = n2t.Config(history=2,validation_records=("b",),validation_samples=4)
    train,valid = n2t.split_samples(records,cfg)
    assert {i for i,t in train} == {0}
    assert {i for i,t in valid} == {1}
    assert n2t.split_samples(records,cfg) == (train,valid)


def test_training_folder_and_per_record_epoch_sampling(tmp_path):
    parent = tmp_path / "prepared"
    for name in ("b", "a"):
        record = parent / name
        record.mkdir(parents=True)
        (record / "frames.npy").touch()
    (parent / "preparation_report.json").touch()
    assert [p.name for p in n2t.resolve_training_records([parent])] == ["a", "b"]
    with pytest.raises(ValueError, match="more than once"):
        n2t.resolve_training_records([parent, parent / "a"])

    training = [(0, t) for t in range(10)] + [(1, t) for t in range(2)]
    choices = n2t.select_train_samples_for_epoch(training, 6, np.random.default_rng(3))
    assert len(choices) == 6
    assert {training[int(index)][0] for index in choices} == {0, 1}
    with pytest.raises(ValueError, match="exceed"):
        n2t.select_train_samples_for_epoch(training, 1, np.random.default_rng(3))


def test_empty_masks_and_impossible_placement_raise():
    with pytest.raises(ValueError,match="Empty"):
        n2t.loss_terms(torch.zeros(1,1,4,4),torch.zeros(1,1,4,4),torch.zeros(1,1,4,4))
    with pytest.raises(ValueError,match="Placed"):
        n2t.replacement(fake_record(),12,n2t.Config(history=2,block_size=32,blocks=2),np.random.default_rng(4))


def test_fractional_phase_aligns_unequal_cycles():
    phase = n2t.phases_from_peaks(90,np.array([0,30]))
    assert phase[15] == phase[45] == 15
    cycle,fraction,intervals=n2t.cycle_coordinates(50,np.array([0,10,25]))
    assert intervals==((0,10),(10,25))
    assert cycle[3]==0 and fraction[3]==pytest.approx(.3)
    # Phase .3 in the 15-frame cycle is coordinate 14.5.
    donor=n2t.PhaseDonor(14,15,.5,1)
    assert n2t.interpolate_donor(np.arange(50,dtype=float),donor)==pytest.approx(14.5)
    assert cycle[25]==-1 and np.isnan(fraction[25])


@pytest.mark.parametrize("objective,expected", [("l1",2.),("l2",4.)])
def test_reconstruction_ablations(objective,expected):
    parts = n2t.loss_terms(torch.full((2,1,4,4),2.),torch.zeros(2,1,4,4),
                          torch.ones(2,1,4,4),objective)
    assert parts["total"].tolist() == [expected,expected]
    assert parts["gradient"].sum() == 0
    assert parts["hessian"].sum() == 0


def test_temporal_memory_uses_history_and_resets():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(6)
        sequence = torch.rand(1,3,32,32,requires_grad=True)
        model = n2t.Noise2Time(base_channels=8).eval()
        result = model(sequence)
        gradient = torch.autograd.grad(result.square().mean(),sequence)[0]
        assert gradient[:,:-1].abs().sum() > 0
        with torch.no_grad():
            model(torch.zeros_like(sequence))
            torch.testing.assert_close(result,model(sequence))
        spatial = n2t.Noise2Time(base_channels=8,convlstm=False).eval()
        result = spatial(sequence)
        gradient = torch.autograd.grad(result.square().mean(),sequence)[0]
        assert gradient[:,:-1].abs().sum() == 0
    finally:
        torch.set_num_threads(previous_threads)


def test_cpu_end_to_end(tmp_path):
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    try:
        rng = np.random.default_rng(2)
        t = np.arange(64)
        frames = np.broadcast_to((.4+.08*np.sin(2*np.pi*t/12))[:,None,None],(64,32,32)).copy()
        frames += rng.normal(0,.005,frames.shape)
        source = tmp_path/"source.npy"
        np.save(source,frames.astype(np.float32))
        prepared = tmp_path/"measure_HD_M0"
        n2t.prepare(SimpleNamespace(input=source,output=prepared,fps=12.,cx=15,cy=15,radius=22,
                                   smooth_window=3,peak_distance=8,prominence=.1,brightness_mask=None))
        cfg = n2t.Config(base_channels=8,history=2,block_size=8,blocks=1,
                         epochs=1,samples_per_epoch=2,batch_size=2,validation_samples=2)
        config = tmp_path/"config.json"
        n2t.write_json(config,n2t.asdict(cfg))
        run = tmp_path/"run"
        n2t.train(SimpleNamespace(config=config,records=[prepared],output=run,device="cpu"))
        assert (run/"best.pt").exists()
        preview = run/"previews"/"epoch_001"/"measure_HD_M0.avi"
        assert preview.exists()
        assert json.loads(preview.with_suffix(".json").read_text())["epoch"] == 1
        result = tmp_path/"denoised.npy"
        n2t.denoise(SimpleNamespace(checkpoint=run/"best.pt",record=prepared,output=result.with_suffix(".avi"),device="cpu"))
        output = np.load(result)
        original = np.load(prepared/"frames.npy")
        assert output.shape == original.shape
        np.testing.assert_array_equal(output[:2],original[:2])
        bundle_root = tmp_path/"results"
        n2t.denoise(SimpleNamespace(checkpoint=run/"best.pt",record=prepared,output=bundle_root,device="cpu"))
        bundle = bundle_root/"measure"
        assert {p.name for p in bundle.iterdir()} == {"original.avi","denoised.avi","denoised.npy","denoised.json"}
        np.testing.assert_array_equal(np.load(bundle/"denoised.npy"),output)
        metadata = json.loads((bundle/"denoised.json").read_text())
        assert metadata["original_avi"] == str(bundle/"original.avi")
        assert metadata["first_original_frame"] > 0
        with pytest.raises(FileExistsError):
            n2t.denoise(SimpleNamespace(checkpoint=run/"best.pt",record=prepared,output=bundle_root,device="cpu"))
        for video_path in (preview,result.with_suffix(".avi"),bundle/"original.avi",bundle/"denoised.avi"):
            cap = n2t.cv2.VideoCapture(str(video_path))
            assert cap.isOpened()
            assert cap.get(n2t.cv2.CAP_PROP_FPS) == pytest.approx(12.)
            count = 0
            while True:
                ok,frame = cap.read()
                if not ok:
                    break
                assert frame.shape[:2] == (32,32)
                if video_path == bundle/"original.avi":
                    expected = np.rint(original[count]*255)
                    assert np.abs(n2t.cv2.cvtColor(frame,n2t.cv2.COLOR_BGR2GRAY).astype(float)-expected).mean() < 3
                count += 1
            cap.release()
            assert count == len(original)
        vessel = np.zeros((32,32),bool); vessel[12:18,12:18] = True
        background = np.zeros_like(vessel); background[5:10,5:10] = True
        np.save(tmp_path/"vessel.npy",vessel)
        np.save(tmp_path/"background.npy",background)
        metrics = tmp_path/"metrics.json"
        # Identity control verifies metric scale and copied-prefix handling.
        np.save(result,original)
        n2t.evaluate(SimpleNamespace(record=prepared,denoised=result,vessel_mask=tmp_path/"vessel.npy",
                                     background_mask=tmp_path/"background.npy",min_hz=.5,max_hz=3.,output=metrics))
        row = json.loads(metrics.read_text())
        assert row["NRR"] == pytest.approx(0)
        assert row["amplitude_ratio"] == pytest.approx(1)
        assert row["temporal_correlation"] == pytest.approx(1)
        assert row["frames_scored"] == len(original)-2
        n2t.cv2.imwrite(str(tmp_path/"vessel.png"),vessel.astype(np.uint8)*255)
        png_metrics = tmp_path/"png_metrics.json"
        n2t.evaluate(SimpleNamespace(record=prepared,denoised=result,vessel_mask=tmp_path/"vessel.png",
                                     background_mask=tmp_path/"background.npy",min_hz=.5,max_hz=3.,output=png_metrics))
        png_row = json.loads(png_metrics.read_text())
        for key in ("NRR","amplitude_ratio","temporal_correlation","vessel_mean_original"):
            assert png_row[key] == pytest.approx(row[key])
        summary = tmp_path/"summary.json"
        n2t.summarize(SimpleNamespace(metrics=[metrics],output=summary))
        assert json.loads(summary.read_text())["summary"]["NRR"]["sample_sd"] is None
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)


def test_prepare_folder_mixed_inputs_failure_and_rerun(tmp_path):
    source = tmp_path/"collected"; source.mkdir()
    output = tmp_path/"prepared"
    values = (.4+.1*np.sin(2*np.pi*np.arange(64)/12)).astype(np.float32)
    frames = np.broadcast_to(values[:,None,None],(64,32,32)).copy()
    np.save(source/"array_HD_M0.npy",frames)
    writer = n2t.cv2.VideoWriter(str(source/"video_HD_M0.avi"),
                                n2t.cv2.VideoWriter_fourcc(*"MJPG"),12.,(32,32),True)
    assert writer.isOpened()
    for frame in frames:
        writer.write(n2t.cv2.cvtColor((frame*255).astype(np.uint8),n2t.cv2.COLOR_GRAY2BGR))
    writer.release()
    np.save(source/"broken.npy",np.zeros(4))
    (source/"collection_report.json").write_text("{}")
    args = ["prepare","--input",str(source),"--output",str(output),"--fps","12",
            "--cx","15","--cy","15","--radius","22","--smooth-window","3","--peak-distance","8"]
    assert n2t.main(args) == 1  # Good recordings still finish after an error.
    report = json.loads(next(output.glob("preparation_*.json")).read_text())
    assert report["counts"] == {"prepared":2,"skipped":0,"error":1}
    assert not (output/"broken").exists()
    assert not list(output.glob(".prepare-*"))
    for name in ("array_HD_M0","video_HD_M0"):
        record = n2t.Record(output/name)
        assert record.frames.shape[1:] == (32,32)
        assert record.metadata["source"].startswith(str(source))
        table = np.genfromtxt(output/name/"brightness.csv",delimiter=",",names=True)
        np.testing.assert_array_equal(table["frame_index"],np.arange(len(record.frames)))
        np.testing.assert_array_equal(table["original_frame_index"],table["frame_index"]+record.metadata["first_original_frame"])
        np.testing.assert_allclose(table["time_seconds"],table["frame_index"]/12.)
        np.testing.assert_allclose(table["smooth_brightness"],record.brightness)
        np.testing.assert_allclose(table["raw_brightness"],np.load(output/name/"brightness_raw.npy"))
        np.testing.assert_array_equal(table["phase_frames"],record.phase)
        np.testing.assert_array_equal(np.flatnonzero(table["is_peak"]),record.metadata["peaks"])
        assert n2t.cv2.imread(str(output/name/"brightness.png")) is not None
        cap = n2t.cv2.VideoCapture(str(output/name/"prepared.avi"))
        assert cap.isOpened()
        assert cap.get(n2t.cv2.CAP_PROP_FPS) == pytest.approx(12.)
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            expected = np.rint(record.frames[count]*255)
            assert np.abs(n2t.cv2.cvtColor(frame,n2t.cv2.COLOR_BGR2GRAY).astype(float)-expected).mean() < 3
            count += 1
        cap.release()
        assert count == len(record.frames)
    saved = (output/"array_HD_M0"/"metadata.json").read_bytes()
    assert n2t.main(args+["--pattern","array*.npy","--skip-existing"]) == 0
    assert (output/"array_HD_M0"/"metadata.json").read_bytes() == saved


def test_prepare_folder_rejects_colliding_names(tmp_path):
    source = tmp_path/"collected"; source.mkdir()
    (source/"same.npy").touch()
    (source/"same.avi").touch()
    with pytest.raises(ValueError,match="same stem"):
        n2t.main(["prepare","--input",str(source),"--output",str(tmp_path/"out")])
    assert not (tmp_path/"out").exists()


def test_prepare_folder_cleans_failed_partial_write(tmp_path,monkeypatch):
    source = tmp_path/"collected"; source.mkdir()
    (source/"video.npy").touch()
    def fail_during_write(args):
        Path(args.output).mkdir()
        (Path(args.output)/"frames.npy").write_bytes(b"incomplete")
        raise OSError("Simulated write failure")
    monkeypatch.setattr(n2t,"prepare_one",fail_during_write)
    output = tmp_path/"out"
    assert n2t.main(["prepare","--input",str(source),"--output",str(output)]) == 1
    assert not (output/"video").exists()
    assert not list(output.glob(".prepare-*"))


def test_export_avi_clips_only_preview_and_restores_model_mode(tmp_path):
    class HighOutput(torch.nn.Module):
        def forward(self,x):
            return x[:,-1:] + 2
    model = HighOutput().train()
    record = fake_record()
    record.frames = record.frames[:5]
    record.metadata = {"fps":12.}
    array, video = tmp_path/"result.npy", tmp_path/"result.avi"
    n2t.export_denoised(model,record,n2t.Config(history=2),torch.device("cpu"),array,video)
    restored = np.load(array)
    np.testing.assert_array_equal(restored[:2],record.frames[:2])
    np.testing.assert_allclose(restored[2:],record.frames[2:]+2)
    assert model.training
    cap = n2t.cv2.VideoCapture(str(video))
    for index in range(5):
        ok,frame = cap.read()
        assert ok
        expected = round(float(record.frames[index,0,0])*255) if index < 2 else 255
        assert abs(float(frame.mean())-expected) < 2
    cap.release()
    with pytest.raises(FileExistsError):
        n2t.export_denoised(model,record,n2t.Config(history=2),torch.device("cpu"),array,video)


def test_export_failure_removes_partial_outputs(tmp_path):
    class Broken(torch.nn.Module):
        def forward(self,x):
            raise RuntimeError("inference failure")
    model = Broken().train()
    record = fake_record()
    record.metadata = {"fps":12.}
    with pytest.raises(RuntimeError,match="inference failure"):
        n2t.export_denoised(model,record,n2t.Config(history=2),torch.device("cpu"),
                           tmp_path/"out.npy",tmp_path/"out.avi")
    assert model.training
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("channels",[1,3,4])
def test_png_mask_color_alpha_and_dimensions(tmp_path,channels):
    values = np.zeros((8,10),np.uint8) if channels == 1 else np.zeros((8,10,channels),np.uint8)
    if channels == 1:
        values[2,3] = 255
    else:
        values[2,3,2] = 255  # Colored brush strokes are supported.
        if channels == 4:
            values[2,3,3] = 255
            values[4,5,:3] = 255  # Hidden RGB under zero alpha must not select pixels.
    path = tmp_path/"mask.png"
    assert n2t.cv2.imwrite(str(path),values)
    mask = n2t.load_evaluation_mask(path,(8,10))
    assert mask.sum() == 1 and mask[2,3]
    with pytest.warns(UserWarning,match="resized"):
        resized = n2t.load_evaluation_mask(path,(10,8))
    assert resized.shape == (10,8)

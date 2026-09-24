"""Dataset workflow: raw preservation, AVI round trip, mask policy, full CPU run."""
import json
from pathlib import Path
import shutil

import cv2
import h5py
import numpy as np
import pytest
import torch

from doppler_denoising import noise2time as n2t


def dataset(tmp_path):
    root=tmp_path/"dataset";folder=root/"measure_1"
    (folder/"manual").mkdir(parents=True);(folder/"pseudo").mkdir()
    artery=np.zeros((32,32),bool);artery[4:15,4:9]=True
    vein=np.zeros_like(artery);vein[4:15,15:20]=True
    choroid=np.zeros_like(artery);choroid[12:23,6:25]=True
    t=np.arange(160)/20
    spatial=np.random.default_rng(8).uniform(-2,2,(32,32))
    frames=np.broadcast_to(50+spatial,(160,32,32)).copy()
    frames[:,artery]+=(10*np.sin(2*np.pi*t))[:,None]
    frames[:,vein]+=(3*np.sin(2*np.pi*t))[:,None]
    # Bright exterior pixels must not control scaling or the arterial trace.
    roi=n2t.circle_mask(32,32,15,15,22)
    frames[:,~roi]=5000
    artery[~roi]=True
    frames=frames.astype(np.float32)
    with h5py.File(folder/"measure_1_DV.h5","w") as h5:
        h5.create_dataset("doppler_signal/M0_ff",data=frames)
    for path,mask in [(folder/"manual/retina_artery_mask.png",artery),
                      (folder/"manual/retina_vein_mask.png",vein),
                      (folder/"pseudo/choroidal_vessel_segmentation_choroidal_vessel_mask_raw.png",choroid),
                      (folder/"manual/choroid_vessel_mask.png",np.ones_like(artery))]:
        assert cv2.imwrite(str(path),mask.astype(np.uint8)*255)
    return root,folder,frames


def prepare_command(root,output):
    return ["prepare","--input",str(root),"--output",str(output),"--fps","20",
            "--cx","15","--cy","15","--radius","22"]


def test_original_holodoppler_layout(tmp_path):
    root,folder,raw=dataset(tmp_path)
    with h5py.File(folder/'measure_1_DV.h5','r+') as h5:
        h5.move('doppler_signal/M0_ff','moment0ff')
        h5.create_dataset('moment0',data=raw*2)
    output=tmp_path/'original_layout'
    assert n2t.main(prepare_command(root,output))==0
    record=output/'prepared/measure_1'
    metadata=json.loads((record/'metadata.json').read_text())
    assert metadata['h5_key']=='moment0ff'
    roi=np.load(record/'roi.npy')
    np.testing.assert_allclose(np.load(record/'frames.npy'),np.where(roi,raw/raw[:,roi].max(),0),rtol=1e-6)
    assert n2t.main(prepare_command(root,output)+['--skip-existing'])==0


def test_m0_key_resolution_errors_and_precedence(tmp_path):
    workflow=n2t.load_sibling('dataset_workflow')
    with h5py.File(tmp_path/'keys.h5','w') as h5:
        h5.create_dataset('moment0',data=np.zeros((3,32,32)))
        with pytest.raises(ValueError,match='Available datasets: moment0'):
            workflow.resolve_m0_key(h5)
        h5.create_dataset('moment0ff',data=np.ones((3,32,32)))
        assert workflow.resolve_m0_key(h5)=='moment0ff'
        h5.create_dataset('doppler_signal/M0_ff',data=np.ones((3,32,32)))
        assert workflow.resolve_m0_key(h5)=='doppler_signal/M0_ff'


def test_raw_and_avi_paths(tmp_path):
    root,folder,raw=dataset(tmp_path)
    source_hash=n2t.sha256(folder/"measure_1_DV.h5")
    outputs=[]
    for mode in ("raw","avi"):
        output=tmp_path/mode
        args=prepare_command(root,output)+(["--avi"] if mode=="avi" else [])
        assert n2t.main(args)==0
        record=output/"prepared/measure_1"
        metadata=json.loads((record/"metadata.json").read_text())
        frames=np.load(record/"frames.npy")
        roi=np.load(record/"roi.npy")
        assert metadata["diaphragm_mask_applied"] is True
        assert metadata["intensity_scale"]==float(raw[:,roi].max())
        assert not frames[:,~roi].any()
        artery=np.load(record/"artery_mask.npy")
        assert not artery[~roi].any()
        np.testing.assert_allclose(np.load(record/"brightness_raw.npy"),
                                   frames[:,artery].mean(axis=1),rtol=1e-6)
        assert frames.shape==raw.shape
        assert metadata["first_original_frame"]==0
        assert metadata["input_mode"]==mode
        assert metadata["peak_method"]=="arterial"
        assert (record/"brightness.png").exists()
        cap=cv2.VideoCapture(str(record/"prepared.avi"));decoded=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            decoded.append(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)/255)
        cap.release();decoded=np.stack(decoded)
        assert len(decoded)==len(raw)
        if mode=="raw":
            np.testing.assert_allclose(frames*metadata["intensity_scale"],raw*roi,rtol=1e-7)
        else:
            np.testing.assert_array_equal(frames,decoded*roi)
        loaded=n2t.Record(record)
        assert loaded.phase[0]==-1 and loaded.phase[-1]==-1
        assert (record/'cycle.npy').exists() and (record/'fractional_phase.npy').exists()
        assert loaded.metadata['matching_phase_definition'].startswith('fractional position')
        assert len(loaded.eligible(9))>0
        for t in loaded.eligible(9)[::25]:
            for donor in loaded.matching_donors(t):
                assert donor.cycle != loaded.cycle[t]
                assert loaded.cycle[donor.lower] == donor.cycle
                assert loaded.cycle[donor.upper] == donor.cycle
        for donors in loaded.donors.values():
            assert loaded.valid[donors].all()
            assert (loaded.phase[donors]>=0).all()
        assert n2t.main(args+["--skip-existing"])==0
        outputs.append(frames)
    assert not np.array_equal(*outputs)
    assert n2t.sha256(folder/"measure_1_DV.h5")==source_hash
    # Compression mode cannot silently reuse a raw preparation.
    assert n2t.main(prepare_command(root,tmp_path/"raw")+["--avi","--skip-existing"])==1
    workflow=n2t.load_sibling("dataset_workflow")
    paths,origin=workflow.choroidal_masks(folder)
    assert len(paths)==1 and paths[0].parent.name=="pseudo"
    # Preparations made before the diaphragm fix must not be silently reused.
    path=tmp_path/"raw/prepared/measure_1/metadata.json"
    metadata=json.loads(path.read_text());metadata.pop("diaphragm_mask_applied")
    n2t.write_json(path,metadata)
    assert n2t.main(prepare_command(root,tmp_path/"raw")+["--skip-existing"])==1
    with pytest.raises(ValueError,match="lacks diaphragm masking"):
        n2t.main(["train","--input",str(root),"--output",str(tmp_path/"raw")])


def test_existing_avi_can_link_dataset_masks_for_benchmark(tmp_path):
    root,folder,raw=dataset(tmp_path)
    avi=folder/'specific_input.avi'
    writer=cv2.VideoWriter(str(avi),cv2.VideoWriter_fourcc(*'MJPG'),20.,(32,32),True)
    assert writer.isOpened()
    scale=float(raw.max())
    for frame in raw:
        gray=np.rint(np.clip(frame/scale,0,1)*255).astype(np.uint8)
        writer.write(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR))
    writer.release()
    record=tmp_path/'prepared'/'measure_1'
    assert n2t.main(['prepare','--input',str(avi),'--output',str(record),
                     '--dataset-measure',str(folder),'--fps','20',
                     '--cx','15','--cy','15','--radius','22'])==0
    metadata=json.loads((record/'metadata.json').read_text())
    assert metadata['schema']=='noise2time.linked_avi.v1'
    assert metadata['input_mode']=='source_avi'
    assert metadata['diaphragm_mask_applied'] is True
    assert Path(metadata['dataset_measure'])==folder.resolve()
    loaded=n2t.Record(record)
    assert not loaded.frames[:,~loaded.roi].any()
    assert loaded.valid.any() and (~loaded.valid).any()
    monitor=n2t.load_sibling('training_monitor').Monitor(loaded,2,n2t,2)
    assert set(monitor.masks)=={'retinal_artery','retinal_vein','choroidal','background'}


def test_full_dataset_cpu_workflow(tmp_path):
    root,folder,raw=dataset(tmp_path)
    output=tmp_path/"output"
    assert n2t.main(prepare_command(root,output))==0
    config=tmp_path/"config.json"
    n2t.write_json(config,dict(base_channels=8,history=2,block_size=8,blocks=1,
                              objective="l2",epochs=1,samples_per_epoch=2,batch_size=2,validation_samples=2))
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        assert n2t.main(["train","--input",str(root),"--output",str(output),"--config",str(config),"--device","cpu"])==0
        assert (output/"runs/best.pt").exists()
        assert (output/"runs/previews/epoch_001/measure_1.avi").exists()
        assert n2t.main(["evaluate","--input",str(root),"--output",str(output),"--device","cpu",
                         "--cardiac-hz","1","--local-size","8"])==0
    finally:
        torch.set_num_threads(previous_threads)
    evaluation=output/"evaluation/measure_1"
    assert (evaluation/"report.html").exists()
    sources=json.loads((evaluation/"dataset_sources.json").read_text())
    assert "pseudo" in sources["choroidal_mask_origin"]
    assert all("pseudo" in p for p in sources["choroidal_masks"])
    metrics=json.loads((evaluation/"metrics.json").read_text())
    assert metrics["frames_scored"]==len(raw)-2
    assert metrics["first_original_scored_frame"]==2
    assert metrics["denoised_provenance"]["input_mode"]=="raw"
    assert metrics["denoised_provenance"]["intensity_scale"]>1
    roi=np.load(output/"prepared/measure_1/roi.npy")
    denoised=np.load(output/"runs/denoised/measure_1/denoised.npy")
    assert not denoised[:,~roi].any()


def test_missing_pseudo_does_not_fall_back_to_manual(tmp_path):
    folder=tmp_path/"measure";(folder/"manual").mkdir(parents=True)
    assert cv2.imwrite(str(folder/"manual/choroid_vessel_mask.png"),np.ones((32,32),np.uint8)*255)
    with pytest.raises(ValueError,match="pseudo"):
        n2t.load_sibling("dataset_workflow").choroidal_masks(folder)


def test_resume_matches_uninterrupted_training(tmp_path):
    root, folder, raw = dataset(tmp_path)
    output = tmp_path/'resumed'
    assert n2t.main(prepare_command(root, output)) == 0
    direct = tmp_path/'direct'
    shutil.copytree(output/'prepared', direct/'prepared')
    config = tmp_path/'config.json'
    n2t.write_json(config, dict(base_channels=8, history=2, block_size=8, blocks=1,
                              objective='l2', epochs=1, samples_per_epoch=2, batch_size=2, validation_samples=2))
    def command(destination):
        return ['train', '--input', str(root), '--output', str(destination),
                '--config', str(config), '--device', 'cpu', '--no-epoch-previews']
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        assert n2t.main(command(output)) == 0
        assert n2t.main(command(output)+['--resume', '--epochs', '2']) == 0
        assert n2t.main(command(direct)+['--epochs', '2']) == 0
    finally:
        torch.set_num_threads(previous)
    resumed = torch.load(output/'runs/last.pt', weights_only=True)
    uninterrupted = torch.load(direct/'runs/last.pt', weights_only=True)
    assert resumed['epoch'] == 2
    for key in resumed['model']:
        torch.testing.assert_close(resumed['model'][key], uninterrupted['model'][key], rtol=0, atol=0)
    rows = [json.loads(line) for line in (output/'runs/metrics.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in rows] == [1, 2]
    assert rows[-1]['diagnostics']['measure_1']['frames_scored'] == len(raw)-2
    assert (output/'runs/metrics.png').exists()
    assert (output/'runs/metrics.csv').exists()
    # Streaming std must match the evaluation definition, not spatial variation.
    monitor = n2t.load_sibling('training_monitor').Monitor(n2t.Record(output/'prepared/measure_1'), 2, n2t, 2)
    for index, frame in enumerate(monitor.record.frames): monitor.update(index, frame*.5)
    result = monitor.result()
    reference = monitor.record.frames[2:,monitor.masks['background']].std(axis=0,dtype=np.float64).mean()
    assert result['background_std_original'] == pytest.approx(reference, abs=1e-12)
    assert result['background_std_denoised'] == pytest.approx(reference*.5, abs=1e-12)
    # Also exercise nonzero temporal variance with a known sinusoidal perturbation.
    monitor.reset()
    for index, frame in enumerate(monitor.record.frames):
        monitor.update(index, frame + .01*np.sin(index))
    assert monitor.result()['background_std_denoised'] == pytest.approx(
        (.01*np.sin(np.arange(2,len(raw)))).std(), abs=1e-8)
    # Older checkpoints still restore model/optimizer and explicitly warn about RNG.
    for key in ('rng', 'torch_rng', 'python_rng', 'cuda_rng', 'stale'):
        resumed.pop(key)
    torch.save(resumed, output/'runs/last.pt')
    with pytest.warns(UserWarning, match='Older checkpoint'):
        assert n2t.main(command(output)+['--resume','--epochs','2']) == 0
    assert torch.load(output/'runs/last.pt', weights_only=True)['epoch'] == 2

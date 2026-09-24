"""Known landmark timing and failure handling; no private dataset dependency."""
from types import SimpleNamespace

import numpy as np
import pytest

from doppler_denoising import arterial_peaks as detector
from doppler_denoising import noise2time as n2t


def pulse_train(onsets,fps=100,n=650):
    t=np.arange(n)/fps;signal=np.full(n,2.);truth=[]
    for start in onsets:
        z=np.maximum(t-start,0)
        pulse=(1-np.exp(-z/.025))*np.exp(-z/.16)
        signal+=pulse;truth.append(np.argmax(pulse))
    return signal,np.array(truth)


@pytest.mark.parametrize('onsets',[
    [.4,1.4,2.4,3.4,4.4,5.4],
    [.4,1.2,2.3,3.2,4.4,5.3],
    [.4,1.4,2.4,4.4,5.4],
])
def test_known_maxima_and_missing_beat(onsets):
    raw,truth=pulse_train(onsets)
    result=detector.detect_arterial_peaks(raw,100)
    assert len(result['peaks']) == len(truth)
    assert np.max(np.abs(result['peaks']-truth))<=3
    assert np.all(result['upstrokes']<=result['peaks'])


def test_spikes_invariance_and_no_input_mutation():
    raw,truth=pulse_train([.4,1.4,2.4,3.4,4.4,5.4])
    raw[195:198]-=1.5;raw[475:477]+=1.5
    original=raw.copy();raw.flags.writeable=False
    result=detector.detect_arterial_peaks(raw,100)
    assert len(result['peaks'])==len(truth)
    assert np.max(np.abs(result['peaks']-truth))<=3
    assert result['artifact_mask'][196] and result['artifact_mask'][475]
    np.testing.assert_array_equal(raw,original)
    transformed=detector.detect_arterial_peaks(3*raw+17,100)
    np.testing.assert_array_equal(result['peaks'],transformed['peaks'])


@pytest.mark.parametrize('signal,fps',[(np.ones(100),100),(np.ones(8),100),
    (np.r_[np.zeros(99),np.nan],100),(np.ones((10,10)),100),(np.arange(100),0)])
def test_reject_invalid_signals(signal,fps):
    with pytest.raises(ValueError): detector.detect_arterial_peaks(signal,fps)


def test_nonperiodic_signal_is_flagged():
    signal=np.random.default_rng(123).normal(size=1000)
    result=detector.detect_arterial_peaks(signal,100)
    assert result['warnings']


def test_preparation_uses_arterial_timing(tmp_path):
    raw,truth=pulse_train([.4,1.4,2.4,3.4,4.4,5.4],fps=40,n=260)
    mask=np.zeros((32,32),bool);mask[10:20,10:20]=True
    frames=np.full((260,32,32),.2,np.float32)
    frames[:,mask]=(.1*raw)[:,None]
    source=tmp_path/'source.npy';np.save(source,frames)
    mask_path=tmp_path/'artery.png';assert n2t.cv2.imwrite(str(mask_path),mask.astype(np.uint8)*255)
    output=tmp_path/'prepared'
    args=SimpleNamespace(input=source,output=output,fps=40,cx=15,cy=15,radius=22,
                         smooth_window=5,peak_distance=15,prominence=.1,brightness_mask=None,
                         artery_mask=mask_path,peak_method='auto',peak_min_hz=.5,peak_max_hz=2.5)
    n2t.prepare_one(args)
    import json
    metadata=json.loads((output/'metadata.json').read_text())
    peaks=np.array(metadata['peaks'])+metadata['first_original_frame']
    assert len(peaks)==len(truth)
    assert np.max(np.abs(peaks-truth))<=2
    assert metadata['peak_method']=='arterial'
    assert (output/'arterial_peak_diagnostics.npz').exists()

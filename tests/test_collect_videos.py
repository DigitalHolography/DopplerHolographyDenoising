import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from doppler_denoising import collect_videos as collector


def make_measure(root,name="measure",h5=False):
    folder = root/name
    source = collector.source_for(folder,h5)
    metadata = collector.metadata_sources(folder)
    metadata["version"].parent.mkdir(parents=True,exist_ok=True)
    metadata["parameters"].parent.mkdir(parents=True,exist_ok=True)
    metadata["version"].write_text("py0.6.0",encoding="utf-8")
    metadata["parameters"].write_text('{"sampling_freq": 37000}',encoding="utf-8")
    source.parent.mkdir(parents=True,exist_ok=True)
    if not h5:
        source.write_bytes(b"avi-test-data")
    return folder,source


def test_names_comments_bom_and_deduplication(tmp_path):
    text = tmp_path/"names.txt"
    text.write_text("\ufeff# comment\nmeasure\n\nMEASURE\nmeasure2\n",encoding="utf-8")
    assert collector.read_measures(text) == ["measure","measure2"]
    text.write_text("../escape",encoding="utf-8")
    with pytest.raises(ValueError):
        collector.read_measures(text)


def test_scan_bounded_and_does_not_enter_measurements(tmp_path,monkeypatch):
    measure,_ = make_measure(tmp_path/"folder1")
    other = tmp_path/"folder1"/"unrelated"; other.mkdir()
    scanned = []
    scan = collector.scan_folder
    def tracked(path):
        scanned.append(Path(path))
        return scan(path)
    monkeypatch.setattr(collector,"scan_folder",tracked)
    matches,errors = collector.find_measurements(["measure"],[tmp_path])
    assert matches["measure"] == [measure]
    assert not errors
    assert set(scanned) == {tmp_path,tmp_path/"folder1"}
    matches,_ = collector.find_measurements(["measure"],[tmp_path],max_depth=1)
    assert matches["measure"] == []


def test_overlapping_roots_do_not_duplicate_sources(tmp_path):
    measure,_ = make_measure(tmp_path/"folder1")
    matches,errors = collector.find_measurements(["measure"],[tmp_path,tmp_path/"folder1",measure],2)
    assert matches["measure"] == [measure]
    assert not errors


def test_copy_skip_overwrite_ambiguity_and_missing(tmp_path):
    folder,source = make_measure(tmp_path/"root")
    output = tmp_path/"output"; output.mkdir()
    row = collector.collect_one("measure",[folder],output)
    assert row["status"] == "copied"
    destination = Path(row["destination"])
    version = output/"measure_version_holodoppler.txt"
    parameters = output/"measure_parameters_holodoppler.json"
    assert version.read_text(encoding="utf-8") == "py0.6.0"
    assert json.loads(parameters.read_text(encoding="utf-8")) == {"sampling_freq":37000}
    assert destination.read_bytes() == source.read_bytes()
    source.write_bytes(b"changed")
    assert collector.collect_one("measure",[folder],output)["status"] == "exists"
    assert destination.read_bytes() != source.read_bytes()
    assert collector.collect_one("measure",[folder],output,overwrite=True)["status"] == "copied"
    assert destination.read_bytes() == b"changed"
    second,_ = make_measure(tmp_path/"root2")
    assert collector.collect_one("measure",[folder,second],output)["status"] == "ambiguous"
    assert collector.collect_one("missing",[],output)["status"] == "missing"


def test_missing_metadata_does_not_publish_video(tmp_path):
    folder,_ = make_measure(tmp_path/"root")
    collector.metadata_sources(folder)["version"].unlink()
    output = tmp_path/"output"; output.mkdir()
    row = collector.collect_one("measure",[folder],output)
    assert row["status"] == "missing"
    assert row["missing_metadata"].endswith("version_holodoppler.txt")
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("axis",[0,1,2])
def test_h5_extract_preserves_dtype_values_and_frame_axis(tmp_path,axis):
    folder,source = make_measure(tmp_path/"root",h5=True)
    expected = np.arange(5*3*4,dtype=np.float64).reshape(5,3,4)
    with h5py.File(source,"w") as handle:
        handle.create_dataset("moment0ff",data=np.moveaxis(expected,0,axis),compression="gzip")
    output = tmp_path/"output"; output.mkdir()
    row = collector.collect_one("measure",[folder],output,h5=True,frame_axis=axis,chunk_mb=1)
    assert row["status"] == "copied"
    actual = np.load(row["destination"])
    np.testing.assert_array_equal(actual,expected)
    assert actual.dtype == expected.dtype
    assert row["normalization"] == "none"


def test_invalid_h5_leaves_no_partial_output(tmp_path):
    folder,source = make_measure(tmp_path/"root",h5=True)
    with h5py.File(source,"w") as handle:
        handle.create_dataset("wrong_key",data=[1,2])
    output = tmp_path/"output"; output.mkdir()
    row = collector.collect_one("measure",[folder],output,h5=True,frame_axis=0)
    assert row["status"] == "error"
    assert list(output.iterdir()) == []


def test_scan_failure_is_reported(tmp_path):
    matches,errors = collector.find_measurements(["measure"],[tmp_path/"not_mounted"])
    assert matches["measure"] == []
    assert len(errors) == 1


def test_cli_dry_run_report_and_real_copy(tmp_path):
    _,source = make_measure(tmp_path/"root")
    text = tmp_path/"names.txt"; text.write_text("measure\nmissing\n",encoding="utf-8")
    roots = tmp_path/"roots.txt"; roots.write_text(str(tmp_path),encoding="utf-8")
    output = tmp_path/"output"
    args = ["--measures",str(text),"--folders-file",str(roots),"--output",str(output)]
    assert collector.main(args+["--dry-run"]) == 1
    assert not list(output.glob("*.avi"))
    report = json.loads(next(output.glob("collection_*.json")).read_text())
    assert report["counts"] == {"planned":1,"missing":1}
    text.write_text("measure",encoding="utf-8")
    assert collector.main(args) == 0
    assert (output/"measure_HD_M0.avi").read_bytes() == source.read_bytes()
    assert (output/"measure_version_holodoppler.txt").is_file()
    assert (output/"measure_parameters_holodoppler.json").is_file()


def test_cli_requires_h5_frame_axis(tmp_path):
    with pytest.raises(SystemExit) as exc:
        collector.main(["--measures","unused.txt","--folders",str(tmp_path),
                        "--output",str(tmp_path/"out"),"--h5"])
    assert exc.value.code == 2

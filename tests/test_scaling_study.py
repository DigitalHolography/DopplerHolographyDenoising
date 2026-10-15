import json

import numpy as np
import torch

from doppler_denoising.scaling_study.evaluation import icc_2_1, repeatability
from doppler_denoising.scaling_study.manifest import _jobs, _split_patients, default_protocol, parse_measure_name
from doppler_denoising.scaling_study.sampling import PatientBalancedSampler
from doppler_denoising.scaling_study.training import train_steps


def test_single_frame_pretraining(tmp_path):
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    try:
        records = []
        rng = np.random.default_rng(3)
        for name in ("train", "validation"):
            record = tmp_path / name
            record.mkdir()
            frames = rng.uniform(.2, .8, (12, 32, 32)).astype(np.float32)
            np.save(record / "frames.npy", frames)
            np.save(record / "roi.npy", np.ones((32, 32), dtype=bool))
            np.save(record / "brightness.npy", frames.mean(axis=(1, 2)))
            np.save(record / "phase.npy", np.arange(12) % 4)
            (record / "metadata.json").write_text(json.dumps({"peaks": [0, 4, 8]}))
            records.append(record)
        config = tmp_path / "config.json"
        config.write_text(json.dumps(dict(history=0, convlstm=False, objective="l2",
                                         base_channels=8, block_size=8, blocks=1)))
        output = tmp_path / "run"
        checkpoint = train_steps(
            train_records=records[:1], train_patients=["P1"],
            validation_records=records[1:], config=config, output=output,
            max_steps=1, eval_every_steps=1, checkpoint_every_steps=1,
            validation_anchors_per_video=2, device="cpu",
        )
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert saved["config"]["history"] == 0
        assert saved["global_step"] == 1
        assert np.isfinite(saved["best_validation"])
        assert (output / "last.pt").exists()
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)


def test_measure_parser_does_not_treat_suffix_as_repeat_role():
    assert parse_measure_name("250604_LIJ_R_4") == {
        "date": "250604", "patient": "LIJ", "eye": "R", "suffix": "4"
    }


def test_patient_split_and_jobs_are_nested_and_patient_disjoint():
    patients = [f"P{i:02d}" for i in range(22)]
    rows = [{"patient": patient, "eye": eye, "repeat_role": role,
             "measure": f"000000_{patient}_{eye}_{role}"}
            for patient in patients for eye in "LR" for role in "ABC"]
    protocol = default_protocol(); split = _split_patients(patients, 7, protocol["patient_split"])
    assert not (set(split["pretraining"]) & set(split["validation"]))
    assert not (set(split["pretraining"]) & set(split["test"]))
    jobs = _jobs(rows, split, protocol, 7)
    first_order = sorted((j for j in jobs if j["kind"] == "scaling" and j["order"] == 1),
                         key=lambda j: j["patient_count"])
    for smaller, larger in zip(first_order, first_order[1:]):
        assert set(smaller["patients"]) < set(larger["patients"])
    assert [len(j["measures"]) for j in first_order] == [6, 12, 24, 48, 72, 84]


def test_patient_balanced_sampler_is_not_anchor_count_weighted():
    rng = np.random.default_rng(2)
    sampler = PatientBalancedSampler([[1], list(range(1000))], ["short", "long"], rng)
    choices = sampler.draw(20_000)
    fraction_long = np.mean([record == 1 for record, _ in choices])
    assert 0.48 < fraction_long < 0.52


def test_repeatability_statistics_use_b_and_c_pairs():
    rows = []
    for patient, values in (("P1", (1.0, 1.1)), ("P2", (2.0, 2.1)), ("P3", (3.0, 3.1))):
        for role, value in zip("BC", values):
            rows.append({"patient": patient, "eye": "L", "repeat_role": role, "metric": value})
    result = repeatability(rows, "metric")
    assert result["n_subject_eyes"] == 3
    assert result["icc_2_1"] > .9
    assert np.isclose(result["bland_altman_bias"], .1)
    assert icc_2_1([[1, 1], [2, 2], [3, 3]]) == 1.0

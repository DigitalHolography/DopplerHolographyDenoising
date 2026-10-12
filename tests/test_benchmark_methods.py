"""Focused invariants for the external self-supervised benchmark methods."""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
import cv2

from test_noise2time_benchmark import prepared


from doppler_denoising.training import method_runner as runner
from doppler_denoising.methods import blind2unblind, neighbor2neighbor, udvd
from doppler_denoising.benchmark import runner as benchmark
from doppler_denoising import noise2time as n2t


@contextmanager
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def method_record(tmp_path):
    """Upscale the common prepared fixture to the methods' 64-pixel patch."""
    source = prepared(tmp_path / "fixture")
    destination = tmp_path / "method_prepared" / source.name
    destination.parent.mkdir(parents=True)
    shutil.copytree(source.path, destination)
    frames = np.repeat(np.repeat(np.asarray(source.frames), 2, axis=1), 2, axis=2)
    roi = np.repeat(np.repeat(np.asarray(source.roi), 2, axis=0), 2, axis=1)
    np.save(destination / "frames.npy", frames.astype(np.float32))
    np.save(destination / "roi.npy", roi.astype(bool))
    np.save(destination / "valid_frames.npy", np.asarray(source.valid, dtype=bool))
    # Keep linked evaluation masks consistent with the enlarged method fixture.
    dataset_measure = Path(source.metadata["dataset_measure"])
    for mask_path in dataset_measure.rglob("*.png"):
        mask = cv2.imdecode(np.fromfile(mask_path, np.uint8), cv2.IMREAD_UNCHANGED)
        if mask is not None:
            ok, encoded = cv2.imencode(".png", cv2.resize(mask, (64, 64), interpolation=cv2.INTER_NEAREST))
            assert ok; encoded.tofile(mask_path)
    return destination


@pytest.mark.parametrize("method", runner.METHODS)
def test_default_configs_are_complete_and_valid(method):
    config = runner.default_config(
        method,
        {"epochs": 12, "batch_size": 3, "validation_samples": 2, "seed": 7},
    )
    runner.validate_config(method, config)
    assert config["schema"] == runner.SCHEMA
    assert config["method"] == method
    assert config["epochs"] == 12
    assert config["batch_size"] == 3
    assert config["validation_samples"] == 2
    if method == "udvd":
        assert config["temporal_frames"] == 5
        assert config["patch_size"] == 128
    elif method == "neighbor2neighbor":
        assert config["schedule_epochs"] == config["epochs"]
        assert config["increase_ratio"] >= 0
        assert config["patch_size"] == 256
    else:
        schedule = config["beta_schedule"]
        assert schedule["schedule_epochs"] == config["epochs"]
        assert 0 <= schedule["start_ratio"] < schedule["end_ratio"] <= 1
        assert config["patch_size"] == 128


def test_epoch_override_updates_derived_schedules_and_unknown_keys_fail():
    neighbor = runner.default_config("neighbor2neighbor", {"epochs": 6},
                                     overrides={"epochs": 20})
    blind = runner.default_config("blind2unblind", {"epochs": 6},
                                  overrides={"epochs": 20})
    assert neighbor["schedule_epochs"] == 20
    assert blind["beta_schedule"]["schedule_epochs"] == 20
    with pytest.raises(ValueError, match="Unknown udvd configuration fields"):
        runner.default_config("udvd", {"epochs": 6}, overrides={"epohcs": 20})


def test_benchmark_selects_and_adds_method_strategies():
    base = n2t.Config(epochs=6)
    selected = benchmark.variants(base, "held_out", ["baseline", *runner.METHODS])
    assert selected["baseline"]["objective"] == base.objective
    for method in runner.METHODS:
        assert selected[method]["method"] == method
        assert selected[method]["epochs"] == 6
    plan = {"variants": {"baseline": selected["baseline"]},
            "validation_video": "held_out"}
    added = benchmark.add_simple_variants(
        plan, {"experiments": list(runner.METHODS)}, list(runner.METHODS)
    )
    assert added == list(runner.METHODS)
    assert set(plan["variants"]) == {"baseline", *runner.METHODS}


def test_method_benchmark_dry_run_writes_current_plan(tmp_path):
    record = method_record(tmp_path)
    config = tmp_path / "base.json"
    n2t.write_json(config, {"epochs": 1, "samples_per_epoch": 3,
                            "batch_size": 1, "validation_samples": 2,
                            "history": 2, "base_channels": 8,
                            "block_size": 8, "blocks": 1, "objective": "l2"})
    selection = tmp_path / "methods.json"
    n2t.write_json(selection, {
        "experiments": list(runner.METHODS),
        "configs": {
            "udvd": {"patch_size": 64},
            "blind2unblind": {"patch_size": 64, "features": 2,
                               "mask_width": 2, "mask_chunk_size": 2},
            "neighbor2neighbor": {"patch_size": 64, "features": 2},
        },
    })
    output = tmp_path / "benchmark"
    assert benchmark.main(["--prepared", str(record), "--output", str(output),
                           "--config", str(config), "--experiments", str(selection),
                           "--device", "cpu", "--dry-run"]) == 0
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert plan["schema"] == "doppler_denoising.benchmark.v2"
    assert list(plan["variants"]) == list(runner.METHODS)


@pytest.mark.parametrize(
    "method,mutation,match",
    [
        ("udvd", {"temporal_frames": 3}, "exactly five"),
        ("neighbor2neighbor", {"patch_size": 96}, "divisible by 64"),
        ("blind2unblind", {"mask_width": 1}, "mask configuration"),
    ],
)
def test_method_specific_config_validation(method, mutation, match):
    config = runner.default_config(method, {"epochs": 2})
    config.update(mutation)
    with pytest.raises(ValueError, match=match):
        runner.validate_config(method, config)


def test_neighbor_pairs_are_all_distinct_orthogonal_neighbors_with_balanced_coverage():
    pairs = neighbor2neighbor._PAIRS.cpu().numpy()
    coordinates = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    assert len({tuple(pair) for pair in pairs}) == 8
    assert np.all(pairs[:, 0] != pairs[:, 1])
    assert np.all(np.abs(coordinates[pairs[:, 0]] - coordinates[pairs[:, 1]]).sum(1) == 1)
    # Every site in a 2x2 cell occurs equally often as source and as target.
    np.testing.assert_array_equal(np.bincount(pairs[:, 0], minlength=4), np.full(4, 2))
    np.testing.assert_array_equal(np.bincount(pairs[:, 1], minlength=4), np.full(4, 2))

    generator = torch.Generator(device="cpu").manual_seed(11)
    first, second = neighbor2neighbor.pair_indices(3, 8, 10, generator, "cpu")
    assert first.shape == second.shape == (3, 4, 5)
    sampled = torch.stack((first, second), -1).reshape(-1, 2).numpy()
    assert set(map(tuple, sampled)).issubset(set(map(tuple, pairs)))


def test_neighbor_subsample_and_roi_intersection_cover_the_selected_pixels():
    image = torch.arange(16.0).reshape(1, 1, 4, 4)
    # Select each of the four cell offsets once across four batches.
    indices = torch.arange(4).reshape(4, 1, 1).expand(4, 2, 2)
    values = neighbor2neighbor.subsample(image.expand(4, -1, -1, -1), indices)
    for offset in range(4):
        expected = torch.nn.functional.pixel_unshuffle(image, 2)[:, offset : offset + 1]
        torch.testing.assert_close(values[offset : offset + 1], expected)

    roi = torch.ones_like(image)
    roi[..., 0, 0] = 0
    source = neighbor2neighbor.subsample(roi, torch.zeros((1, 2, 2), dtype=torch.long))
    target = neighbor2neighbor.subsample(roi, torch.ones((1, 2, 2), dtype=torch.long))
    valid = source * target
    assert valid.sum().item() == 3
    assert valid[0, 0, 0, 0].item() == 0


def test_blind2unblind_masks_partition_every_pixel_once_and_beta_is_piecewise_linear():
    masks = [
        blind2unblind.fixed_mask(2, 7, 9, 4, position, "cpu", torch.float32)
        for position in range(16)
    ]
    coverage = torch.stack(masks).sum(0)
    torch.testing.assert_close(coverage, torch.ones((2, 1, 7, 9)))
    assert all(torch.logical_or(mask == 0, mask == 1).all() for mask in masks)

    config = runner.default_config("blind2unblind", {"epochs": 100})
    assert blind2unblind.beta_at(1, config) == pytest.approx(2.0)
    assert blind2unblind.beta_at(40, config) == pytest.approx(2.0)
    assert blind2unblind.beta_at(70, config) == pytest.approx(11.0)
    assert blind2unblind.beta_at(100, config) == pytest.approx(20.0)


@pytest.mark.parametrize(
    "method,implementation",
    [("neighbor2neighbor", neighbor2neighbor), ("blind2unblind", blind2unblind)],
)
def test_image_method_forward_shapes_and_losses_are_finite(method, implementation):
    config = runner.default_config(method, {"epochs": 2})
    config.update(features=2, patch_size=64)
    if method == "blind2unblind":
        config.update(mask_width=2, mask_chunk_size=2)
    runner.validate_config(method, config)
    generator = torch.Generator(device="cpu").manual_seed(5)
    image = torch.rand((1, 1, 64, 64), generator=generator)
    roi = torch.ones_like(image)
    with one_torch_thread():
        model = implementation.build_model(config)
        output = implementation.predict(model.eval(), image, config)
        assert output.shape == image.shape
        model.train()
        parts = implementation.loss(model, image, roi, config, 1, generator)
        assert torch.isfinite(parts["total"])
        if method == "neighbor2neighbor":
            # Official code numbers epochs from one: 1/2 * increase_ratio.
            assert parts["regularizer_weight"] == pytest.approx(1.0)
        parts["total"].backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_udvd_forward_loss_and_centre_blind_spot_invariance():
    config = runner.default_config("udvd", {"epochs": 1})
    generator = torch.Generator(device="cpu").manual_seed(19)
    image = torch.rand((1, 5, 8, 8), generator=generator)
    changed = image.clone()
    changed[0, 2, 4, 4] += 10
    roi = torch.ones((1, 1, 8, 8))
    with one_torch_thread():
        torch.manual_seed(3)
        model = udvd.build_model(config).eval()
        output = udvd.predict(model, image, config)
        changed_output = udvd.predict(model, changed, config)
        assert output.shape == (1, 1, 8, 8)
        torch.testing.assert_close(output[..., 4, 4], changed_output[..., 4, 4], atol=1e-6, rtol=0)
        parts = udvd.loss(model, image, roi, config, 1, generator)
        assert torch.isfinite(parts["total"])


def test_udvd_temporal_split_keeps_every_window_inside_its_partition(tmp_path):
    record = runner.PreparedRecord(method_record(tmp_path))
    config = runner.default_config(
        "udvd",
        {
            "epochs": 1,
            "samples_per_epoch": 2,
            "validation_samples": 2,
            "split_mode": "temporal",
            "validation_fraction": 0.25,
        },
    )
    config["patch_size"] = 64
    train, valid, ranges = runner.split_samples("udvd", [record], config)
    cut = ranges[record.name]["train"][1]
    train_support = {frame for _, centre in train for frame in range(centre - 2, centre + 3)}
    valid_support = {frame for _, centre in valid for frame in range(centre - 2, centre + 3)}
    assert train_support <= set(range(0, cut))
    assert valid_support <= set(range(cut, len(record.frames)))
    assert train_support.isdisjoint(valid_support)


class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.5))

    def forward(self, images):
        return images * self.scale


class TinyMethod:
    @staticmethod
    def build_model(config):
        del config
        return TinyDenoiser()

    @staticmethod
    def predict(model, images, config):
        del config
        return model(images)

    @staticmethod
    def loss(model, images, roi, config, epoch, generator):
        del config, epoch, generator
        error = (model(images) - images).square()
        value = (error * roi).sum() / roi.sum().clamp_min(1)
        return {"total": value, "reconstruction": value}


def test_denoised_float_artifact_preserves_shape_roi_and_exclusion_metadata(tmp_path, monkeypatch):
    record_path = method_record(tmp_path)
    record = runner.PreparedRecord(record_path)
    config = runner.default_config("neighbor2neighbor", {"epochs": 1})
    config.update(features=2, patch_size=64, inference_batch_size=7)
    model = TinyMethod.build_model(config)
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "schema": runner.SCHEMA,
            "method": "neighbor2neighbor",
            "model": model.state_dict(),
            "config": config,
            "epoch": 1,
        },
        checkpoint,
    )
    original_loader = runner.method_module
    monkeypatch.setattr(
        runner,
        "method_module",
        lambda name: TinyMethod if name == "neighbor2neighbor" else original_loader(name),
    )
    output = tmp_path / "denoised"
    runner.denoise(
        SimpleNamespace(
            method="neighbor2neighbor",
            record=str(record_path),
            checkpoint=str(checkpoint),
            output=str(output),
            device="cpu",
            excluded_prefix=3,
            npy_only=True,
        )
    )
    bundle = output / record.name
    denoised = np.load(bundle / "denoised.npy")
    metadata = json.loads((bundle / "denoised.json").read_text(encoding="utf-8"))
    assert denoised.dtype == np.float32
    assert denoised.shape == record.frames.shape
    np.testing.assert_allclose(denoised[:, record.roi],
                               np.asarray(record.frames)[:, record.roi] * 1.5)
    assert not denoised[:, ~record.roi].any()
    assert metadata["method"] == "neighbor2neighbor"
    assert metadata["copied_prefix"] == 0
    assert metadata["excluded_prefix"] == 3
    assert metadata["diaphragm_mask_applied"] is True
    assert metadata["checkpoint_sha256"] == runner.sha256(checkpoint)


def test_interrupted_training_resumes_bit_exactly_with_tiny_model(tmp_path, monkeypatch):
    record_path = method_record(tmp_path)
    config = runner.default_config(
        "neighbor2neighbor",
        {
            "epochs": 2,
            "samples_per_epoch": 2,
            "batch_size": 1,
            "validation_samples": 1,
            "seed": 29,
            "patience": 5,
        },
    )
    config.update(features=2, patch_size=64)
    config_path = tmp_path / "method_config.json"
    runner.write_json(config_path, config)
    original_loader = runner.method_module
    monkeypatch.setattr(
        runner,
        "method_module",
        lambda name: TinyMethod if name == "neighbor2neighbor" else original_loader(name),
    )

    def arguments(output, resume=False):
        return SimpleNamespace(
            method="neighbor2neighbor",
            records=[str(record_path)],
            output=str(output),
            config=str(config_path),
            device="cpu",
            resume=resume,
                no_epoch_metrics=True,
                no_epoch_previews=True,
                no_progress=True,
        )

    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    with one_torch_thread():
        runner.train(arguments(uninterrupted))
        real_run_stage = runner.run_stage
        calls = 0

        def interrupt_before_second_epoch(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("simulated interruption")
            return real_run_stage(*args, **kwargs)

        monkeypatch.setattr(runner, "run_stage", interrupt_before_second_epoch)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            runner.train(arguments(resumed))
        monkeypatch.setattr(runner, "run_stage", real_run_stage)
        runner.train(arguments(resumed, resume=True))

    expected = torch.load(uninterrupted / "last.pt", map_location="cpu", weights_only=True)
    actual = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
    assert expected["epoch"] == actual["epoch"] == 2
    assert expected["best_validation"] == actual["best_validation"]
    for key, value in expected["model"].items():
        torch.testing.assert_close(value, actual["model"][key], rtol=0, atol=0)
    assert (uninterrupted / "metrics.jsonl").read_bytes() == (resumed / "metrics.jsonl").read_bytes()
    torch.testing.assert_close(expected["sampling_rng"], actual["sampling_rng"], rtol=0, atol=0)


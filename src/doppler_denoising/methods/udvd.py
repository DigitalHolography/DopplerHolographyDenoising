"""UDVD adapter for one-channel, five-frame Doppler sequences."""
import torch

from .udvd_arch import BlindVideoNet


def build_model(config):
    del config
    return BlindVideoNet(channels_per_frame=1, out_channels=1, bias=False,
                         blind=True, sigma_known=True)


def predict(model, images, config):
    del config
    prediction, _ = model(images)
    return prediction


def loss(model, images, roi, config, epoch, generator):
    del config, epoch, generator
    target = images[:, 2:3]
    prediction = predict(model, images, None)
    squared = (prediction - target).square() * roi
    value = squared.sum() / roi.sum().clamp_min(1)
    return {"total": value, "reconstruction": value}


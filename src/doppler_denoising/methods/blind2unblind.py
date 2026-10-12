"""Clean-room Blind2Unblind implementation from the paper equations.

The authors' public repository has no redistribution license.  This file uses
the published global mask mapper and re-visible objective, with the separately
BSD-licensed Neighbor2Neighbor U-Net as its grayscale backbone.
"""
import torch
import torch.nn.functional as F

from .neighbor_arch import UNet


def build_model(config):
    return UNet(in_nc=1, out_nc=1, n_feature=int(config["features"]),
                blindspot=False)


def fixed_mask(batch, height, width, mask_width, position, device, dtype):
    rows = torch.arange(height, device=device)[:, None]
    columns = torch.arange(width, device=device)[None, :]
    pattern = ((rows % mask_width) * mask_width + columns % mask_width) == position
    return pattern.to(dtype)[None, None].expand(batch, 1, height, width)


def _interpolate(images, mask):
    kernel = torch.tensor([[.5, 1., .5], [1., 0., 1.], [.5, 1., .5]],
                          dtype=images.dtype, device=images.device)
    kernel = (kernel / kernel.sum())[None, None]
    neighbors = F.conv2d(images, kernel, padding=1)
    return images * (1 - mask) + neighbors * mask


def blind_prediction(model, images, config):
    """Assemble one prediction per hidden pixel without a huge 16x batch."""
    batch, _, height, width = images.shape
    mask_width = int(config["mask_width"])
    chunk_size = int(config["mask_chunk_size"])
    result = torch.zeros_like(images)
    positions = list(range(mask_width * mask_width))
    for start in range(0, len(positions), chunk_size):
        chunk = positions[start:start + chunk_size]
        masked, masks = [], []
        for position in chunk:
            mask = fixed_mask(batch, height, width, mask_width, position,
                              images.device, images.dtype)
            masked.append(_interpolate(images, mask)); masks.append(mask)
        predictions = model(torch.cat(masked, dim=0))
        for prediction, mask in zip(predictions.chunk(len(chunk), dim=0), masks):
            result = result + prediction * mask
    return result


def beta_at(epoch, config):
    schedule = config["beta_schedule"]
    progress = min(max(float(epoch) / float(schedule["schedule_epochs"]), 0.), 1.)
    start, end = float(schedule["start_ratio"]), float(schedule["end_ratio"])
    low, high = float(schedule["start_beta"]), float(schedule["end_beta"])
    if progress <= start:
        return low
    if progress >= end:
        return high
    return low + (progress - start) * (high - low) / (end - start)


def predict(model, images, config):
    beta = beta_at(int(config["beta_schedule"]["schedule_epochs"]), config)
    blind = blind_prediction(model, images, config)
    visible = model(images)
    return (blind + beta * visible) / (1. + beta)


def _masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


def loss(model, images, roi, config, epoch, generator):
    del generator
    blind = blind_prediction(model, images, config)
    difference = blind - images
    with torch.no_grad():
        visible_difference = model(images) - images
    beta = beta_at(epoch, config)
    regularizer = _masked_mean(difference.square(), roi)
    revisible = _masked_mean((difference + beta * visible_difference).square(), roi)
    total = float(config["lambda_blind"]) * regularizer + revisible
    return {"total": total, "reconstruction": revisible,
            "blind_regularizer": regularizer, "beta": beta}


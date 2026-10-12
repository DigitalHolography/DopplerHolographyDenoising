"""Neighbor2Neighbor pair construction and self-supervised objective."""
import torch
import torch.nn.functional as F

from .neighbor_arch import UNet


# All directed horizontal or vertical pairs in a 2 x 2 cell.
_PAIRS = torch.tensor(
    [[0, 1], [0, 2], [1, 3], [2, 3],
     [1, 0], [2, 0], [3, 1], [3, 2]], dtype=torch.long
)


def build_model(config):
    return UNet(in_nc=1, out_nc=1, n_feature=int(config["features"]),
                blindspot=False)


def predict(model, images, config):
    del config
    return model(images)


def pair_indices(batch, height, width, generator, device):
    if height % 2 or width % 2:
        raise ValueError("Neighbor2Neighbor requires even image dimensions")
    choices = torch.randint(0, len(_PAIRS), (batch, height // 2, width // 2),
                            generator=generator, device="cpu")
    pairs = _PAIRS[choices].to(device)
    return pairs[..., 0], pairs[..., 1]


def subsample(images, indices):
    """Select one of the four pixels in each non-overlapping 2 x 2 cell."""
    unpacked = F.pixel_unshuffle(images, 2)
    if images.shape[1] != 1:
        raise ValueError("The Doppler adapter expects one-channel images")
    return torch.gather(unpacked, 1, indices[:, None].long())


def _masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


def loss(model, images, roi, config, epoch, generator):
    first, second = pair_indices(*images.shape[0:1], *images.shape[-2:],
                                 generator, images.device)
    source = subsample(images, first)
    target = subsample(images, second)
    valid = subsample(roi, first) * subsample(roi, second)
    output = model(source)
    difference = output - target
    reconstruction = _masked_mean(difference.square(), valid)
    with torch.no_grad():
        full = model(images)
        expected_difference = subsample(full, first) - subsample(full, second)
    regularizer = _masked_mean((difference - expected_difference).square(), valid)
    weight = min(float(epoch) / float(config["schedule_epochs"]), 1.0) * float(config["increase_ratio"])
    total = reconstruction + weight * regularizer
    return {"total": total, "reconstruction": reconstruction,
            "regularizer": regularizer, "regularizer_weight": weight}


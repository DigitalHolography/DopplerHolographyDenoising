"""Validated training configuration shared by training and inference."""
from dataclasses import dataclass

SAMPLING_SCHEMA = "fractional_cycle_linear_v1"

@dataclass
class Config:
    seed: int = 2026
    history: int = 9
    base_channels: int = 32
    convlstm: bool = True
    block_size: int = 32
    blocks: int = 32
    objective: str = "article"  # Eq. 19; alternatives: l1, l2
    epochs: int = 250
    samples_per_epoch: int = 8000
    batch_size: int = 2
    validation_samples: int = 35  # Not specified by the article; explicit choice.
    learning_rate: float = 5e-5
    weight_decay: float = 0.01  # AdamW default in the supplied scripts.
    patience: int = 10
    validation_records: tuple = ()  # Empty => legacy fixed sequence split.
    input_mode: str = "patched"  # no_patch predicts another cycle's donor from untouched frames.
    split_mode: str = "mixed"  # mixed, temporal, or record
    brightness_correction: bool = True
    validation_fraction: float = .5

    def validate(self):
        if self.input_mode not in ("patched", "history_only", "no_patch"):
            raise ValueError("input_mode must be patched, no_patch or legacy history_only")
        if self.split_mode not in ("mixed", "temporal", "record"):
            raise ValueError("split_mode must be mixed, temporal or record")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.split_mode == "record" and not self.validation_records:
            raise ValueError("record split requires validation_records")
        for key in ("history", "base_channels", "block_size", "blocks", "epochs",
                    "samples_per_epoch", "batch_size", "validation_samples", "patience"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.base_channels % 8:
            raise ValueError("base_channels must be divisible by 8 (GroupNorm)")
        if self.objective not in ("article", "l1", "l2", "l1_grad_hessian", "l2_grad_hessian"):
            raise ValueError("objective must be article, l1, l2, l1_grad_hessian or l2_grad_hessian")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if self.objective in ("article", "l1_grad_hessian", "l2_grad_hessian") and self.input_mode != "no_patch" and self.block_size < 3:
            raise ValueError("Use l1/l2 for tiny-block ablations; Hessian needs 3 pixels")




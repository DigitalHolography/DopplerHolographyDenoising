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
    # Legacy checkpoint field. New benchmark plans use patch_mode; None maps
    # patched -> spatial_patches and no_patch -> none.
    input_mode: str = "patched"
    split_mode: str = "mixed"  # mixed, temporal, or record
    brightness_correction: bool = True
    validation_fraction: float = .5
    frame_pairing: str = "cycle_phase"  # random, next, or cycle_phase
    patch_mode: str | None = None  # none, spatial_patches, or vessel_patches

    def effective_patch_mode(self):
        """Translate old input_mode checkpoints into the explicit patch policy."""
        if self.patch_mode is not None:
            return self.patch_mode
        return "none" if self.input_mode == "no_patch" else "spatial_patches"

    def inference_prefix(self):
        """Frames copied before an aligned prediction can be produced."""
        return self.history + int(self.effective_patch_mode()=="none" and self.frame_pairing=="next")

    def validate(self):
        if self.input_mode not in ("patched", "history_only", "no_patch"):
            raise ValueError("input_mode must be patched, no_patch or legacy history_only")
        if self.patch_mode not in (None, "none", "spatial_patches", "vessel_patches"):
            raise ValueError("patch_mode must be none, spatial_patches or vessel_patches")
        if self.patch_mode is not None and self.input_mode != "patched":
            raise ValueError("Explicit patch_mode cannot be combined with legacy input_mode")
        if self.frame_pairing not in ("random", "next", "cycle_phase"):
            raise ValueError("frame_pairing must be random, next or cycle_phase")
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
        if (self.objective in ("article", "l1_grad_hessian", "l2_grad_hessian")
                and self.effective_patch_mode() != "none" and self.block_size < 3):
            raise ValueError("Use l1/l2 for tiny-block ablations; Hessian needs 3 pixels")




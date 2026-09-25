"""Prepared-record access and phase-matched training sample generation."""
from dataclasses import dataclass
import json
from pathlib import Path, PureWindowsPath

import numpy as np

from ..common import load_sibling, sha256
from ..preparation.core import cycle_coordinates

@dataclass(frozen=True)
class PhaseDonor:
    """A phase-matched donor, possibly between two measured frames."""
    lower: int
    upper: int
    upper_weight: float
    cycle: int


def interpolate_donor(values, donor):
    """Linearly sample a frame sequence or scalar trace at a donor coordinate."""
    lower = np.asarray(values[donor.lower])
    if donor.upper == donor.lower or donor.upper_weight == 0:
        return lower
    return lower * (1.0 - donor.upper_weight) + np.asarray(values[donor.upper]) * donor.upper_weight



class Record:
    """Memory-mapped prepared video plus its cardiac-cycle coordinates."""

    def __init__(self, directory):
        self.path = Path(directory).resolve()
        self.name = self.path.name
        self.frames = np.load(self.path / "frames.npy", mmap_mode="r", allow_pickle=False)
        self.roi = np.load(self.path / "roi.npy", allow_pickle=False).astype(bool)
        self.brightness = np.load(self.path / "brightness.npy", allow_pickle=False)
        self.phase = np.load(self.path / "phase.npy", allow_pickle=False)
        self.metadata = json.loads((self.path / "metadata.json").read_text(encoding="utf-8"))
        source_measure = Path(self.metadata["dataset_measure"]) if self.metadata.get("dataset_measure") else None
        if source_measure is not None and not source_measure.is_dir():
            original = self.metadata["dataset_measure"]
            dataset_name = (PureWindowsPath(original).parent.name if "\\" in original
                            else Path(original).parent.name)
            candidate = self.path.parent.parent.parent / dataset_name / self.name
            if candidate.is_dir():
                source_measure = candidate.resolve()
        self.dataset_measure = source_measure
        if self.frames.ndim != 3 or self.roi.shape != self.frames.shape[1:]:
            raise ValueError(f"Invalid frame/ROI dimensions: {self.path}")
        if self.frames.dtype != np.float32 or any(s % 32 for s in self.frames.shape[1:]):
            raise ValueError("Prepared frames must be float32 with H,W divisible by 32")
        if self.brightness.shape != (len(self.frames),) or self.phase.shape != (len(self.frames),):
            raise ValueError("Frame/brightness/phase lengths differ")
        dataset_record = self.metadata.get("schema") == "noise2time.dataset.v1"
        valid_path = self.path/"valid_frames.npy"
        has_valid_frames = valid_path.exists()
        if (not np.isfinite(self.frames).all() or self.frames.min() < 0 or self.frames.max() > 1
                or not np.isfinite(self.brightness).all() or np.any(self.brightness < 0)
                or not np.issubdtype(self.phase.dtype, np.integer)
                or np.any(self.phase < (-1 if dataset_record or has_valid_frames else 0))):
            raise ValueError("Invalid prepared intensities, brightness or phase")
        self.valid = (np.load(valid_path, allow_pickle=False).astype(bool)
                      if has_valid_frames else np.ones(len(self.frames),bool))
        if self.valid.shape != (len(self.frames),):
            raise ValueError("Invalid valid_frames dimensions")
        self.cycle, self.fractional_phase, self.cycles = cycle_coordinates(
            len(self.frames), self.metadata.get("peaks", []))
        for filename, computed in (("cycle.npy", self.cycle),
                                   ("fractional_phase.npy", self.fractional_phase)):
            path = self.path / filename
            if path.exists():
                stored = np.load(path, allow_pickle=False)
                if stored.shape != computed.shape or not np.allclose(stored, computed, equal_nan=True):
                    raise ValueError(f"{filename} disagrees with metadata peaks")
        # Keep the discrete table for old reports and audit files. Sampling
        # below uses normalized phase and does not assume equal cycle lengths.
        self.donors = {int(p): np.flatnonzero((self.phase == p) & self.valid & (self.phase >= 0) & (self.brightness > 1e-8))
                       for p in np.unique(self.phase)}
        self._training_vessel_mask = None
        self.vessel_mask_provenance = None

    def training_vessel_mask(self):
        """Load the retinal and pseudo-choroidal vessel union on first use."""
        if self._training_vessel_mask is not None:
            return self._training_vessel_mask
        if self.dataset_measure is None or not self.dataset_measure.is_dir():
            raise ValueError(f"{self.name}: vessel patches require a linked dataset measurement")
        workflow = load_sibling("dataset_workflow")
        from ..evaluation.metrics import load_evaluation_mask
        folder = self.dataset_measure
        paths = [workflow.manual_mask(folder,"artery"), workflow.manual_mask(folder,"vein")]
        paths += workflow.choroidal_masks(folder)[0]
        union = np.zeros(self.roi.shape,bool)
        for path in paths:
            union |= load_evaluation_mask(path,self.roi.shape)
        union &= self.roi
        if not union.any():
            raise ValueError(f"{self.name}: vessel-mask union is empty inside the diaphragm")
        self._training_vessel_mask = union
        self.vessel_mask_provenance = {
            "definition":"manual retinal artery | manual retinal vein | pseudo choroidal vessels, intersected with ROI",
            "sources":[{"path":str(Path(path).resolve()),"sha256":sha256(path)} for path in paths],
            "selected_pixels":int(union.sum()),
        }
        return union

    def matching_donors(self, t, bounds=None, exclude=None):
        """Find the same fractional phase in every other eligible cycle.

        Both endpoints of an interpolated coordinate must be valid, bright,
        inside ``bounds`` and outside ``exclude``. Thus an artifact or split
        boundary cannot enter through the second interpolation endpoint.
        """
        target_cycle = int(self.cycle[t])
        if target_cycle < 0:
            return []
        fraction = float(self.fractional_phase[t])
        lo_bound, hi_bound = bounds if bounds is not None else (0, len(self.frames))
        donor_valid = getattr(self, "donor_valid", self.valid)
        result = []
        for cycle_id, (start, stop) in enumerate(self.cycles):
            if cycle_id == target_cycle:
                continue
            coordinate = start + fraction * (stop - start)
            lower = int(np.floor(coordinate))
            weight = float(coordinate - lower)
            if weight < 1e-10:
                upper, weight = lower, 0.0
            else:
                upper = lower + 1
                # Close the donor cycle periodically. Using ``stop`` here
                # would borrow the first frame of the following cycle and, for
                # adjacent cycles, could reintroduce the target cycle.
                if upper == stop:
                    upper = start
            endpoints = (lower,) if upper == lower else (lower, upper)
            if any(i < lo_bound or i >= hi_bound for i in endpoints):
                continue
            if exclude is not None and any(exclude[0] <= i < exclude[1] for i in endpoints):
                continue
            if not all(donor_valid[i] and self.brightness[i] > 1e-8 for i in endpoints):
                continue
            result.append(PhaseDonor(lower, upper, weight, cycle_id))
        return result

    def eligible(self, history, cfg=None):
        """Return anchors with valid history and at least one requested pair."""
        if cfg is None:
            return [t for t in range(history, len(self.frames))
                    if self.valid[t] and self.cycle[t] >= 0 and self.brightness[t] > 1e-8
                    and self.matching_donors(t, getattr(self, "donor_bounds", None))]
        return [t for t in range(history, len(self.frames))
                if self.valid[t] and self.brightness[t] > 1e-8
                and _pairing_candidates(self,t,cfg,
                    (t-history,t+1) if cfg.patch_mode is not None or cfg.input_mode=="no_patch" else None)]


def _matching_donors(record, target, exclude):
    """Return phase donors, including support for older test fixtures."""
    if hasattr(record, "matching_donors"):
        return record.matching_donors(
            target, getattr(record, "donor_bounds", None), exclude,
        )

    # Prepared Record objects use fractional cycle coordinates above. This
    # branch preserves the former integer-phase API for external fixtures.
    indices = record.donors[int(record.phase[target])]
    if exclude is not None:
        indices = indices[(indices < exclude[0]) | (indices >= exclude[1])]
    return [PhaseDonor(int(i), int(i), 0.0, -1) for i in indices if i != target]


def _brightness_ratio(record, target, donor, enabled):
    """Scale donor intensity to the target frame's global brightness."""
    if not enabled:
        return 1.0
    donor_brightness = float(interpolate_donor(record.brightness, donor))
    return float(record.brightness[target] / donor_brightness)


def _pairing_candidates(record, target, cfg, exclude):
    """Return eligible paired coordinates for the requested frame policy."""
    if cfg.frame_pairing == "cycle_phase":
        return _matching_donors(record,target,exclude)
    if cfg.frame_pairing == "next":
        index = target + 1
        bounds = getattr(record,"target_bounds",(0,len(record.frames)))
        valid = getattr(record,"valid",np.ones(len(record.frames),bool))
        if (index >= len(record.frames) or not bounds[0] <= index < bounds[1]
                or not valid[index] or record.brightness[index] <= 1e-8):
            return []
        return [PhaseDonor(index,index,0.,int(record.cycle[index]) if hasattr(record,"cycle") else -1)]
    bounds = getattr(record,"donor_bounds",(0,len(record.frames)))
    donor_valid = getattr(record,"donor_valid",
                          getattr(record,"valid",np.ones(len(record.frames),bool)))
    candidates = [index for index in range(*bounds)
                  if donor_valid[index] and record.brightness[index] > 1e-8
                  and not (exclude and exclude[0] <= index < exclude[1])]
    return [PhaseDonor(index,index,0.,int(record.cycle[index]) if hasattr(record,"cycle") else -1)
            for index in candidates]


def _vessel_mask(record):
    """Support real Records and lightweight scientific test fixtures."""
    if hasattr(record,"training_vessel_mask"):
        return record.training_vessel_mask()
    mask = getattr(record,"vessel_mask",None)
    if mask is None:
        raise ValueError("vessel_patches require a vessel mask")
    mask = np.asarray(mask,dtype=bool) & np.asarray(record.roi,dtype=bool)
    if not mask.any():
        raise ValueError("vessel mask is empty inside the ROI")
    return mask



def replacement(record, t, cfg, rng, stage=None):
    """Build one self-supervised input, target and loss mask.

    All maintained modes receive history+1 frames ending at anchor ``t``.
    With no patches, the paired frame is the full target. With patches, it
    supplies replacement content and the original anchor remains the target.
    """
    if cfg.split_mode == "temporal":
        if stage not in ("train", "valid"):
            raise ValueError("Temporal replacement requires an explicit train/valid stage")
        record = record.stage_views[stage]
    sequence = np.array(record.frames[t-cfg.history:t+1], copy=True)
    target = sequence[-1].copy()
    mask = np.zeros(record.roi.shape, bool)
    occupied = np.zeros_like(mask)
    if cfg.split_mode == "temporal":
        if stage not in ("train", "valid"):
            raise ValueError("Temporal replacement requires an explicit train/valid stage")
        target_lo, target_hi = record.split_ranges[stage]
        if not target_lo <= t-cfg.history <= t < target_hi:
            raise ValueError("Target history crosses temporal partition")
    patch_mode = cfg.effective_patch_mode()
    # New explicit combinations never pair against a frame visible in the
    # input. Legacy patched checkpoints retain their former donor pool.
    exclude = ((t-cfg.history,t+1)
               if cfg.patch_mode is not None or cfg.input_mode=="no_patch" else None)
    donors = _pairing_candidates(record,t,cfg,exclude)
    if not donors:
        if cfg.frame_pairing=="cycle_phase" and patch_mode=="none":
            raise ValueError("No same-phase donor outside the input window")
        raise ValueError(f"No eligible {cfg.frame_pairing} frame for anchor")
    if patch_mode == "none":
        donor = donors[int(rng.integers(len(donors)))]
        ratio = _brightness_ratio(record, t, donor, cfg.brightness_correction)
        target = np.clip(interpolate_donor(record.frames, donor) * ratio, 0, 1)
        return sequence, target[None], record.roi[None].astype(np.float32)
    h, w = mask.shape
    size = cfg.block_size
    if size > min(h, w):
        raise ValueError("Block exceeds frame size")
    vessel = _vessel_mask(record) if patch_mode == "vessel_patches" else None
    vessel_coordinates = np.argwhere(vessel) if vessel is not None else None
    accepted = 0
    for _ in range(cfg.blocks * 100):
        if vessel_coordinates is None:
            y, x = int(rng.integers(h-size+1)), int(rng.integers(w-size+1))
        else:
            center_y,center_x = vessel_coordinates[int(rng.integers(len(vessel_coordinates)))]
            y = int(np.clip(center_y-size//2,0,h-size))
            x = int(np.clip(center_x-size//2,0,w-size))
        region = np.s_[y:y+size, x:x+size]
        if (not record.roi[region].all() or occupied[region].any()
                or (vessel is not None and not vessel[region].any())):
            continue
        donor = donors[int(rng.integers(len(donors)))]
        ratio = _brightness_ratio(record, t, donor, cfg.brightness_correction)
        sequence[-1][region] = np.clip(interpolate_donor(record.frames, donor)[region] * ratio, 0, 1)
        mask[region] = occupied[region] = True
        accepted += 1
        if accepted == cfg.blocks:
            break
    if accepted != cfg.blocks:
        region_name = "vessel patches" if vessel is not None else "patches"
        raise ValueError(f"Placed {accepted}/{cfg.blocks} {region_name}; reduce blocks/size or check masks/ROI")
    # Keep the same loss locations and sampling as the baseline. The target frame
    # (including its replaced patches) is entirely absent in history-only mode.
    if cfg.input_mode == "history_only":
        sequence = sequence[:-1]
    return sequence, target[None], mask[None].astype(np.float32)



def split_samples(records, cfg):
    if cfg.split_mode == "temporal":
        if cfg.validation_records:
            raise ValueError("Temporal and video-disjoint validation cannot be combined")
        from .. import noise2time as api
        return load_sibling("benchmark_splits").temporal_split(records, cfg, api)
    rng = np.random.default_rng(cfg.seed)
    pools = [[(i,t) for t in (rec.eligible(cfg.history,cfg) if isinstance(rec,Record)
                              else rec.eligible(cfg.history))]
             for i,rec in enumerate(records)]
    if any(not pool for pool in pools):
        raise ValueError(f"Each record needs an eligible anchor and {cfg.frame_pairing} pairing")
    if cfg.validation_records:
        unknown = set(cfg.validation_records) - {r.name for r in records}
        if unknown:
            raise ValueError(f"Unknown validation records: {unknown}")
        train = [s for i,pool in enumerate(pools) if records[i].name not in cfg.validation_records for s in pool]
        candidates = [s for i,pool in enumerate(pools) if records[i].name in cfg.validation_records for s in pool]
    else:
        # Reserve one sample per recording for training, as in the existing scripts.
        reserved = [int(rng.integers(len(pool))) for pool in pools]
        train = [pool[j] for pool,j in zip(pools,reserved)]
        candidates = [s for pool,j in zip(pools,reserved) for k,s in enumerate(pool) if k != j]
    if not train or len(candidates) < cfg.validation_samples:
        raise ValueError("Not enough independent pools/samples for the requested split")
    selected = set(rng.choice(len(candidates), cfg.validation_samples, replace=False).tolist())
    valid = [s for i,s in enumerate(candidates) if i in selected]
    if not cfg.validation_records:
        train += [s for i,s in enumerate(candidates) if i not in selected]
    return train, valid


def resolve_training_records(paths):
    """Expand a prepared-record parent folder into its direct child records."""
    resolved = []
    for path in map(Path, paths):
        if (path / "frames.npy").is_file():
            resolved.append(path)
        elif path.is_dir():
            children = sorted(
                (child for child in path.iterdir()
                 if child.is_dir() and (child / "frames.npy").is_file()),
                key=lambda child: child.name.casefold(),
            )
            if not children:
                raise ValueError(f"No prepared records found in {path}")
            resolved.extend(children)
        else:
            raise ValueError(f"Not a prepared record or record folder: {path}")
    paths_by_identity = [path.resolve() for path in resolved]
    if len(set(paths_by_identity)) != len(paths_by_identity):
        raise ValueError("The same prepared record was supplied more than once")
    return resolved


def select_train_samples_for_epoch(training, samples_per_epoch, rng):
    """Choose at least one target per record, then fill from the pooled split."""
    by_record = {}
    for index, (record_index, _) in enumerate(training):
        by_record.setdefault(record_index, []).append(index)
    if len(by_record) > samples_per_epoch:
        raise ValueError(
            f"{len(by_record)} training records exceed the "
            f"{samples_per_epoch} samples-per-epoch budget"
        )
    mandatory = [int(rng.choice(indices)) for _, indices in sorted(by_record.items())]
    mandatory_set = set(mandatory)
    remaining_pool = [i for i in range(len(training)) if i not in mandatory_set]
    remaining_count = samples_per_epoch - len(mandatory)
    if remaining_count:
        if not remaining_pool:
            remaining_pool = list(range(len(training)))
        extra = rng.choice(
            remaining_pool, size=remaining_count,
            replace=remaining_count > len(remaining_pool),
        ).tolist()
    else:
        extra = []
    choices = np.asarray(mandatory + extra, dtype=np.int64)
    rng.shuffle(choices)
    return choices




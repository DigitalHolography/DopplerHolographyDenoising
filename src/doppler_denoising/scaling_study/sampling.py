"""Balanced sampling utilities for the scaling study."""
from __future__ import annotations

import numpy as np


class PatientBalancedSampler:
    """Sample patient -> video -> anchor, giving every patient equal weight.

    A pooled list silently weights long recordings more heavily.  This sampler
    explicitly implements the experimental unit used by the study: a patient.
    """

    def __init__(self, pools, patients, rng):
        if len(pools) != len(patients) or not pools or any(not pool for pool in pools):
            raise ValueError("Every training record needs a patient and at least one eligible anchor")
        self.pools, self.rng = pools, rng
        self.by_patient = {}
        for record_index, patient in enumerate(patients):
            self.by_patient.setdefault(patient, []).append(record_index)
        self.patient_names = sorted(self.by_patient)

    def draw(self, count):
        samples = []
        for _ in range(count):
            patient = self.patient_names[int(self.rng.integers(len(self.patient_names)))]
            records = self.by_patient[patient]
            record_index = records[int(self.rng.integers(len(records)))]
            pool = self.pools[record_index]
            samples.append((record_index, int(pool[int(self.rng.integers(len(pool)))])))
        return samples


def eligible_pools(records, cfg):
    pools = [record.eligible(cfg.history, cfg) for record in records]
    if any(not pool for pool in pools):
        missing = [records[i].name for i, pool in enumerate(pools) if not pool]
        raise ValueError(f"No eligible training anchors: {missing}")
    return pools


def deterministic_validation(records, cfg, anchors_per_record):
    """Choose an equal, deterministic number of anchors in every video."""
    result = []
    for record_index, pool in enumerate(eligible_pools(records, cfg)):
        if len(pool) < anchors_per_record:
            raise ValueError(f"{records[record_index].name}: only {len(pool)} validation anchors; "
                             f"{anchors_per_record} requested")
        positions = np.linspace(0, len(pool) - 1, anchors_per_record, dtype=int)
        result.extend((record_index, int(pool[position])) for position in positions)
    return result

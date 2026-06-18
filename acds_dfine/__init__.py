"""ACDS migration helpers for D-FINE style detectors."""

from .core import (
    DFineACDSCriterion,
    DFineACDSConfig,
    ScaleAwareQueryBias,
    apply_rsnds_to_sampling_offsets,
    build_gt_instances_for_acq,
    rsnds_gamma_for_dfine,
    small_object_hungarian_indices,
)
from .hooks import add_acds_losses, build_acds_criterion

__all__ = [
    "DFineACDSCriterion",
    "DFineACDSConfig",
    "ScaleAwareQueryBias",
    "add_acds_losses",
    "apply_rsnds_to_sampling_offsets",
    "build_acds_criterion",
    "build_gt_instances_for_acq",
    "rsnds_gamma_for_dfine",
    "small_object_hungarian_indices",
]

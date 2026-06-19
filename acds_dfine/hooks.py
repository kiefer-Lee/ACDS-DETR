"""Small integration hooks for patched D-FINE training code."""

from __future__ import annotations

from typing import Any, MutableMapping, Sequence

import torch

from .core import DFineACDSCriterion, DFineACDSConfig


def build_acds_criterion(config: DFineACDSConfig | dict[str, Any] | None = None) -> DFineACDSCriterion:
    """Build the auxiliary ACDS criterion used by a patched D-FINE criterion."""

    return DFineACDSCriterion(config)


def add_acds_losses(
    loss_dict: MutableMapping[str, torch.Tensor],
    outputs: dict[str, Any],
    targets: Sequence[Any],
    criterion: DFineACDSCriterion,
    *,
    indices: Sequence[tuple[torch.Tensor, torch.Tensor]] | Sequence[Sequence[tuple[torch.Tensor, torch.Tensor]]] | None = None,
    prefix: str = "",
    include_metrics: bool = False,
) -> MutableMapping[str, torch.Tensor]:
    """Add ACDS losses to D-FINE's existing loss dictionary in-place."""

    acds_losses = criterion(outputs, targets, indices=indices)
    for name, value in acds_losses.items():
        if not include_metrics and not name.startswith("loss_"):
            continue
        loss_dict[f"{prefix}{name}"] = value
    return loss_dict

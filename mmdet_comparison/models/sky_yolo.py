"""Sky-YOLO modules for MMYOLO/MMDetection experiments.

This implementation follows the public Sky-YOLO paper description: YOLOv8n
with the second backbone convolution replaced by MSFConv, a lightweight BiFPN
neck over P2/P3/P4, and Wise-IoU v3 as the box regression loss.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - exercised in the target MMYOLO environment.
    from mmcv.cnn import ConvModule
    from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
    from mmengine.model import BaseModule
    from mmyolo.models.layers import CSPLayerWithTwoConv
    from mmyolo.models.utils import make_divisible, make_round
    from mmyolo.registry import MODELS
except Exception:  # pragma: no cover - lets local syntax/unit tests run.
    from .compat import MODELS

    ConvModule = None
    CSPLayerWithTwoConv = None
    ConfigType = OptConfigType = OptMultiConfig = dict

    class BaseModule(nn.Module):
        def __init__(self, init_cfg=None):
            super().__init__()


def _require_mmyolo() -> None:
    if ConvModule is None or CSPLayerWithTwoConv is None:
        raise ImportError(
            "Sky-YOLO modules require mmcv, mmengine, mmdet and mmyolo. "
            "Activate the MMDetection/MMYOLO environment before building the "
            "model."
        )


class MSFConv(BaseModule):
    """Multi-scale feature fusion convolution based on partial convolution.

    The active channel subset is processed by 3x3 and 5x5 branches, while a
    small passthrough projection preserves part of the remaining channels.
    Branch outputs are concatenated, normalized, and activated as described in
    Sky-YOLO.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        partial_ratio: int = 2,
        skip_ratio: int = 4,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
        init_cfg: OptMultiConfig = None,
    ) -> None:
        _require_mmyolo()
        super().__init__(init_cfg=init_cfg)
        if partial_ratio < 1:
            raise ValueError("partial_ratio must be positive.")

        self.in_channels = in_channels
        self.part_channels = max(1, in_channels // partial_ratio)
        remaining_channels = in_channels - self.part_channels
        self.skip_channels = min(out_channels // skip_ratio, remaining_channels)
        branch_channels = out_channels - self.skip_channels
        branch3_channels = branch_channels // 2
        branch5_channels = branch_channels - branch3_channels

        self.branch3 = nn.Conv2d(
            self.part_channels,
            branch3_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.branch5 = nn.Conv2d(
            self.part_channels,
            branch5_channels,
            kernel_size=5,
            stride=stride,
            padding=2,
            bias=False,
        )

        if self.skip_channels > 0:
            skip_layers = []
            if stride > 1:
                skip_layers.append(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
                )
            skip_layers.append(
                nn.Conv2d(
                    remaining_channels,
                    self.skip_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                )
            )
            self.skip = nn.Sequential(*skip_layers)
        else:
            self.skip = None

        self.post = ConvModule(
            out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_part = x[:, : self.part_channels]
        outs = [self.branch3(x_part), self.branch5(x_part)]
        if self.skip is not None:
            x_rest = x[:, self.part_channels :]
            outs.append(self.skip(x_rest))
        return self.post(torch.cat(outs, dim=1))


@MODELS.register_module()
class SkyYOLOBackbone(BaseModule):
    """Truncated YOLOv8 CSPDarknet backbone used by Sky-YOLO.

    It keeps stages up to P4 so the detector predicts on strides 4/8/16 instead
    of the original YOLOv8 strides 8/16/32.
    """

    arch_settings = {
        "P4": [[64, 128, 3, True], [128, 256, 6, True], [256, 512, 6, True]]
    }

    def __init__(
        self,
        arch: str = "P4",
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        input_channels: int = 3,
        out_indices: Sequence[int] = (1, 2, 3),
        frozen_stages: int = -1,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
        norm_eval: bool = False,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        _require_mmyolo()
        super().__init__(init_cfg=init_cfg)
        if arch not in self.arch_settings:
            raise KeyError(f"Unsupported SkyYOLOBackbone arch: {arch}")
        if not set(out_indices).issubset(range(len(self.arch_settings[arch]) + 1)):
            raise ValueError("out_indices must reference stem/stage indices.")

        self.arch_setting = self.arch_settings[arch]
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.input_channels = input_channels
        self.out_indices = tuple(out_indices)
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.stem = ConvModule(
            input_channels,
            make_divisible(self.arch_setting[0][0], widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.layers = ["stem"]
        for idx, setting in enumerate(self.arch_setting):
            self.add_module(f"stage{idx + 1}", nn.Sequential(*self._make_stage(idx, setting)))
            self.layers.append(f"stage{idx + 1}")

    def _make_stage(self, stage_idx: int, setting: list) -> List[nn.Module]:
        in_channels, out_channels, num_blocks, add_identity = setting
        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)

        if stage_idx == 0:
            downsample = MSFConv(
                in_channels,
                out_channels,
                stride=2,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        else:
            downsample = ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )

        csp = CSPLayerWithTwoConv(
            out_channels,
            out_channels,
            num_blocks=num_blocks,
            add_identity=add_identity,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        return [downsample, csp]

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                module = getattr(self, self.layers[i])
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outs = []
        for idx, layer_name in enumerate(self.layers):
            x = getattr(self, layer_name)(x)
            if idx in self.out_indices:
                outs.append(x)
        return tuple(outs)


class FastWeightedFusion(BaseModule):
    """EfficientDet-style positive weighted feature fusion."""

    def __init__(self, num_inputs: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.eps = eps

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        out = inputs[0] * weights[0]
        for idx in range(1, len(inputs)):
            out = out + inputs[idx] * weights[idx]
        return out


@MODELS.register_module()
class LightBiFPN(BaseModule):
    """Lightweight BiFPN neck for P2/P3/P4 Sky-YOLO features."""

    def __init__(
        self,
        in_channels: List[int],
        out_channels: Union[List[int], Tuple[int, int, int]],
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        num_csp_blocks: int = 3,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
        init_cfg: OptMultiConfig = None,
    ) -> None:
        _require_mmyolo()
        super().__init__(init_cfg=init_cfg)
        if len(in_channels) != 3:
            raise ValueError("LightBiFPN expects three feature levels: P2/P3/P4.")
        if isinstance(out_channels, int):
            out_channels = [out_channels, out_channels * 2, out_channels * 4]
        if len(out_channels) != 3:
            raise ValueError("out_channels must contain three channel values.")

        self.in_channels = [make_divisible(c, widen_factor) for c in in_channels]
        self.out_channels = [make_divisible(c, widen_factor) for c in out_channels]
        blocks = make_round(num_csp_blocks, deepen_factor)

        c2, c3, c4 = self.out_channels
        self.lateral_convs = nn.ModuleList(
            [
                ConvModule(i, o, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
                for i, o in zip(self.in_channels, self.out_channels)
            ]
        )
        self.p4_to_p3 = ConvModule(c4, c3, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.p3_to_p2 = ConvModule(c3, c2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.p2_to_p3 = ConvModule(
            c2, c3, 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.p3_to_p4 = ConvModule(
            c3, c4, 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg
        )

        self.fuse_p3_td = FastWeightedFusion(2)
        self.fuse_p2 = FastWeightedFusion(2)
        self.fuse_p3_out = FastWeightedFusion(3)
        self.fuse_p4_out = FastWeightedFusion(2)

        self.p3_td_csp = CSPLayerWithTwoConv(
            c3, c3, num_blocks=blocks, add_identity=False, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.p2_out_csp = CSPLayerWithTwoConv(
            c2, c2, num_blocks=blocks, add_identity=False, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.p3_out_csp = CSPLayerWithTwoConv(
            c3, c3, num_blocks=blocks, add_identity=False, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.p4_out_csp = CSPLayerWithTwoConv(
            c4, c4, num_blocks=blocks, add_identity=False, norm_cfg=norm_cfg, act_cfg=act_cfg
        )

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        if len(inputs) != 3:
            raise ValueError("LightBiFPN expects three input tensors.")
        p2, p3, p4 = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]

        p3_td = self.fuse_p3_td(
            [p3, F.interpolate(self.p4_to_p3(p4), size=p3.shape[-2:], mode="nearest")]
        )
        p3_td = self.p3_td_csp(p3_td)

        p2_out = self.fuse_p2(
            [p2, F.interpolate(self.p3_to_p2(p3_td), size=p2.shape[-2:], mode="nearest")]
        )
        p2_out = self.p2_out_csp(p2_out)

        p3_out = self.fuse_p3_out([p3, p3_td, self.p2_to_p3(p2_out)])
        p3_out = self.p3_out_csp(p3_out)

        p4_out = self.fuse_p4_out([p4, self.p3_to_p4(p3_out)])
        p4_out = self.p4_out_csp(p4_out)
        return p2_out, p3_out, p4_out


@MODELS.register_module()
class WiseIoULoss(nn.Module):
    """Wise-IoU v1/v3 loss for xyxy boxes.

    Defaults follow common WIoU v3 settings: alpha=1.7 and delta=2.7.
    """

    def __init__(
        self,
        version: str = "v3",
        bbox_format: str = "xyxy",
        reduction: str = "sum",
        loss_weight: float = 1.0,
        alpha: float = 1.7,
        delta: float = 2.7,
        momentum: float = 0.01,
        eps: float = 1e-7,
        return_iou: bool = False,
    ) -> None:
        super().__init__()
        if version not in {"v1", "v3"}:
            raise ValueError("WiseIoULoss only supports version='v1' or 'v3'.")
        if bbox_format != "xyxy":
            raise ValueError("WiseIoULoss currently expects bbox_format='xyxy'.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean' or 'sum'.")
        self.version = version
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.delta = delta
        self.momentum = momentum
        self.eps = eps
        self.return_iou = return_iou
        self.register_buffer("iou_mean", torch.tensor(1.0))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[float] = None,
        reduction_override: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        reduction = reduction_override or self.reduction
        iou = self._bbox_iou(pred, target)
        iou_loss = 1.0 - iou

        center_pred = (pred[:, :2] + pred[:, 2:]) * 0.5
        center_target = (target[:, :2] + target[:, 2:]) * 0.5
        center_dist = (center_pred - center_target).pow(2).sum(dim=-1)
        enclose_lt = torch.minimum(pred[:, :2], target[:, :2])
        enclose_rb = torch.maximum(pred[:, 2:], target[:, 2:])
        enclose_wh = (enclose_rb - enclose_lt).clamp(min=self.eps)
        enclose_diag = enclose_wh.pow(2).sum(dim=-1).detach().clamp(min=self.eps)
        loss = torch.exp(center_dist / enclose_diag) * iou_loss

        if self.version == "v3":
            if self.training and iou_loss.numel() > 0:
                batch_mean = iou_loss.detach().mean().clamp(min=self.eps)
                self.iou_mean.mul_(1.0 - self.momentum).add_(self.momentum * batch_mean)
            beta = iou_loss.detach() / self.iou_mean.clamp(min=self.eps)
            focus = beta / (self.delta * torch.pow(self.alpha, beta - self.delta))
            loss = focus * loss

        if weight is not None:
            while weight.dim() > loss.dim():
                weight = weight.squeeze(-1)
            loss = loss * weight

        if reduction == "mean":
            if avg_factor is not None:
                loss = loss.sum() / max(float(avg_factor), self.eps)
            else:
                loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        loss = loss * self.loss_weight
        if self.return_iou:
            return loss, iou
        return loss

    def _bbox_iou(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        lt = torch.maximum(pred[:, :2], target[:, :2])
        rb = torch.minimum(pred[:, 2:], target[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]

        pred_wh = (pred[:, 2:] - pred[:, :2]).clamp(min=0)
        target_wh = (target[:, 2:] - target[:, :2]).clamp(min=0)
        union = (
            pred_wh[:, 0] * pred_wh[:, 1]
            + target_wh[:, 0] * target_wh[:, 1]
            - inter
        ).clamp(min=self.eps)
        return inter / union

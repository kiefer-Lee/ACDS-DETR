"""DN-DETR detector built as a DAB-DETR comparison baseline."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .compat import MODELS
from .dn_query_generator import DNQueryGenerator

try:  # pragma: no cover - exercised in the target MMDetection environment.
    from mmdet.models.detectors import DABDETR as _MMDetDABDETR
except Exception:  # pragma: no cover
    _MMDetDABDETR = nn.Module


@MODELS.register_module()
class DNDETR(_MMDetDABDETR):
    """DAB-DETR with denoising queries prepended during training."""

    def __init__(self, *args: Any, dn_cfg: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if _MMDetDABDETR is nn.Module:
            raise ImportError("mmdet is required to instantiate DNDETR")
        super().__init__(*args, **kwargs)
        cfg = dict(dn_cfg or {})
        cfg.setdefault("num_groups", 5)
        cfg.setdefault("label_noise_scale", 0.2)
        cfg.setdefault("box_noise_scale", 0.4)
        num_classes = int(getattr(self.bbox_head, "num_classes", cfg.pop("num_classes", 80)))
        embed_dims = int(getattr(self, "embed_dims", cfg.pop("embed_dims", 256)))
        self.dn_query_generator = DNQueryGenerator(num_classes=num_classes, embed_dims=embed_dims, **cfg)

    def forward_transformer(self, img_feats: tuple[torch.Tensor], batch_data_samples: list[Any] | None = None) -> dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(img_feats, batch_data_samples)
        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        try:
            pre_decoder_inputs_dict, head_inputs_dict = self.pre_decoder(
                **encoder_outputs_dict, batch_data_samples=batch_data_samples
            )
        except TypeError:
            pre_decoder_inputs_dict, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)

        decoder_inputs_dict.update(pre_decoder_inputs_dict)
        decoder_inputs_dict.update(encoder_outputs_dict)
        if self.training and batch_data_samples is not None:
            self._prepend_dn_queries(decoder_inputs_dict, head_inputs_dict, batch_data_samples)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def forward_decoder(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None,
        memory_pos: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict:
        decoder_kwargs = dict(
            query=query,
            key=memory,
            query_pos=query_pos,
            key_pos=memory_pos,
            key_padding_mask=memory_mask,
            reg_branches=getattr(self.bbox_head, "fc_reg", None),
        )
        if self_attn_mask is not None:
            decoder_kwargs["self_attn_masks"] = self_attn_mask
        try:
            hidden_states, references = self.decoder(**decoder_kwargs)
        except TypeError:
            if self_attn_mask is None:
                raise
            decoder_kwargs.pop("self_attn_masks", None)
            decoder_kwargs["self_attn_mask"] = self_attn_mask
            hidden_states, references = self.decoder(**decoder_kwargs)
        return dict(hidden_states=hidden_states, references=references)

    def _prepend_dn_queries(
        self,
        decoder_inputs_dict: dict[str, Any],
        head_inputs_dict: dict[str, Any],
        batch_data_samples: list[Any],
    ) -> None:
        query = decoder_inputs_dict.get("query")
        if query is None:
            raise RuntimeError("DNDETR expects DAB-DETR decoder inputs to contain `query`.")

        num_matching_queries = int(query.shape[1])
        dn_output = self.dn_query_generator(batch_data_samples, num_matching_queries, device=query.device)
        if dn_output.meta["num_denoising_queries"] <= 0:
            decoder_inputs_dict["query"] = query + self.dn_query_generator.label_embedding.weight.sum() * 0.0
            head_inputs_dict["dn_meta"] = dn_output.meta
            return

        decoder_inputs_dict["query"] = torch.cat((dn_output.label_query, query), dim=1)
        reference_key = self._find_reference_key(decoder_inputs_dict)
        if reference_key is None:
            raise RuntimeError("DNDETR expects DAB-DETR decoder inputs to contain reference/query position boxes.")
        decoder_inputs_dict[reference_key] = torch.cat((dn_output.bbox_query, decoder_inputs_dict[reference_key]), dim=1)
        decoder_inputs_dict["self_attn_mask"] = dn_output.self_attn_mask
        head_inputs_dict["dn_meta"] = dn_output.meta

    @staticmethod
    def _find_reference_key(decoder_inputs_dict: dict[str, Any]) -> str | None:
        for key in ("query_pos", "reference_points", "references"):
            value = decoder_inputs_dict.get(key)
            if isinstance(value, torch.Tensor) and value.dim() == 3 and value.shape[-1] == 4:
                return key
        return None

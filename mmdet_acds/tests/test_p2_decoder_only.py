import torch

from mmdet_acds.models.acds_detr import replace_encoded_levels, resolve_encoder_level_indices


def test_p2_decoder_only_replaces_selected_levels():
    spatial_shapes = torch.tensor([[2, 2], [1, 2], [1, 1]])
    original = torch.arange(7 * 2, dtype=torch.float32).view(1, 7, 2)
    encoded = torch.full((1, 3, 2), -1.0)
    merged = replace_encoded_levels(original, encoded, spatial_shapes, [1, 2])
    assert torch.allclose(merged[:, :4], original[:, :4])
    assert torch.allclose(merged[:, 4:], encoded)


def test_resolve_encoder_level_indices_fallback():
    assert resolve_encoder_level_indices(3, [1, 2]) == [1, 2]
    assert resolve_encoder_level_indices(3, [9]) == [0, 1, 2]


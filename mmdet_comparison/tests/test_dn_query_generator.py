from types import SimpleNamespace

import torch

from mmdet_comparison.models.dn_query_generator import DNQueryGenerator


def make_sample(labels, boxes, img_shape=(100, 200)):
    gt_instances = SimpleNamespace(
        labels=torch.tensor(labels, dtype=torch.long),
        bboxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
    )
    return SimpleNamespace(gt_instances=gt_instances, metainfo={"img_shape": img_shape})


def test_dn_query_generator_handles_empty_gt():
    generator = DNQueryGenerator(num_classes=10, embed_dims=32, num_groups=5)
    output = generator([make_sample([], [])], num_matching_queries=300)

    assert output.label_query.shape == (1, 0, 32)
    assert output.bbox_query.shape == (1, 0, 4)
    assert output.self_attn_mask.shape == (300, 300)
    assert output.meta["num_denoising_queries"] == 0


def test_dn_query_generator_shapes_and_mask():
    generator = DNQueryGenerator(
        num_classes=10,
        embed_dims=32,
        num_groups=2,
        label_noise_scale=0.0,
        box_noise_scale=0.0,
    )
    sample = make_sample([1, 3], [[10, 20, 30, 50], [50, 10, 80, 40]])
    output = generator([sample], num_matching_queries=5)

    assert output.label_query.shape == (1, 8, 32)
    assert output.bbox_query.shape == (1, 8, 4)
    assert output.self_attn_mask.shape == (13, 13)
    assert output.meta["num_denoising_queries"] == 8
    assert output.meta["num_denoising_groups"] == 2
    assert output.meta["target_weights"].sum().item() == 8
    assert bool(output.self_attn_mask[8, 0])
    assert bool(output.self_attn_mask[0, 4])
    assert not bool(output.self_attn_mask[0, 1])

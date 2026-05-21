import torch

from mmdet_acds.models.rsnds_msda import ReliabilityGuidedScaleSampler


def test_rsnds_disabled_gamma_base():
    sampler = ReliabilityGuidedScaleSampler(enabled=False, gamma_base=1.0)
    query = torch.zeros(2, 3, 4)
    gamma, rho = sampler(query, pred_boxes=torch.rand(2, 3, 4), pred_logits=torch.rand(2, 3, 5))
    assert torch.allclose(gamma, torch.ones(2, 3, 1))
    assert torch.allclose(rho, torch.zeros(2, 3, 1))


def test_rsnds_cls_conf_formula():
    sampler = ReliabilityGuidedScaleSampler(
        enabled=True, beta=1.0, gamma_base=1.0, gamma_min=0.35, gamma_max=1.25, reliability="cls_conf"
    )
    query = torch.zeros(1, 1, 4)
    pred_boxes = torch.tensor([[[0.5, 0.5, 0.04, 0.09]]])
    pred_logits = torch.tensor([[[4.0, 0.0, -1.0]]])
    gamma, rho = sampler(query, pred_boxes, pred_logits)
    expected_rho = pred_logits.softmax(-1)[..., :-1].max(-1, keepdim=True).values
    gamma_scale = torch.sqrt(torch.tensor([[[0.04 * 0.09]]])).clamp(0.35, 1.25)
    expected_gamma = (1 - expected_rho) * 1.0 + expected_rho * gamma_scale
    assert torch.allclose(rho, expected_rho)
    assert torch.allclose(gamma, expected_gamma)


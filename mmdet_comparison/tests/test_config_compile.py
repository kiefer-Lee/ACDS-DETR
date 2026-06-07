from pathlib import Path


def test_comparison_configs_and_models_compile():
    root = Path(__file__).resolve().parents[1]
    for folder in ("configs", "models"):
        for path in (root / folder).glob("*.py"):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_yolov8_config_overrides_head_and_assigner_classes():
    path = Path(__file__).resolve().parents[1] / "configs" / "yolov8_s_visdrone.py"
    namespace = {}
    exec(path.read_text(encoding="utf-8"), namespace)
    model = namespace["model"]

    assert model["bbox_head"]["head_module"]["num_classes"] == 10
    assert model["train_cfg"]["assigner"]["num_classes"] == 10


def test_sky_yolo_config_uses_p2_p3_p4_and_wise_iou():
    path = Path(__file__).resolve().parents[1] / "configs" / "sky_yolo_n_visdrone.py"
    namespace = {}
    exec(path.read_text(encoding="utf-8"), namespace)
    model = namespace["model"]

    assert model["backbone"]["type"] == "SkyYOLOBackbone"
    assert model["backbone"]["out_indices"] == (1, 2, 3)
    assert model["neck"]["type"] == "LightBiFPN"
    assert model["bbox_head"]["head_module"]["featmap_strides"] == [4, 8, 16]
    assert model["bbox_head"]["prior_generator"]["strides"] == [4, 8, 16]
    assert model["bbox_head"]["loss_bbox"]["type"] == "WiseIoULoss"

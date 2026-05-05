from .collate import collate_fn
from .visdrone import VisDroneDetection


def build_dataset(split, cfg):
    name = cfg["dataset"]["name"].lower()
    if name != "visdrone":
        raise ValueError(f"Unsupported dataset: {name}")
    ds_split = cfg["dataset"]["train_split"] if split == "train" else cfg["dataset"]["val_split"]
    return VisDroneDetection(
        root=cfg["dataset"]["root"],
        split=ds_split,
        img_size=cfg["dataset"]["img_size"],
        max_size=cfg["dataset"]["max_size"],
        train=split == "train",
        min_area=cfg["dataset"].get("min_area", 1),
        max_samples=cfg["dataset"].get("max_samples"),
        augment=cfg["dataset"].get("augment", {}),
    )

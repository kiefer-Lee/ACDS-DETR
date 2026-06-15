_base_ = ["./dab_detr_r50_visdrone.py"]

num_classes = 3
data_root = "/root/blockdata/Datasets/UAVDT/"
data_root = data_root if data_root.endswith(("/", "\\")) else data_root + "/"
train_ann_file = "annotations/train.json"
val_ann_file = "annotations/val.json"
uavdt_classes = ("car", "bus", "truck")
metainfo = dict(classes=uavdt_classes)

model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=""),
        metainfo=metainfo,
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=""),
        metainfo=metainfo,
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + val_ann_file)
test_evaluator = val_evaluator

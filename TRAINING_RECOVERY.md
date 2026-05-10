# 新配置训练策略

当前 `configs/` 已重建为无 `_base_` 的完整配置集，每个 YAML 都能独立运行。

## 推荐最高性能流程

### 1. 定位启动阶段

先让模型学会基础同类定位，目标是把 `topk_recall_same_class_iou_0.50` 从 1%-2% 拉到 5%-10% 以上。

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py \
  --config configs/stage1_localization_bootstrap.yaml
```

### 2. 完整小目标微调

从 stage1 最佳权重恢复，开启 ACQ、R-SNDS、小目标 matching/loss、P2、高 query。

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py \
  --config configs/stage2_acds_full_finetune.yaml \
  --resume outputs/stage1_localization_bootstrap/best_map.pth
```

### 3. 单阶段性能配置

如果只想跑一个配置，使用 `default.yaml`。它按单卡 4090 24GB 设计，双卡时每卡 batch size 仍为 1。

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py \
  --config configs/default.yaml
```

## 诊断命令

```bash
python tools/debug_eval_sanity.py \
  --checkpoint outputs/stage1_localization_bootstrap/best_map.pth \
  --gpu 0 \
  --use-ema \
  --batches 8
```

重点看：

```text
topk_recall_same_class_iou_0.50
topk_recall_same_class_iou_0.75
```

如果 `same_class_iou_0.50` 仍低于 0.03，不建议直接进入完整小目标微调，应继续检查定位、增强和数据。

## 消融配置

```bash
python tools/train.py --config configs/ablation_baseline_deformable.yaml --gpu 0
python tools/train.py --config configs/ablation_no_p2.yaml --gpu 0
python tools/train.py --config configs/ablation_no_acq.yaml --gpu 0
python tools/train.py --config configs/ablation_no_rsnds.yaml --gpu 0
```

## Smoke Test

```bash
python tools/train.py --config configs/smoke_visdrone_mini.yaml --device cpu
```

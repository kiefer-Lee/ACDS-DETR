# ACDS-DETR_pro 论文实验版改造说明

## 1. 数据与标注诊断

- `datasets/visdrone.py`
  - 支持常见 VisDrone 目录结构：`root/split/images` 与 `root/split/split/images`。
  - 明确 VisDrone 原始 bbox 是 absolute `xywh`，训练统一转换为 clipped `xyxy`。
  - 明确 `category_id` 从 VisDrone `1-10` 映射到训练用 `0-9`，COCO-style eval 再映射回 `1-10`。
  - 增加 `diagnostics()`，统计空标注、最大目标数、小/中/大目标数量、类别分布、缺失标注文件。

为什么有用：bbox 格式、类别映射和越界框是小目标检测 APs/ARs 异常低的高频原因；密集小目标还必须知道 max objects per image，避免 query 数量和 `maxDets` 截断召回。

## 2. 小目标友好后处理与评估

- `utils/metrics.py`
  - `postprocess()` 增加 `min_detections`。
  - 阈值过滤后若候选太少，会保留 top-k 低分候选进入 COCO evaluator。
- `engine/evaluator.py`
  - 传入 `eval.min_detections`。
- `configs/default.yaml`
  - 默认 `score_thresh=0.03`、`max_detections=300`、`min_detections=100`。

为什么有用：小目标分类分数通常低于大目标，过高阈值会让 APs/ARs 被后处理直接压低。COCO-style 评估应让 evaluator 根据 score 排序，而不是在进入评估前过早丢弃候选。

## 3. 高分辨率特征层修复

- `models/backbone.py`
  - 修复额外 feature levels 只能生成一层的问题。
  - 支持从 backbone 最后一层连续下采样生成 P6/P7 等额外层。
  - `use_p2=true` 时可输出 stride-4 P2，配合 `encoder_feature_indices` 让 P2 进入 deformable decoder cross-attention，但跳过二次方复杂度的 vanilla encoder。

为什么有用：小目标在 stride16/32 上定位信息严重丢失；P2/P3 能提供边界与纹理细节。让 P2 跳过全局 encoder 可避免显存爆炸，又保留 decoder 稀疏采样的高分辨率信息。

## 4. EMA 稳定训练

- `utils/ema.py`
  - 新增 `ModelEma`。
- `tools/train.py`
  - 支持 `train.ema.enabled`、`train.ema.decay`、`train.ema.eval`。
  - 保存 checkpoint 时同时保存 `model_ema`。
- `engine/trainer.py`
  - 每次 optimizer step 后更新 EMA。

为什么有用：DETR 类模型小目标匹配噪声较大，EMA 能平滑训练后期权重，通常提升验证稳定性，尤其是 APs/ARs。

## 5. 训练策略默认值

- `configs/default.yaml`
  - scheduler 改为 `multistep`。
  - 默认 `warmup_epochs=2`。
  - 默认启用 AMP、gradient clipping 和 EMA。

为什么有用：Deformable DETR 对学习率和 warmup 敏感；小目标 loss/matching 波动更强，warmup + clipping + EMA 能降低 early collapse 和非有限梯度风险。

## 6. 论文级诊断入口

- `tools/diagnose_project.py`
  - 输出 dataset raw stats、pipeline 后 bbox 健康度、query 数量与 max objects 是否匹配、P2 是否启用、`max_detections` 是否可能截断召回。

推荐运行：

```bash
python tools/diagnose_project.py --config configs/paper_full_small_object.yaml --json outputs/diagnose_full.json
```

## 7. 推荐实验顺序

先跑核心方法消融：

```bash
python tools/train.py --config configs/exp_baseline_stable.yaml --gpu 0
python tools/train.py --config configs/exp_acq_only_stable.yaml --gpu 0
python tools/train.py --config configs/exp_rsnds_only_stable.yaml --gpu 0
python tools/train.py --config configs/exp_acds_full_stable.yaml --gpu 0
```

再跑论文增强配置：

```bash
python tools/train.py --config configs/paper_baseline_original.yaml --gpu 0
python tools/train.py --config configs/paper_ablation_highres.yaml --gpu 0
python tools/train.py --config configs/paper_ablation_p2_p3.yaml --gpu 0
python tools/train.py --config configs/paper_ablation_small_queries.yaml --gpu 0
python tools/train.py --config configs/paper_ablation_small_loss.yaml --gpu 0
python tools/train.py --config configs/paper_ablation_aug.yaml --gpu 0
python tools/train.py --config configs/paper_full_small_object.yaml --gpu 0
```

核心判据：

- APs/ARs 上升；
- AP overall 不显著下降；
- Dense_AP_small / Dense_AR_small 上升；
- Query Collision Rate 下降；
- FPS 和 Params 可接受。

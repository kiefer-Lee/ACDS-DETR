# ACDS-DETR to MMDetection 3.x Migration Plan

本计划用于把旧仓库的 ACDS-DETR 迁移到 MMDetection 3.x + MMEngine 体系内。迁移原则是：不从零重写，不改旧 `models/`、`losses/`、`engine/` 目录；以 mmdet 内置 Deformable DETR 为基底，把 ACQ 和 R-SNDS 做成可独立开关的增量模块，并保留 P2 decoder-only 的小目标设计。

## 1. 目标边界

- 新代码全部放在 `mmdet_acds/` 下，旧实现只读保留。
- 基底采用 mmdet 3.x 的 `DeformableDETR` / `DeformableDETRHead` / `MultiScaleDeformableAttention` / `HungarianAssigner` 相关实现。
- ACQ 必须作为 head/loss 中的可微损失参与反传，不能实现成 hook。
- R-SNDS 必须在 forward 阶段基于 per-query `gamma` 调制 decoder cross-attention sampling offsets，不能做成初始化技巧。
- CUDA op 直接依赖 MMCV 2.x，不复制旧 `models/ops/`。
- mmdet 命令行统一使用 `--cfg-options`，不保留旧 `--opts` 风格。

## 2. 旧模块到新模块映射

| 旧文件 | 核心职责 | 新文件 | 迁移方式 |
|---|---|---|---|
| `models/acds_detr.py` | 组装 backbone、transformer、query、class/bbox head；输出 aux、reference_points | `mmdet_acds/models/acds_detr.py` | 继承 mmdet `DeformableDETR`，只增加 `use_p2`、ACDS transformer/head 配置入口 |
| `models/backbone.py` | ResNet P2/P3/P4/P5/P6 特征、输入投影、冻结层 | mmdet config + `ACDSDeformableDETR` | 优先用 mmdet ResNet + ChannelMapper；P2 用 `out_indices=(0,1,2,3)`，baseline 用 `(1,2,3)` 或标准 Deformable DETR neck |
| `models/transformer.py` | encoder P2 排除、decoder 使用全部层、reference_points 暴露、R-SNDS 注入 | `mmdet_acds/models/acds_transformer.py` | 继承 Deformable DETR transformer/decoder；encoder levels 由 `encoder_level_indices` 控制；decoder cross-attn 替换为 RSNDS 版本 |
| `models/deformable_attention.py` | MSDA wrapper；`gamma` 缩放 offsets | `mmdet_acds/models/rsnds_msda.py` | 继承 MMCV `MultiScaleDeformableAttention`；保持 mmcv op 调用，仅在计算 sampling locations 前乘 `gamma` |
| `models/sampling_modules.py` | `ReliabilityGuidedScaleSampler` 计算 `gamma/rho` | `mmdet_acds/models/rsnds_msda.py` | 保留公式：`gamma=(1-rho)*gamma_base+rho*clip(beta*sqrt(w*h), gamma_min, gamma_max)` |
| `losses/criterion.py` | Hungarian matching、CE/L1/GIoU、aux weight_dict、ACQ 汇总 | `mmdet_acds/models/acds_head.py` | 继承 `DeformableDETRHead`；在 `loss_by_feat` 内补 ACQ，并返回 `loss_acq` 与 `query_collision_rate` |
| `losses/acq_loss.py` | assignment-aware collision decoupling | `mmdet_acds/models/acq_loss.py` | 注册到 `MODELS`；输入 cls scores、bbox preds、reference points、assign results、原图面积 |
| `losses/detr_losses.py` | label/box loss helper；小目标 box loss gain | `mmdet_acds/models/acds_head.py` | mmdet 内置 CE/L1/GIoU 为主；小目标 loss gain 作为 head 可选分支，默认与旧配置一致 |
| `models/matcher.py` | Hungarian cost + `small_object_cost_gain` | `mmdet_acds/models/small_object_assigner.py` | 继承 `HungarianAssigner`，对小目标 gt 的 bbox/GIoU cost 乘 `1 + gain` |
| `datasets/visdrone.py` | VISDRONE_CLASSES 1-based 到 0-based 映射 | `mmdet_acds/datasets/visdrone_metainfo.py` | 只提供 metainfo，数据用 COCO JSON + `CocoDataset` |
| `datasets/transforms.py` | resize、multi-scale、random zoom crop、flip、normalize | `mmdet_acds/configs/_base_/visdrone_coco.py`，必要时 `mmdet_acds/datasets/transforms.py` | 先用 mmdet `RandomChoiceResize`/`RandomFlip`/`PackDetInputs` 对齐；`zoom_crop_*` 若无法等价则实现自定义 transform |
| `datasets/collate.py` | pad image + mask | mmdet dataloader | 使用 mmdet 默认 det data preprocessor padding，不复制 collate |
| `tools/visdrone_to_coco.py` | 离线生成 COCO JSON | `mmdet_acds/tools/convert_visdrone_to_coco.py` 或复用旧脚本 | 保持 category id 为 1-10，name 与 `VISDRONE_CLASSES` 一致 |
| `engine/trainer.py` | AMP、clip grad、EMA、logging、non-finite skip | mmdet runtime config + hooks | 用 `AmpOptimWrapper`/`OptimWrapper`、`clip_grad`、`EMAHook`；non-finite skip 不默认复制，作为风险项 |
| `engine/evaluator.py` / `utils/coco_eval.py` | COCO eval，`max_detections=500` | config `val_evaluator`/`test_evaluator` | 用 `CocoMetric`，显式 `proposal_nums=(100,300,500)` 或确认 mmdet 版本支持；IoU 阈值显式设置 |
| `utils/metrics.py` | dense-small AP/AR，原图坐标口径 | `mmdet_acds/models` 或 `mmdet_acds/evaluation/dense_small_metric.py` | 新增 mmdet metric，按原图 `ori_shape` 和 COCO 原始 annotation area 计算，`dense_small_count_thr=20` |
| `utils/ema.py` | EMA 权重 | mmdet `EMAHook` | config 开关，decay 对齐旧 YAML |

## 3. 新目录与文件清单

计划新增：

- `mmdet_acds/__init__.py`
- `mmdet_acds/models/__init__.py`
- `mmdet_acds/models/acds_detr.py`
- `mmdet_acds/models/acds_transformer.py`
- `mmdet_acds/models/rsnds_msda.py`
- `mmdet_acds/models/acds_head.py`
- `mmdet_acds/models/acq_loss.py`
- `mmdet_acds/models/small_object_assigner.py`
- `mmdet_acds/datasets/__init__.py`
- `mmdet_acds/datasets/visdrone_metainfo.py`
- `mmdet_acds/datasets/transforms.py`，仅在内置 transform 不能等价复现 zoom crop 时新增
- `mmdet_acds/evaluation/__init__.py`
- `mmdet_acds/evaluation/dense_small_metric.py`
- `mmdet_acds/tools/convert_legacy_ckpt.py`，可选
- `mmdet_acds/configs/_base_/visdrone_coco.py`
- `mmdet_acds/configs/_base_/runtime.py`
- `mmdet_acds/configs/_base_/schedule_acds.py`
- `mmdet_acds/configs/acds_detr_r50_visdrone.py`
- `mmdet_acds/configs/stage1_localization_bootstrap.py`
- `mmdet_acds/configs/stage2_acds_full_finetune.py`
- `mmdet_acds/configs/ablation_baseline_deformable.py`
- `mmdet_acds/configs/ablation_no_acq.py`
- `mmdet_acds/configs/ablation_no_rsnds.py`
- `mmdet_acds/configs/ablation_no_p2.py`
- `mmdet_acds/tests/test_acq_loss.py`
- `mmdet_acds/tests/test_rsnds_msda.py`
- `mmdet_acds/tests/test_small_object_assigner.py`
- `mmdet_acds/tests/test_config_equivalence.py`
- `mmdet_acds/tests/test_metric_area_space.py`
- `mmdet_acds/README.md`

禁止修改：

- `models/`
- `losses/`
- `engine/`
- 旧 `datasets/`、`utils/`、`tools/` 除非后续 review 明确要求修旧脚本

## 4. 配置参数对照表

### 数据与增强

| 旧 YAML key | 旧值/语义 | 新 config 路径 | 备注 |
|---|---|---|---|
| `dataset.root` | VisDrone 根目录 | `data_root` | 默认保留服务器路径，可用 `--cfg-options data_root=...` 覆盖 |
| `dataset.train_split` / `val_split` | `train` / `val` | `train_dataloader.dataset.ann_file` / `val_dataloader.dataset.ann_file` | 指向离线 COCO `train.json` / `val.json` |
| `dataset.num_classes` | 10 | `metainfo.classes` + `model.bbox_head.num_classes` | 类顺序必须为旧 `VISDRONE_CLASSES` 的 1 到 10 |
| `dataset.img_size` | short edge | `train_pipeline.RandomChoiceResize.scales` / `test_pipeline.Resize.scale` | 单尺度 eval 使用 `(max_size, img_size)` |
| `dataset.max_size` | long edge 上限 | resize scale 的 long side | mmdet resize 需确认等价的 keep ratio 规则 |
| `dataset.min_area` | 过滤小 bbox | COCO 转换脚本或自定义 filter | 旧默认 1；COCO JSON 应在转换阶段保留 |
| `dataset.max_samples` | 小数据调试 | `train_dataloader.dataset.indices` 或 `filter_cfg`/wrapper | 用于 200 sample 验收 |
| `dataset.augment.multi_scale` | 多尺度 short edge | `RandomChoiceResize.scales=[(max_size, s), ...]` | 对齐 `[896,960,1024,1088,1152]` 等 |
| `dataset.augment.zoom_crop_prob` | zoom crop 概率 | `RandomChoice` 或 `ACDSRandomZoomCrop.prob` | 内置 transform 不能表达 `min_small_keep` 时写自定义 |
| `dataset.augment.zoom_crop_ratio` | crop ratio | `ACDSRandomZoomCrop.ratio_range` | 旧 full/stage2 为 `[0.80,1.0]` |
| `dataset.augment.zoom_crop_min_boxes` | crop 后最少目标 | `ACDSRandomZoomCrop.min_boxes` | 必须保留 |
| `dataset.augment.zoom_crop_min_small_keep` | crop 后最少小目标 | `ACDSRandomZoomCrop.min_small_keep` | mmdet 内置 RandomCrop 无此语义 |
| `dataset.augment.zoom_crop_min_visibility` | 最小可见比例 | `ACDSRandomZoomCrop.min_visibility` | 必须以 crop 前像素面积计算 |
| `dataset.augment.small_area_thr` | 1024 px^2 | `ACDSRandomZoomCrop.small_area_thr` / `ACQLoss.small_area_thr` / dense metric | 必须明确是否原图/当前图，见风险项 |
| `dataset.augment.zoom_crop_attempts` | 尝试次数 | `ACDSRandomZoomCrop.attempts` | 旧 full 为 12 |

### 模型

| 旧 YAML key | 新 config 路径 | 迁移说明 |
|---|---|---|
| `model.num_classes` | `model.bbox_head.num_classes` | 10 |
| `model.num_queries` | `model.num_queries` | full 1000；baseline 300 |
| `model.hidden_dim` | `model.embed_dims` / head transformer embed dims | 256 |
| `model.nheads` | `model.encoder.layer_cfg.self_attn_cfg.num_heads` / decoder attn | 8 |
| `model.num_feature_levels` | `model.num_feature_levels` / neck outputs | P2 full 为 5，baseline 为 4 |
| `model.enc_layers` | `model.encoder.num_layers` | paper full 6；多数 ablation 3 |
| `model.dec_layers` | `model.decoder.num_layers` | 6 |
| `model.dim_feedforward` | `feedforward_channels` | 1024 |
| `model.dropout` | `dropout` | 0.1 |
| `model.num_points` | `num_points` | 4 |
| `model.return_intermediates` | decoder `return_intermediate=True` + head aux | mmdet Deformable DETR 通常固定返回中间层用于 aux |
| `model.backbone` | `model.backbone.depth` | resnet50 |
| `model.pretrained_backbone` | `init_cfg` | 使用 mmdet 预训练权重机制 |
| `model.use_p2` | `model.use_p2` + backbone `out_indices` | P2 只进 decoder cross-attn，不进 encoder |
| `model.train_backbone_layers` | `paramwise_cfg` 或 `frozen_stages`/`requires_grad` | 旧 stage1 冻结全部 backbone，full 训练 layer3/4 |
| `model.encoder_feature_indices` | `model.encoder_level_indices` | full `[1,2,3,4]`，baseline `None` |
| `model.scale_aware_query.enabled` | `model.scale_aware_query.enabled` | 需要在 detector 初始化 query embedding 时加 group embedding |
| `model.scale_aware_query.groups` | `model.scale_aware_query.groups` | full 为 4 |
| `model.scale_aware_query.strength` | `model.scale_aware_query.strength` | stage1 0.20，full 0.35 |

### ACQ / R-SNDS / loss

| 旧 YAML key | 新 config 路径 | 迁移说明 |
|---|---|---|
| `acq.enabled` | `model.bbox_head.acq_loss.enabled` | 关闭时返回可反传 0 |
| `acq.lambda_acq` | `model.bbox_head.loss_acq.loss_weight` | 旧 criterion 的 `weight_dict["loss_acq"]` |
| `acq.small_area_thr` | `model.bbox_head.acq_loss.small_area_thr` + metric | 默认 1024 px^2 |
| `acq.topk_unmatched` | `model.bbox_head.acq_loss.topk_unmatched` | 默认 30 |
| `acq.delta` | `model.bbox_head.acq_loss.delta` | 默认 0.03 normalized distance |
| `acq.sigma` | `model.bbox_head.acq_loss.sigma` | 默认 0.06 |
| `acq.min_score` | `model.bbox_head.acq_loss.min_score` | 默认 0.40 |
| `acq.apply_last_n_layers` | `model.bbox_head.acq_apply_last_n_layers` | 只对最后 N 层算 ACQ |
| `rsnds.enabled` | `model.decoder.layer_cfg.cross_attn_cfg.rsnds.enabled` | 关闭时 `gamma=gamma_base` |
| `rsnds.beta` | `...rsnds.beta` | 默认 1.0 |
| `rsnds.gamma_base` | `...rsnds.gamma_base` | 默认 1.0 |
| `rsnds.gamma_min` / `gamma_max` | `...rsnds.gamma_min` / `gamma_max` | 默认 0.35 / 1.25 |
| `rsnds.reliability` | `...rsnds.reliability` | 默认 `cls_conf` |
| `loss.eos_coef` | `model.bbox_head.loss_cls.bg_cls_weight` 或 class weight | 0.1 |
| `loss.cost_class` | `model.train_cfg.assigner.match_costs[*].weight` | 2.0 |
| `loss.cost_bbox` | `model.train_cfg.assigner.match_costs[*].weight` | 6.0 或 7.0 |
| `loss.cost_giou` | `model.train_cfg.assigner.match_costs[*].weight` | 2.0 或 3.0 |
| `loss.weight_class` | `model.bbox_head.loss_cls.loss_weight` | 1.0 |
| `loss.weight_bbox` | `model.bbox_head.loss_bbox.loss_weight` | 6.0 或 7.0 |
| `loss.weight_giou` | `model.bbox_head.loss_iou.loss_weight` | 2.0 或 3.0 |
| `loss.aux_loss_weight` | `model.bbox_head.aux_loss_weight` | 旧实现 aux loss 乘 0.5 |
| `loss.aux_loss_layers` | `model.bbox_head.aux_loss_layers` | `all` |
| `loss.small_object_cost_gain` | `model.train_cfg.assigner.small_object_cost_gain` | small gt cost gain |
| `loss.small_object_loss_gain` | `model.bbox_head.small_object_loss_gain` | bbox/GIoU small gt gain |

### 训练、调度、评估

| 旧 YAML key | 新 config 路径 | 迁移说明 |
|---|---|---|
| `train.epochs` | `train_cfg.max_epochs` 或 `max_iters` | 验收 smoke 用 `--cfg-options train_cfg.max_iters=20` |
| `train.batch_size` | `train_dataloader.batch_size` | 默认 1 |
| `train.num_workers` | `train_dataloader.num_workers` | 默认 8，smoke 可覆盖 0 |
| `train.lr` | `optim_wrapper.optimizer.lr` | full 2.5e-5 或 ablation 4e-5 |
| `train.lr_backbone` | `optim_wrapper.paramwise_cfg.custom_keys.backbone.lr_mult` | 按 `lr_backbone / lr` 计算 |
| `train.weight_decay` | `optim_wrapper.optimizer.weight_decay` | 1e-4 |
| `train.clip_max_norm` | `optim_wrapper.clip_grad.max_norm` | 0.05 |
| `train.amp` | `optim_wrapper.type` | true 用 `AmpOptimWrapper`，false 用 `OptimWrapper` |
| `train.amp_init_scale` | `optim_wrapper.loss_scale.initial_scale` | 512 |
| `train.warmup_epochs` | `param_scheduler[0]` | 旧按 epoch warmup，mmdet 可转为 iter warmup |
| `train.lr_drop_epochs` | `param_scheduler[1].milestones` | 例如 `[120,145]` |
| `train.lr_drop_gamma` | `param_scheduler[1].gamma` | 0.1 |
| `train.ema.enabled` | `custom_hooks.EMAHook` | true 开启 |
| `train.ema.decay` | `custom_hooks.EMAHook.momentum` | 需确认 mmdet 参数是 momentum 还是 decay，避免反向 |
| `train.resume` | `resume` / `load_from` | stage2 用 `load_from=stage1 best` 且 `resume=False` |
| `eval.score_thresh` | metric 或 test cfg | 默认 0.01 |
| `eval.max_detections` | `val_evaluator.proposal_nums` / `test_evaluator.proposal_nums` | 旧默认 500，目标 `maxDets=(100,300,500)` 或更大 |
| `eval.min_detections` | 自定义 postprocess 如需 | 旧 eval 至少保留 200，CocoMetric 默认无此逻辑，列为差异 |
| `eval.iou_thresholds` | `CocoMetric.iou_thrs` | `[0.50,...,0.95]` |
| `eval.dense_small_count_thr` | `DenseSmallCocoMetric.dense_small_count_thr` | 默认 20 |

## 5. 配置文件落地策略

- `acds_detr_r50_visdrone.py`：对应 `paper_full_small_object.yaml`，full method，P2 + ACQ + R-SNDS，1000 queries。
- `stage1_localization_bootstrap.py`：对应 stage1，ACQ/R-SNDS 关闭，backbone 冻结，`load_from=None`。
- `stage2_acds_full_finetune.py`：对应 stage2，ACQ/R-SNDS 开启，`load_from` 指向 stage1 best，通过 `resume=False` 重置 optimizer。
- `ablation_baseline_deformable.py`：300 queries，4 feature levels，`use_p2=False`，ACQ/R-SNDS/scale-aware query 关闭；用于与 mmdet 内置 Deformable DETR 参数量对比。
- `ablation_no_p2.py`：full 设置但 `use_p2=False`，4 feature levels。
- `ablation_no_acq.py`：full 设置但 `acq.enabled=False`，`lambda_acq=0`。
- `ablation_no_rsnds.py`：full 设置但 `rsnds.enabled=False`，forward 中 `gamma=gamma_base`。

## 6. 最小单测计划

- `test_acq_loss_disabled_returns_zero_and_grad`：`enabled=False` 或 `lambda_acq=0` 时 `loss_acq` 为 0，且通过 `pred_boxes.sum()*0` 保持计算图。
- `test_acq_small_thr_same_coordinate_space`：构造原图尺寸与 resize 尺寸不同的样本，断言 `small_thr_norm` 与 target area 使用同一坐标系；迁移后 metric 用原图面积。
- `test_rsnds_disabled_gamma_base`：关闭 R-SNDS 时所有 query 的 `gamma == gamma_base`。
- `test_rsnds_cls_conf_formula`：给定 bbox wh 和 logits，断言 `gamma` 与旧公式一致。
- `test_small_object_assigner_gain`：小目标 gt 的 bbox/GIoU cost 被 `1 + small_object_cost_gain` 放大，大目标不变。
- `test_aux_loss_weight_dict_all_layers`：所有 decoder aux 层的 ce/bbox/giou key 都有权重，防止旧缺陷复现。
- `test_baseline_param_count_close_to_mmdet_deformable_detr`：关闭 ACQ/R-SNDS/P2/scale-aware query 后，与 mmdet 内置 deformable detr 参数量差异小于 0.5%。

## 7. 验证与验收流程

1. 环境检查：
   - Python + PyTorch 可用。
   - 安装 `mmengine>=0.7`、`mmcv>=2.0`、`mmdet>=3.0`。
   - 运行 `python -c "import mmengine, mmcv, mmdet"`。
2. 单测：
   - `pytest mmdet_acds/tests -q`
3. CPU smoke：
   - `python tools/train.py mmdet_acds/configs/acds_detr_r50_visdrone.py --cfg-options train_dataloader.batch_size=1 train_cfg.max_iters=20`
4. 四个 ablation 配置 smoke：
   - baseline / no_p2 / no_acq / no_rsnds 各跑最小 iter。
5. 配置等价性：
   - baseline 结构与 mmdet 内置 `deformable-detr_r50_16xb2-50e_coco.py` 对比参数量。
   - 训练 1 epoch 记录 `loss_ce/loss_bbox/loss_giou`，与旧 `ablation_baseline_deformable.yaml` 日志对照。
6. 小数据数值：
   - `max_samples=200` 跑 10 epoch，对比 full method 的 `AP_small/AR_small` 与旧仓库同条件结果，允许 ±10%。

## 8. 风险项

- 当前本机未安装 `mmengine`、`mmcv`、`mmdet`，无法在本轮直接执行 mmdet smoke test；需要安装后完成验收 A-C。
- mmdet 3.x 的 Deformable DETR 内部类路径随小版本变化较多，`acds_transformer.py` 需要以实际安装版本源码为准，优先继承公开 registry 类，必要时只替换 decoder cross-attn 模块。
- mmdet 的 `CocoMetric` 默认 `maxDets` 通常是 `(100,300,1000)` 或版本相关，必须显式设置并检查输出 key 是否按 500 统计。
- 旧 `eval.min_detections` 会在低置信预测不足时补 top-k；标准 mmdet `CocoMetric` 没有该逻辑。若为了公平复现旧日志，需自定义 metric 或 test cfg 后处理；若为了和 mmdet 内置 DETR 公平比较，应在计划 review 时确认是否移除此旧策略。
- 旧代码在训练中用 resize 后的 `target["area"]` 参与部分 small-object loss/metric，但需求要求迁移后 AP_small 统一到原图坐标系。迁移中需要把训练增强空间、loss 空间、metric 空间明确分开，并用单测锁定。
- EMAHook 的参数语义可能是 `momentum` 而不是直接 `decay`，需要查当前 mmdet/mmengine 版本源码后设置。
- Stage1 旧配置 `train_backbone_layers=[]` 表示冻结全部 backbone；mmdet 的 `frozen_stages` 不完全等价，需要 paramwise 或自定义 constructor 精确控制。
- P2 decoder-only 不是 mmdet 标准 Deformable DETR 路径，encoder level 切片和 decoder spatial_shapes/level_start_index 必须保持一致，否则会出现 shape 兼容但语义错误。
- `scale_aware_query` 会增加额外 embedding，做 baseline 参数量对比时必须关闭。

## 9. 需要 review 的决定

- `eval.min_detections=200`：为复现旧仓库 baseline 日志保留，还是为公平 mmdet 对比去掉。
- `zoom_crop_*`：是否接受 mmdet 内置 RandomCrop 的近似实现，还是必须新增 `ACDSRandomZoomCrop` 完整复现 `min_small_keep/min_visibility`。
- 训练循环中的 non-finite skip：是否迁入 mmdet custom hook，还是使用 mmdet 默认异常策略。
- stage1/stage2 的配置是否需要同时保留旧 epoch-based 训练，还是统一换成 iter-based 以匹配验收命令。

## 10. 验收清单

- [ ] `MIGRATION_PLAN.md` 已由人 review。
- [ ] 新增 `mmdet_acds/` 目录且旧 `models/`、`losses/`、`engine/` 未改动。
- [ ] ACQ 最小单测通过，关闭时 `loss_acq=0`。
- [ ] R-SNDS 最小单测通过，关闭时 `gamma=gamma_base`。
- [ ] small-object assigner 单测通过。
- [ ] aux loss 所有层 key 进入 loss weight dict。
- [ ] small target area threshold 与目标面积坐标系单测通过。
- [ ] 四个 ablation 配置可独立构建并进入对应分支。
- [ ] CPU smoke 20 iter 跑通。
- [ ] baseline 参数量与 mmdet 内置 Deformable DETR 差异小于 0.5%。
- [ ] baseline 1 epoch `loss_ce/loss_bbox/loss_giou` 与旧仓库日志完成对照。
- [ ] 200 sample / 10 epoch full method `AP_small/AR_small` 与旧仓库同条件结果完成对照。
- [ ] 迁移文档写明最低 `mmcv/mmdet/pytorch` 版本。

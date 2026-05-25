# ACDS-DETR: Assignment-aware Collision Decoupling and Reliability-guided Scale-normalized Sampling for Dense Small Object Detection

ACDS-DETR 是一个面向 VisDrone 等密集小目标检测场景的 Deformable DETR 系列模型实现。项目目标不是重新发明 DETR，而是在成熟的多尺度可变形注意力架构上，针对密集小目标中常见的 query 分配冲突、采样偏移失准和小目标召回不足问题，加入可解释、可消融、可可视化的改进模块。

当前代码已经从早期实验原型调整为更接近论文实验的实现：encoder 和 decoder 均基于 Multi-Scale Deformable Attention；服务器上优先使用官方 CUDA op，本地调试时自动回退到 PyTorch `grid_sample` 实现；高分辨率 P2 特征、1000 queries、EMA、AMP、小目标友好评估和两阶段训练策略均已配置。

## 解决的问题

密集小目标检测对 DETR 类模型并不友好，主要瓶颈来自三类失败模式：

1. **Query assignment collision**  
   在密集区域中，多个 object queries 容易围绕同一个显著目标竞争，而相邻小目标缺少独立 query 覆盖。表面上模型输出了很多候选框，实际 recall 仍被 query 分配质量限制。

2. **Scale-insensitive sampling deviation**  
   Deformable attention 的采样偏移由 query feature 自由预测。对小目标而言，采样点只要偏离几个像素，就可能落到背景或邻近目标上，导致分类和定位相互拖累。

3. **High-resolution cost explosion**  
   小目标需要 stride-4/stride-8 高分辨率特征，但普通 Transformer encoder 的全局自注意力会在高分辨率 token 上产生二次复杂度。ACDS-DETR 使用成熟的多尺度可变形注意力 encoder，避免高分辨率输入下的全局 attention 矩阵爆炸。

## 模型结构

```text
Image
  -> ResNet backbone
  -> P2/P3/P4/P5/P6 multi-scale features
  -> Deformable encoder
       - encoded levels: P3/P4/P5/P6 by default
       - P2 is preserved for decoder cross-attention
  -> Object queries with scale-aware query embeddings
  -> Deformable decoder
       - R-SNDS adjusts sampling radius by prediction reliability and box scale
  -> Classification and box heads
  -> Hungarian matching
       - small-object aware matching cost
       - ACQ collision decoupling loss
  -> COCO-style and dense-small-object evaluation
```

默认设计保留 P2 进入 decoder cross-attention，但不把 P2 放进 encoder。这样可以让 decoder 使用 stride-4 细节定位小目标，同时避免 encoder 在超高分辨率 token 上消耗过大。

## 创新点

### 1. ACQ: Assignment-aware Collision Decoupling

ACQ 是训练阶段的 query 解耦约束。它不是简单地让所有 query 彼此远离，而是利用 Hungarian matching 结果识别“冗余竞争 query”：

- matched query 负责真实小目标；
- unmatched 或低质量 query 若在空间上靠近该 matched query，则视为 collision candidate；
- 只对这些 assignment-aware collision pairs 施加轻量 repulsion。

这种设计的重点是避免误伤真实相邻小目标。密集小目标本来就可能彼此很近，因此普通 query repulsion 容易把合理预测也推开。ACQ 的可发表价值在于：它把 query collision 从一个模糊现象，落到了 matching 后的可度量 pair 集合上，可视化和消融都比较清楚。

相关代码：

- `losses/acq_loss.py`
- `losses/criterion.py`
- `models/acds_detr.py`

### 2. R-SNDS: Reliability-guided Scale-normalized Deformable Sampling

R-SNDS 修改 decoder cross-attention 中 sampling offsets 的有效半径：

```text
gamma_scale = beta * sqrt(w * h)
rho = max_class_probability
gamma = (1 - rho) * gamma_base + rho * clip(gamma_scale, gamma_min, gamma_max)
sampling_location = reference_point + gamma * offset
```

直觉是：预测还不可靠时，采样范围接近默认值；预测逐渐可靠后，小目标 query 的采样半径会收缩到更符合目标尺度的区域，减少采到背景或邻近目标的概率。该模块不增加采样点数量，推理额外开销很小。

相关代码：

- `models/sampling_modules.py`
- `models/transformer.py`
- `models/deformable_attention.py`

### 3. Mature Deformable Encoder/Decoder Backbone

早期原型中的 encoder 使用普通 `MultiheadAttention`，在高分辨率多尺度 token 上会非常慢，也不符合 Deformable DETR 的成熟实现。当前版本已经替换为 Multi-Scale Deformable Attention：

- CUDA op 可用时自动使用官方扩展；
- CUDA op 不可用时自动使用 PyTorch fallback，便于 CPU/Windows smoke test；
- encoder reference points 使用有效区域比例生成，支持 padding mask；
- decoder 仍保留 ACDS 的 R-SNDS 采样半径调制。

这部分不是论文创新点，应作为可靠工程基础和公平 baseline 的必要条件。

## 创新性与发表判断

ACDS-DETR 的思路具备论文雏形，但是否足以发表取决于实验能否证明以下三点：

1. **性能收益主要来自 ACQ 和 R-SNDS，而不是训练技巧。**  
   必须给出 baseline、no ACQ、no R-SNDS、no P2、full method 的完整消融。

2. **收益集中体现在小目标和密集小目标。**  
   建议报告 AP、AP50、AP75、APs、APm、APl、ARs，以及自定义 dense-small subset 的 AP/AR。

3. **机制指标能解释性能提升。**  
   建议增加 query collision rate、top-k same-class IoU recall、sampling point visualization、reference point distribution 等分析。

如果 full model 只提升整体 mAP，而 ACQ/R-SNDS 消融不明显，则创新点不足以支撑强论文；如果 APs/ARs 和 dense subset 显著提升，并且 collision 与 sampling 可视化能解释提升，则可以支撑一篇小目标检测方向的会议或期刊论文。

## 配置文件

所有主配置均保留服务器 Linux 数据路径：

```text
/data/libaichuan/Projects/SOD/Datasets/VisDrone
```

本地验证时不要修改 YAML，使用 `--opts dataset.root=...` 临时覆盖即可。

主要配置：

- `configs/paper_full_small_object.yaml`：论文主实验配置，6-layer deformable encoder/decoder，P2/P3/P4/P5/P6，1000 queries，ACQ，R-SNDS，EMA，AMP。
- `configs/stage1_localization_bootstrap.yaml`：定位启动阶段，关闭 ACQ/R-SNDS，先让模型学到稳定同类定位。
- `configs/stage2_acds_full_finetune.yaml`：完整 ACDS 微调阶段，从 stage1 权重恢复。
- `configs/default.yaml`：单阶段完整配置。
- `configs/ablation_baseline_deformable.yaml`：Deformable-DETR-like baseline。
- `configs/ablation_no_p2.yaml`、`configs/ablation_no_acq.yaml`、`configs/ablation_no_rsnds.yaml`：关键消融。
- `configs/smoke_visdrone_mini.yaml`：小规模 smoke test。

## 安装与 CUDA op

在 Linux 服务器上建议先编译官方 MultiScaleDeformableAttention CUDA op：

```bash
cd /data/libaichuan/Projects/SOD/ACDS-DETR
cd models/ops
sh make.sh
python test.py
cd ../..
```

如果 op 未编译，代码会自动使用 PyTorch fallback。fallback 适合 smoke test 和调试，不适合正式训练速度评估。

## 推荐训练流程

### 1. 数据与配置诊断

```bash
python tools/diagnose_project.py \
  --config configs/paper_full_small_object.yaml
```

重点检查：

- `max_boxes_per_image` 是否超过 `num_queries`；
- `eval.max_detections` 是否足够大；
- 小目标比例是否符合实验目标；
- bbox 是否越界或面积异常。

### 2. Stage 1: localization bootstrap

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py \
  --config configs/stage1_localization_bootstrap.yaml
```

Stage 1 的目标不是最终 AP，而是让 query 先具备可靠定位能力。训练后运行：

```bash
python tools/debug_eval_sanity.py \
  --checkpoint outputs/stage1_localization_bootstrap/best_map.pth \
  --gpu 0 \
  --use-ema \
  --batches 8
```

如果 `topk_recall_same_class_iou_0.50` 仍低于 0.03，不建议直接进入 Stage 2，应先检查数据、类别映射、bbox 归一化和学习率。

### 3. Stage 2: full ACDS-DETR fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py \
  --config configs/stage2_acds_full_finetune.yaml \
  --resume outputs/stage1_localization_bootstrap/best_map.pth \
  --reset-optimizer
```

### 4. Paper full config

如果需要单独跑论文主配置：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py \
  --config configs/paper_full_small_object.yaml
```

### 5. Final evaluation

```bash
python tools/eval.py \
  --checkpoint outputs/paper_full_small_object/best_ap_small.pth \
  --gpu 0 \
  --use-ema
```

## 本地 smoke test

本地 Windows 或 CPU 环境可以临时覆盖数据路径和模型规模：

```bash
python tools/train.py \
  --config configs/smoke_visdrone_mini.yaml \
  --device cpu \
  --output-dir outputs/local_smoke \
  --opts dataset.root=../Datasets/VisDroneMini dataset.max_samples=1 \
         model.hidden_dim=64 model.dim_feedforward=128 \
         model.num_queries=20 model.num_feature_levels=3 \
         model.enc_layers=1 model.dec_layers=1 model.nheads=4 \
         model.use_p2=false model.encoder_feature_indices=null \
         train.epochs=1 train.num_workers=0 eval.use_coco_eval=false
```

该测试只验证代码链路，不代表模型性能。

## 建议的论文实验表

主表建议包含：

- AP / AP50 / AP75
- APs / APm / APl
- AR@100 / ARs / ARm / ARl
- Dense_AP_small / Dense_AR_small
- FPS / Params

消融建议包含：

- baseline Deformable DETR
- baseline + P2
- baseline + ACQ
- baseline + R-SNDS
- full ACDS-DETR
- full without ACQ
- full without R-SNDS
- full without P2

机制分析建议包含：

- ACQ 前后的 query collision rate；
- top-k same-class IoU recall；
- reference points 分布；
- R-SNDS 前后的 sampling point 可视化；
- dense-small subset 上的 recall 改善。

## 工程注意事项

- 正式训练必须编译 CUDA op，否则 fallback 会明显慢。
- `grad_norm` 是裁剪前梯度范数，不能直接等同于实际更新幅度。
- 如果出现连续 non-finite gradient，应停止该次训练，不建议从连续 skip 后的 `last.pth` 恢复。
- VisDrone 原始类别为 1-10，训练内部映射为 0-9，COCO-style eval 会再映射回 1-10。
- 小目标评估中 `max_detections` 不应过小，默认主配置使用 500。

## 当前定位

ACDS-DETR 当前更适合作为“可验证的论文方法原型”，而不是已经完成调参的 SOTA 结果包。代码已替换掉早期不适合高分辨率训练的全局 encoder，并保留了可发表所需的模块化消融入口。下一步应集中在服务器上完成 baseline/full/ablation 的稳定训练和机制可视化，只有当这些实验闭环成立时，ACDS-DETR 才能被严肃地包装成论文贡献。

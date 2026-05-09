# ACDS-DETR：面向密集小目标检测的分配感知查询解耦与可靠性引导尺度采样

## 0. 论文级判断

**结论：可以作为论文级 idea，但需要按“机制问题 + 有约束的方法设计 + 可验证证据链”来写。**

ACDS-DETR 的核心价值不在于简单堆叠小目标增强技巧，而在于针对 Deformable DETR decoder 侧的两个具体失败模式提出修正：

1. **Query assignment collision**：密集小目标中，多个 object queries 容易围绕同一目标或同一局部区域竞争，导致相邻小目标缺少有效 query 覆盖。
2. **Scale-insensitive sampling deviation**：deformable attention 的 sampling offsets 缺少对目标尺度和预测可靠性的显式约束，小目标 query 的采样点容易落到背景或邻近目标上。

因此，ACDS-DETR 可以作为一个论文 idea。它的论文表述应聚焦于 **decoder 机制修正**，而不是泛泛地说“提高小目标检测”。核心贡献应限定为：

- 分配感知的 query collision 建模；
- 可靠性引导的尺度归一化 deformable sampling；
- 面向密集小目标的机制验证实验。

## 1. 方法名称

**ACDS-DETR** 表示：

```text
Assignment-aware Collision Decoupling and Reliability-guided Scale-normalized Sampling
```

中文可写为：

```text
面向密集小目标检测的分配感知查询解耦与可靠性引导尺度采样
```

推荐论文题目：

```text
ACDS-DETR: Assignment-aware Collision Decoupling and Reliability-guided Scale-normalized Sampling for Dense Small Object Detection
```

## 2. 问题定义

给定基于 Deformable DETR 的检测器，decoder 通过 object queries 和 multi-scale deformable attention 从 encoder memory 中聚合目标特征。对于密集小目标场景，模型容易出现两类 decoder 侧问题。

### 2.1 Query Assignment Collision

DETR 类方法依赖 Hungarian matching 建立 query 与 ground truth 的一对一匹配。该机制避免了 NMS，但在密集小目标中会出现一种隐性浪费：

```text
多个高响应 queries 聚集到同一小目标附近
    ↓
Hungarian matching 只允许其中一个 query 成为该目标正样本
    ↓
其他 query 变成 unmatched 或低质量匹配
    ↓
相邻小目标缺少独立 query 覆盖
    ↓
AR_small / AP_small 下降
```

这不是单纯的 query 数量不足，而是 **query 利用率不足**。

### 2.2 Scale-insensitive Sampling Deviation

Deformable attention 为每个 query 预测少量 sampling offsets。该机制高效，但 offsets 主要由 query feature 自由预测，并没有显式感知当前目标尺度和预测可靠性。

对于小目标，目标区域很小，采样点稍微偏移就可能落到背景或邻近目标上：

```text
小目标空间范围有限
    ↓
采样点偏离目标区域
    ↓
attention 特征被背景或邻近目标污染
    ↓
分类和定位质量下降
```

因此，小目标检测不只需要更高分辨率特征，也需要更稳定的 decoder sampling 机制。

## 3. 核心贡献

本文贡献建议写成三点：

1. **提出 Assignment-aware Collision Decoupling，简称 ACQ。**  
   ACQ 利用 Hungarian assignment 的匹配结果识别冗余竞争 queries，只对“同一小目标附近的冗余 query 响应”施加解耦约束，避免简单 query repulsion 误伤真实相邻目标。

2. **提出 Reliability-guided Scale-normalized Deformable Sampling，简称 R-SNDS。**  
   R-SNDS 将当前预测框尺度引入 deformable attention 的采样半径，同时使用预测可靠性门控，避免 decoder 早期不可靠 box 过度控制采样范围。

3. **建立面向密集小目标的机制验证协议。**  
   除 AP / AP_small 外，报告 query collision rate、AR_small、密集区域子集结果和 sampling point 可视化，以证明方法确实缓解 decoder 侧失败模式。

## 4. 整体框架

ACDS-DETR 以 Deformable DETR 为基础，主要修改 decoder 侧：

```text
输入图像
    ↓
Backbone + multi-scale features
    ↓
Transformer encoder
    ↓
Decoder object queries
    ↓
[ACQ] assignment-aware query collision decoupling
    ↓
[R-SNDS] reliability-guided scale-normalized sampling
    ↓
分类头与边框回归头
    ↓
检测结果
```

其中：

- **ACQ** 是训练阶段的辅助约束，不增加推理开销；
- **R-SNDS** 修改 decoder cross-attention 的 sampling point 生成方式；
- P2、高分辨率输入、更多 queries 等属于可选实验配置，不作为核心创新。

## 5. ACQ：分配感知查询碰撞解耦

### 5.1 动机

简单地让所有近距离 queries 相互远离并不合适，因为密集小目标本来就可能彼此非常接近。论文级设计必须区分两种情况：

- 合法情况：两个相邻小目标分别拥有各自匹配 query；
- 冲突情况：多个 queries 围绕同一个小目标竞争，其中只有一个 query 是有效匹配。

ACQ 的关键是利用 Hungarian matching 结果识别第二种情况。

### 5.2 碰撞对定义

设第 `i` 个 query 在 decoder 第 `t` 层的 reference point 为：

```text
r_i^t = (x_i^t, y_i^t)
```

预测框为：

```text
b_i^t = (x_i^t, y_i^t, w_i^t, h_i^t)
```

分类置信度为：

```text
s_i^t = max_c p_i^t(c)
```

若 query `j` 匹配到小目标 ground truth，query `i` 未匹配或匹配质量较低，并且二者在空间上高度接近，则认为 `(i, j)` 是一个 assignment-aware collision pair。

碰撞 pair 集合 `Omega` 可由以下条件构造：

1. `j` 是 matched small-object query；
2. `i` 是 unmatched query 或 low-quality matched query；
3. `IoU(b_i, b_j)` 或 `||r_i - r_j||` 表明二者存在局部竞争；
4. `s_i` 和 `s_j` 高于置信度阈值。

### 5.3 ACQ 损失

定义小目标权重：

```text
m_j = exp(-A_j / tau_s)
```

其中 `A_j` 是目标面积，`tau_s` 是尺度温度。目标越小，`m_j` 越大。

定义 collision score：

```text
C_ij = s_i s_j exp(-||r_i - r_j||_2 / sigma) m_j
```

ACQ 损失为：

```text
L_acq = sum_(i,j in Omega) C_ij max(0, delta - ||r_i - r_j||_2)
```

最终训练损失：

```text
L = L_det + lambda_acq L_acq
```

其中 `L_det` 是 Deformable DETR 原始分类、L1 box 和 GIoU 损失。

### 5.4 为什么不是普通 query repulsion

普通 query repulsion 通常只基于空间距离，容易把真实相邻小目标对应的 queries 也推开。ACQ 的不同点是：

```text
只约束 assignment 之后被识别为冗余竞争的 query pair，
保留真实相邻目标对应的合法 matched queries。
```

这使 ACQ 更符合 DETR 一对一匹配机制，也更容易通过实验解释。

## 6. R-SNDS：可靠性引导尺度归一化采样

### 6.1 动机

Deformable attention 的标准采样形式可写为：

```text
p_{i,l,k} = r_i + Delta p_{i,l,k}
```

其中 `r_i` 是 reference point，`Delta p_{i,l,k}` 是第 `l` 个特征层、第 `k` 个采样点的预测偏移。标准机制并不显式限制小目标 query 的采样半径。

直接使用预测框尺度缩放采样偏移也有风险，因为 decoder 早期 box prediction 不稳定。R-SNDS 因此引入可靠性门控。

### 6.2 采样半径

根据当前预测框尺度得到尺度因子：

```text
gamma_i^scale = clip(beta sqrt(w_i h_i), gamma_min, gamma_max)
```

定义预测可靠性：

```text
rho_i = sigmoid(MLP(q_i))
```

也可以在轻量版本中使用分类置信度近似：

```text
rho_i = max_c p_i(c)
```

最终采样因子：

```text
gamma_i = (1 - rho_i) gamma_base + rho_i gamma_i^scale
```

新的 sampling point 为：

```text
p_{i,l,k} = r_i + gamma_i Delta p_{i,l,k}
```

当预测不可靠时，采样半径接近默认值；当预测可靠时，采样半径更多服从目标尺度。

### 6.3 与 Deformable DETR 的差异

R-SNDS 不增加 sampling point 数量，而是改变已有 offsets 的空间分布：

```text
标准 Deformable DETR:
offsets 由 query 自由预测

R-SNDS:
offsets 由目标尺度归一化，并由预测可靠性门控
```

因此，它的目标不是扩大模型容量，而是减少小目标 attention 的无效采样。

## 7. 论文风险与应对

### 风险 1：ACQ 被认为只是启发式正则

应对方式：

- 与普通 query repulsion 做对比；
- 报告 query collision rate；
- 可视化 ACQ 前后的 reference points；
- 证明 ACQ 主要提升 `AR_small` 和密集区域召回，而不是无差别影响所有目标。

### 风险 2：R-SNDS 被认为只是手工 scale-aware sampling

应对方式：

- 消融 reliability gate；
- 比较固定尺度缩放、仅 box 尺度缩放、R-SNDS；
- 可视化 sampling points；
- 分析 decoder 早期和后期的采样半径变化。

### 风险 3：工程增强掩盖核心方法

应对方式：

- 核心实验固定 query 数量、backbone、输入尺度和训练 schedule；
- P2、高分辨率、900 queries 只作为附加实验；
- 论文主表应先报告纯 ACDS-DETR 对 Deformable DETR 的增益。

## 8. 实验设计

### 8.1 数据集

推荐：

- **COCO 2017**：验证通用检测与 `AP_small`；
- **VisDrone2019**：验证密集航拍小目标；
- **AI-TOD**：验证极小目标检测。

### 8.2 对比方法

最低必要对比：

- Deformable DETR；
- DINO 或 DN-DETR；
- ACDS-DETR。

可选对比：

- DAB-DETR；
- Faster R-CNN；
- RetinaNet；
- 其他小目标检测方法。

### 8.3 指标

主指标：

- AP；
- AP50 / AP75；
- AP_small / AP_medium / AP_large；
- AR_small；
- Params；
- FLOPs；
- FPS。

机制指标：

- Query Collision Rate；
- dense-small subset AP / AR；
- sampling point target-hit ratio；
- reference point 分布可视化。

### 8.4 消融实验

| ID | ACQ | R-SNDS | 目的 |
| --- | --- | --- | --- |
| A0 | 关闭 | 关闭 | Deformable DETR baseline |
| A1 | 开启 | 关闭 | 验证 assignment-aware collision decoupling |
| A2 | 关闭 | 开启 | 验证 reliability-guided scale-normalized sampling |
| A3 | 开启 | 开启 | 验证完整 ACDS-DETR |

必要敏感性实验：

- `lambda_acq`；
- `delta`；
- `sigma`；
- `tau_s`；
- `gamma_min` / `gamma_max`；
- `gamma_base`；
- reliability gate 类型；
- R-SNDS 作用于全部 decoder layers 或后几层。

## 9. 可选工程增强

以下内容有助于提高小目标实验效果，但不建议写成 ACDS-DETR 的核心创新：

- P2/P3/P4/P5/P6 多尺度特征；
- 高分辨率输入；
- 600 或 900 queries；
- small-object-aware matching cost；
- small-object-aware loss reweighting；
- 小目标可见性约束的数据增强；
- COCO evaluator 的 `maxDets` 调整；
- FN 分析和预测可视化工具。

论文中可以将这些作为 **implementation details**、**stronger setting** 或 **additional analysis**。如果主方法声称“不增加 query 数量”，主实验就必须保持 query 数量与 baseline 一致；更多 query 只能作为附加实验。

## 10. 预期结果写法

不建议在 README 或论文草稿中写“必然提升 +X AP”。更稳妥的写法是：

```text
We expect larger gains on AP_small and AR_small than on AP_medium/AP_large,
because the proposed modules directly target query collision and sampling deviation
in dense small-object regions.
```

中文表述：

```text
如果方法有效，提升应主要体现在 AP_small、AR_small 和密集小目标子集上；
若整体 AP 提升但机制指标没有改善，则说明收益可能来自训练或配置因素，而不是 ACDS-DETR 的核心设计。
```

## 11. 项目使用

### 11.1 Smoke Test

```bash
python tools/train.py --config configs/acds_detr_smoke_visdrone_mini.yaml
```

### 11.2 核心消融

```bash
python tools/train.py --config configs/exp_baseline_stable.yaml --gpu 0
python tools/train.py --config configs/exp_acq_only_stable.yaml --gpu 0
python tools/train.py --config configs/exp_rsnds_only_stable.yaml --gpu 0
python tools/train.py --config configs/exp_acds_full_stable.yaml --gpu 0
```

### 11.3 论文增强配置

```bash
python tools/train.py --config configs/paper_baseline_original.yaml --gpu 0
python tools/train.py --config configs/paper_ablation_highres.yaml --gpu 0
python tools/train.py --config configs/paper_ablation_p2_p3.yaml --gpu 0
python tools/train.py --config configs/paper_ablation_small_queries.yaml --gpu 0
python tools/train.py --config configs/paper_ablation_small_loss.yaml --gpu 0
python tools/train.py --config configs/paper_ablation_aug.yaml --gpu 0
python tools/train.py --config configs/paper_full_small_object.yaml --gpu 0
```

### 11.4 评估

```bash
python tools/eval.py --config configs/exp_acds_full_stable.yaml --checkpoint outputs/exp_acds_full_stable/last.pth --gpu 0
```

### 11.5 可视化与诊断

```bash
python tools/export_predictions.py --config configs/paper_full_small_object.yaml --checkpoint outputs/paper_full_small_object/best_ap_small.pth --output outputs/paper_full_small_object/val_predictions.json --gpu 0
python tools/sod_debug.py stats --root /data/libaichuan/Projects/SOD/Datasets/VisDrone --split train
python tools/sod_debug.py vis-ann --root /data/libaichuan/Projects/SOD/Datasets/VisDrone --split val --small-only --output-dir outputs/debug_gt_small
python tools/sod_debug.py vis-pred --root /data/libaichuan/Projects/SOD/Datasets/VisDrone --split val --predictions outputs/paper_full_small_object/val_predictions.json --output-dir outputs/debug_pred
python tools/sod_debug.py fn --root /data/libaichuan/Projects/SOD/Datasets/VisDrone --split val --predictions outputs/paper_full_small_object/val_predictions.json --score-thresh 0.03 --iou-thr 0.5
```

## 12. 最终写作建议

ACDS-DETR 的论文叙事应围绕一句话展开：

```text
Dense small-object detection in Deformable DETR is limited not only by feature resolution,
but also by decoder-side query assignment collision and scale-insensitive deformable sampling.
```

中文可以写成：

```text
密集小目标检测的瓶颈不只是特征分辨率不足，还包括 decoder 侧 query 分配碰撞和尺度不敏感采样偏移。
```

只要实验能证明：

- ACQ 降低 query collision rate；
- R-SNDS 让 sampling points 更集中于小目标区域；
- `AP_small` / `AR_small` / dense subset 指标提升明显；
- 计算开销较小；
- 对普通 query repulsion 和普通 scale-aware sampling 有优势；

那么该方法具备较完整的论文级 idea 形态。

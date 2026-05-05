# ACDS-DETR: Assignment-aware Collision Decoupling and Reliability-guided Scale-normalized Sampling for Small Object Detection

## 1. Method Name

**ACDS-DETR** is short for **Assignment-aware Collision Decoupling and Reliability-guided Scale-normalized Sampling for Small Object Detection**.

中文名称可写为：

**面向小目标检测的分配感知查询解耦与可靠性引导尺度采样 Deformable DETR**

本文以 **Deformable DETR** 为基础模型，针对密集小目标检测中 decoder 侧的两个机制性失败模式进行改进：

1. **Query assignment collapse**：密集小目标场景下，多个 object queries 容易竞争同一目标或同一局部区域，导致相邻小目标缺少独立 query 负责。
2. **Sampling deviation**：Deformable attention 的 sampling offsets 对目标尺度和预测可靠性不敏感，小目标 query 的采样点容易偏离真实目标区域。

与简单的特征增强或多尺度融合不同，ACDS-DETR 直接改造 Deformable DETR 的 **query assignment / query collision 机制** 和 **deformable attention sampling 机制**。该方法不增加 query 数量，不增加采样点数量，也不引入复杂 neck，目标是在较低计算开销下提升 `AP_small` 和密集小目标召回率。

---

## 2. Contributions

本文的主要贡献可以概括为以下三点：

1. **提出 Assignment-aware Collision Query Decoupling, ACQ。**  
   针对密集小目标中多个 queries 竞争同一目标的问题，本文利用 Hungarian assignment 和小目标邻域关系建模 query collision，仅对存在分配冲突或冗余响应的 queries 施加解耦约束，从而提升小目标 query 的有效利用率。

2. **提出 Reliability-guided Scale-Normalized Deformable Sampling, R-SNDS。**  
   针对小目标 attention 采样点易偏离的问题，本文将预测框尺度反馈到 deformable sampling radius 中，并引入预测可靠性调制，避免 decoder 早期低质量 box 误导采样范围。

3. **提出面向小目标的 decoder-side mechanism refinement。**  
   ACDS-DETR 不依赖额外特征金字塔或重型注意力增强，而是从 query 分配和 sampling 位置生成两个机制层面修正 Deformable DETR 的小目标失败模式，具有清晰的因果解释和可视化验证路径。

---

## 3. Review-risk-driven Motivation

原始方案 CDS-DETR 包含 CDQ 和 SNDS 两个模块，但从论文审稿角度看仍存在两个风险：

1. **CDQ 容易被认为是普通 query regularization。**  
   如果简单惩罚所有距离过近的小目标 queries，审稿人可能认为它只是 heuristic repulsion loss，并且可能误伤真实相邻小目标。

2. **SNDS 容易被认为是手工 scale-aware sampling。**  
   如果直接使用预测框面积控制采样半径，decoder 早期不稳定的 box 预测可能导致采样半径错误收缩或扩张。

因此，ACDS-DETR 对原始方案进行两点升级：

```text
CDQ  -> ACQ:
从简单空间排斥升级为 assignment-aware collision modeling。

SNDS -> R-SNDS:
从直接尺度归一化升级为 reliability-guided scale-normalized sampling。
```

该升级使方法从“工程式模块组合”转向“针对 DETR 小目标失败机制的显式建模”。

---

## 4. Overall Framework

给定输入图像，ACDS-DETR 首先通过 backbone 提取多尺度特征，并将不同尺度特征投影到统一通道维度。随后，多尺度特征被输入 transformer encoder，得到上下文增强后的 memory features。Decoder 使用一组 object queries 通过 multi-scale deformable cross-attention 从 encoder memory 中聚合目标特征，并逐层更新 reference points 和 bounding boxes。

ACDS-DETR 的改动集中在 decoder 侧：

```text
Input image
    ↓
Backbone
    ↓
Multi-scale feature projection
    ↓
Transformer encoder
    ↓
Decoder object queries
    ↓
[ACQ] Assignment-aware Collision Query Decoupling
    ↓
[R-SNDS] Reliability-guided Scale-Normalized Deformable Sampling
    ↓
Classification head + box regression head
    ↓
Detection results
```

其中，ACQ 主要作用于训练阶段的 query 分工约束；R-SNDS 作用于 decoder cross-attention 中 sampling point 的位置生成过程。

---

## 5. Assignment-aware Collision Query Decoupling, ACQ

### 5.1 Motivation

Deformable DETR 使用一组 object queries 进行目标解码，并通过 Hungarian matching 建立 query 与 ground-truth objects 的一对一匹配。该机制能够避免传统 dense detector 中的 NMS 依赖，但在密集小目标场景中存在 query assignment collapse。

小目标通常尺寸小、距离近、外观相似。在 decoder 早期，多个 queries 的 reference points 和预测框可能聚集到同一小目标附近。由于 Hungarian matching 只能为一个 ground truth 分配一个正样本 query，其他高响应 queries 会变成 unmatched 或 low-quality matched queries。它们仍然可能占用局部响应区域，从而导致相邻小目标缺少有效 query 覆盖。

因此，小目标漏检不只是 query 数量不足，而是 query 利用率不足：

```text
Dense small objects
    ↓
Several queries respond to the same local object
    ↓
Nearby objects receive insufficient query coverage
    ↓
Small-object recall decreases
```

### 5.2 Mechanism Changed

ACQ 改变的是 **DETR decoder 中 query assignment 后的 query collision 约束机制**。

原始 Deformable DETR：

```text
Hungarian matching
    ↓
One-to-one detection loss
    ↓
Implicit query specialization
```

ACDS-DETR：

```text
Hungarian matching
    ↓
One-to-one detection loss
    +
Assignment-aware collision decoupling
    ↓
More effective query specialization for dense small objects
```

与简单 query repulsion 不同，ACQ 不惩罚所有距离接近的 queries，而是利用 Hungarian assignment 和预测质量判断哪些 queries 属于冗余竞争响应。

### 5.3 Collision Modeling

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

Hungarian matching 后，若 query `j` 匹配到小目标 ground truth，且 query `i` 未匹配或匹配质量较低，同时 `i` 与 `j` 在空间上高度接近，则认为 `i` 可能是 `j` 的冗余竞争 query。

定义小目标权重：

```text
m_j = exp(-A_j / tau_s)
```

其中 `A_j` 是 query `j` 对应目标或预测框的面积，`tau_s` 是小目标尺度阈值。

定义 query collision score：

```text
C_ij = s_i s_j exp(-||r_i - r_j||_2 / sigma) m_j
```

其中 `sigma` 控制 reference point 距离的衰减速度。两个高置信 queries 越接近，且目标越小，collision score 越高。

ACQ 的解耦损失为：

```text
L_acq = sum_(i,j in Omega) C_ij max(0, delta - ||r_i - r_j||_2)
```

其中 `Omega` 是 assignment-aware collision pair set，只包含满足以下条件的 query pair：

1. `j` 是 matched small-object query；
2. `i` 是 unmatched query 或 low-quality matched query；
3. `IoU(b_i, b_j)` 或 reference point 距离表明二者存在局部竞争；
4. `s_i` 和 `s_j` 高于置信度阈值。

最终训练损失为：

```text
L = L_det + lambda_acq L_acq
```

其中 `L_det` 是 Deformable DETR 原有的分类损失、L1 box loss 和 GIoU loss。

### 5.4 Why ACQ Is Better Than Simple Query Repulsion

简单 query repulsion 的风险是误伤真实相邻小目标，因为相邻小目标本来就可能拥有非常接近的 reference points。

ACQ 通过 Hungarian assignment 限定约束对象：

```text
Only redundant or conflicting query responses are decoupled.
Legitimate neighboring object queries are preserved.
```

因此，ACQ 更符合 DETR 的一对一分配机制，也更容易通过实验说明其作用不是“让所有 query 分散”，而是“减少无效 query 竞争”。

### 5.5 Why It Works for Small Objects

ACQ 对小目标有效的因果链为：

```text
Dense small objects
    ↓
Assignment-aware collision pairs are identified
    ↓
Redundant queries around the same small object are decoupled
    ↓
Neighboring small objects receive more independent query responses
    ↓
AR_small and AP_small improve
```

该模块尤其适合 VisDrone、AI-TOD 等目标密集、尺寸较小的检测场景。

---

## 6. Reliability-guided Scale-Normalized Deformable Sampling, R-SNDS

### 6.1 Motivation

Multi-scale deformable attention 是 Deformable DETR 的核心机制。对于每个 query，模型不再对整张 feature map 执行 dense attention，而是在 reference point 附近预测少量 sampling offsets，从多尺度特征中聚合局部信息。

该机制计算高效，但对小目标存在 sampling deviation 问题：

```text
Small objects occupy limited spatial extent
    ↓
Unconstrained sampling offsets may fall on background or nearby objects
    ↓
Attention features become contaminated
    ↓
Classification confidence and box localization degrade
```

直接使用预测框尺度控制 sampling radius 是一个自然选择，但 decoder 早期预测框并不可靠。如果早期 box 预测过小，采样半径可能过度收缩；如果预测过大，采样仍可能被背景污染。因此，R-SNDS 引入 reliability gate，使采样半径在预测可靠时才更多依赖预测尺度。

### 6.2 Mechanism Changed

R-SNDS 改变的是 **deformable attention 中 sampling points 的位置生成机制**。

原始 Deformable DETR：

```text
p_{i,l,k} = r_i + Delta p_{i,l,k}
```

ACDS-DETR：

```text
p_{i,l,k} = r_i + gamma_i Delta p_{i,l,k}
```

其中 `gamma_i` 由预测尺度和预测可靠性共同决定。

### 6.3 Scale-normalized Sampling with Reliability Gate

设 `r_i` 表示第 `i` 个 query 的 reference point，`Delta p_{i,l,k}` 表示第 `l` 个特征层、第 `k` 个采样点的预测偏移。

首先，根据当前预测框尺度得到尺度归一化因子：

```text
gamma_i^scale = clip(beta sqrt(w_i h_i), gamma_min, gamma_max)
```

其中 `w_i` 和 `h_i` 来自当前 decoder layer 的预测框，`beta` 是缩放系数，`gamma_min` 和 `gamma_max` 限制采样半径上下界。

然后，定义预测可靠性：

```text
rho_i = sigmoid(MLP(q_i))
```

或使用分类置信度近似：

```text
rho_i = max_c p_i(c)
```

最终采样半径为：

```text
gamma_i = (1 - rho_i) gamma_base + rho_i gamma_i^scale
```

新的采样位置为：

```text
p_{i,l,k} = r_i + gamma_i Delta p_{i,l,k}
```

其中 `gamma_base` 是默认采样半径。当预测不可靠时，模型更多使用保守默认采样半径；当预测可靠时，模型更多采用尺度归一化采样半径。

### 6.4 Difference from Deformable DETR

原始 Deformable DETR 的 sampling offsets 由 query 自由预测，并没有显式考虑目标尺度和预测可靠性。

R-SNDS 的核心差异是：

```text
Sampling offsets are not only query-dependent,
but also normalized by object scale and gated by prediction reliability.
```

这使小目标 query 的 sampling points 更集中，同时避免 decoder 早期错误预测导致采样范围异常。

### 6.5 Why It Works for Small Objects

R-SNDS 对小目标有效的因果链为：

```text
Small object has small spatial extent
    ↓
Reliable small-object prediction leads to smaller sampling radius
    ↓
Sampling points concentrate around the object region
    ↓
Background and neighboring-object sampling is reduced
    ↓
Attention feature purity improves
    ↓
AP_small improves
```

相比普通 scale-aware sampling，R-SNDS 额外考虑 decoder 预测可靠性，因此更稳定，也更适合逐层 refinement 的 DETR decoder。

---

## 7. Difference from Deformable DETR

| Component | Deformable DETR | ACDS-DETR |
| --- | --- | --- |
| Query assignment | Relies on Hungarian matching for implicit query specialization | Adds assignment-aware collision decoupling for dense small objects |
| Query regularization | None | Only decouples redundant or conflicting query responses |
| Sampling offsets | Freely predicted offsets | Scale-normalized offsets gated by prediction reliability |
| Query number | Fixed | Unchanged |
| Sampling points | Fixed sparse points | Unchanged number, adjusted spatial distribution |
| Extra feature neck | Not required | Not introduced |
| Main target | General object detection | Small and dense object detection |

---

## 8. Computational Complexity

ACDS-DETR is lightweight for the following reasons:

1. **No extra decoder layers.**  
   The number of encoder and decoder layers is kept unchanged.

2. **No extra object queries.**  
   The number of object queries remains the same as the baseline, for example 300 queries.

3. **No extra sampling points.**  
   R-SNDS rescales existing offsets and does not introduce additional attention sampling points.

4. **Almost no additional inference parameters.**  
   ACQ is a training loss and introduces no inference overhead. R-SNDS only introduces a lightweight reliability gate, which can be implemented as a small MLP or approximated by classification confidence.

Expected complexity change:

```text
Params: +0M to +0.2M
FLOPs:  +1% to +4%
FPS:    decrease < 5%
```

During inference, ACQ is removed because it is only a training regularization term. The main inference overhead comes from the reliability gate and scalar multiplication in R-SNDS, which is negligible compared with transformer attention computation.

---

## 9. Experiments Needed for Reviewer Convincing

The most important experiments are not only AP comparisons, but mechanism verification experiments.

### 9.1 Query Collision Rate

Define a query collision rate to measure redundant query responses around small objects:

```text
QCR = N_collision / N_small
```

where `N_collision` denotes the number of redundant high-confidence queries located around matched small objects, and `N_small` denotes the number of small objects.

Expected observation:

```text
Deformable DETR: high QCR
+ ACQ: lower QCR
```

This experiment directly supports the claim that ACQ reduces query collision.

### 9.2 Small Object Recall

Because ACQ mainly targets missed detections, the following metrics should be emphasized:

```text
AR_small
Recall@100_small
Recall@300_small
```

Expected observation:

```text
+ ACQ improves AR_small more significantly than AP_large.
```

### 9.3 Sampling Point Visualization

Visualize decoder sampling points before and after R-SNDS:

```text
Baseline:
sampling points often fall on background or neighboring objects.

R-SNDS:
sampling points become more concentrated around small object regions.
```

This visualization is critical for proving that R-SNDS changes the attention mechanism rather than merely adding a heuristic scalar.

### 9.4 Dense-region Subset Evaluation

Construct a dense small-object subset from VisDrone or AI-TOD:

```text
images with more than N small objects
or regions where average object distance is below a threshold
```

Expected observation:

```text
ACDS-DETR gains are larger on dense small-object subsets.
```

This directly supports the motivation that the method targets dense small-object detection rather than general AP tuning.

---

## 10. Experimental Design

### 10.1 Datasets

Recommended datasets:

- **COCO 2017**: general object detection benchmark, used to verify overall AP and AP_small.
- **VisDrone2019**: dense aerial object detection dataset with many small objects.
- **AI-TOD**: tiny object detection benchmark, suitable for validating extremely small object detection.

### 10.2 Baselines

Recommended comparison methods:

- Deformable DETR
- DINO
- DN-DETR
- DAB-DETR
- Faster R-CNN
- RetinaNet

Minimum necessary comparison:

- Deformable DETR baseline
- DINO
- ACDS-DETR

### 10.3 Metrics

Main metrics:

- AP
- AP50
- AP75
- AP_small
- AP_medium
- AP_large
- AR_small
- Params
- FLOPs
- FPS
- Query Collision Rate

### 10.4 Ablation Study

| Experiment | ACQ | R-SNDS | Purpose |
| --- | --- | --- | --- |
| A0 | No | No | Baseline |
| A1 | Yes | No | Verify assignment-aware query decoupling |
| A2 | No | Yes | Verify reliability-guided scale-normalized sampling |
| A3 | Yes | Yes | Verify full ACDS-DETR |

Additional sensitivity studies:

- `lambda_acq`: weight of assignment-aware collision loss.
- `delta`: minimum distance margin between collision queries.
- `tau_s`: small-object scale threshold.
- `sigma`: distance decay factor in collision score.
- `gamma_min` and `gamma_max`: lower and upper bounds of sampling radius.
- `gamma_base`: default sampling radius when prediction reliability is low.
- `rho_i`: compare MLP-based reliability gate and classification-confidence reliability gate.
- Apply R-SNDS to all decoder layers vs. only later decoder layers.

---

## 11. Expected Results

On COCO 2017 with a ResNet-50 Deformable DETR baseline, expected improvements are:

```text
AP_small: +2.0 to +3.5
AP:       +0.8 to +2.0
FLOPs:    +1% to +4%
FPS:      decrease < 5%
```

On small and dense object datasets such as VisDrone2019 and AI-TOD, expected improvements may be larger:

```text
AP_small or AP_tiny: +3.0 to +5.0
AR_small:            +3.0 to +6.0
AP:                  +1.0 to +3.0
Computational cost:   < +5%
```

---

## 12. Suggested Paper Writing Focus

The paper should avoid describing ACDS-DETR as a generic feature enhancement framework. The method should be positioned around two decoder-side mechanism failures:

```text
Query assignment collapse in dense small-object regions
Sampling deviation in deformable attention
```

Recommended title:

```text
ACDS-DETR: Assignment-aware Collision Decoupling and Reliability-guided Scale-normalized Sampling for Small Object Detection
```

Recommended Chinese title:

```text
面向小目标检测的分配感知查询解耦与可靠性引导尺度采样 Deformable DETR
```

The strongest experimental visualizations are:

1. Query collision rate comparison before and after ACQ.
2. Reference point distribution before and after ACQ.
3. Sampling point visualization before and after R-SNDS.
4. Dense small-object subset evaluation.
5. AP_small and AR_small ablation curves.

---

## 13. Project Implementation

This repository contains a complete PyTorch implementation scaffold for ACDS-DETR, including VisDrone data loading, model definition, ACQ loss, R-SNDS sampling, training, evaluation, inference, and smoke-test configuration.

### 13.1 Directory Structure

```text
ACDS-DETR/
├── configs/              # YAML experiment configs
├── datasets/             # VisDrone dataset, transforms, collate function
├── models/               # Backbone, transformer, deformable attention, ACQ/R-SNDS-related model code
├── losses/               # DETR losses and ACQ loss
├── engine/               # train/eval/inference loops
├── utils/                # box ops, metrics, checkpoint, distributed helpers
├── tools/                # train.py, eval.py, infer.py
└── outputs/              # checkpoints and logs
```

### 13.2 Smoke Test

Use the mini VisDrone config to verify the full pipeline on CPU or a small GPU:

```bash
python tools/train.py --config configs/acds_detr_smoke_visdrone_mini.yaml
```

This runs 1 epoch with 2 samples and prints training loss plus validation metrics.

### 13.3 Single-GPU Training

Use GPU 0:

```bash
python tools/train.py --config configs/acds_detr_r50_visdrone.yaml --gpu 0
```

Use GPU 1:

```bash
python tools/train.py --config configs/acds_detr_r50_visdrone.yaml --gpu 1
```

### 13.4 Multi-GPU Training

Use `torchrun` for distributed training:

```bash
torchrun --nproc_per_node=2 tools/train.py --config configs/acds_detr_r50_visdrone.yaml
```

### 13.5 Evaluation

```bash
python tools/eval.py --config configs/acds_detr_r50_visdrone.yaml --checkpoint outputs/acds_detr/last.pth --gpu 0
```

Each validation prints:

```text
loss, loss_ce, loss_bbox, loss_giou, loss_acq,
mAP, AP_small, mAP50_95, AP50, precision, recall, FPS
```

### 13.6 Notes

The current deformable attention implementation is pure PyTorch for portability and readability. It is suitable for method verification and thesis experiments, but slower than the official CUDA `MSDeformAttn` operator. For large-scale final training, replacing `models/deformable_attention.py` with the CUDA operator is recommended while preserving the same R-SNDS interface.

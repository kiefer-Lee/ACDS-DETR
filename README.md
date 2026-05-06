# ACDS-DETR：面向小目标检测的分配感知查询解耦与可靠性引导尺度采样

## 1. 方法名称

**ACDS-DETR** 表示 **Assignment-aware Collision Decoupling and Reliability-guided Scale-normalized Sampling**，中文可写为：

**面向小目标检测的分配感知查询解耦与可靠性引导尺度采样 Deformable DETR**

本文以 **Deformable DETR** 为基础模型，针对密集小目标检测中 decoder 侧的两个机制性失败模式进行改进：

1. **查询分配塌缩**：密集小目标场景下，多个 object queries 容易竞争同一目标或同一局部区域，导致相邻小目标缺少独立 query 负责。
2. **采样偏移**：Deformable attention 的 sampling offsets 对目标尺度和预测可靠性不敏感，小目标 query 的采样点容易偏离真实目标区域。

与简单的特征增强或多尺度融合不同，ACDS-DETR 直接改造 Deformable DETR 的 **query assignment / query collision 机制** 和 **deformable attention sampling 机制**。该方法不增加 query 数量，不增加采样点数量，也不引入复杂 neck，目标是在较低计算开销下提升 `AP_small` 和密集小目标召回率。

---

## 2. 主要贡献

本文的主要贡献可以概括为以下三点：

1. **提出分配感知碰撞查询解耦 ACQ。**  
   针对密集小目标中多个 queries 竞争同一目标的问题，本文利用 Hungarian assignment 和小目标邻域关系建模 query collision，仅对存在分配冲突或冗余响应的 queries 施加解耦约束，从而提升小目标 query 的有效利用率。

2. **提出可靠性引导尺度归一化可变形采样 R-SNDS。**  
   针对小目标 attention 采样点易偏离的问题，本文将预测框尺度反馈到 deformable sampling radius 中，并引入预测可靠性调制，避免 decoder 早期低质量 box 误导采样范围。

3. **提出面向小目标的 decoder 侧机制修正。**  
   ACDS-DETR 不依赖额外特征金字塔或重型注意力增强，而是从 query 分配和 sampling 位置生成两个机制层面修正 Deformable DETR 的小目标失败模式，具有清晰的因果解释和可视化验证路径。

---

## 3. 审稿风险驱动的动机

原始方案 CDS-DETR 包含 CDQ 和 SNDS 两个模块，但从论文审稿角度看仍存在两个风险：

1. **CDQ 容易被认为是普通 query regularization。**  
   如果简单惩罚所有距离过近的小目标 queries，审稿人可能认为它只是启发式排斥损失，并且可能误伤真实相邻小目标。

2. **SNDS 容易被认为是手工 scale-aware sampling。**  
   如果直接使用预测框面积控制采样半径，decoder 早期不稳定的 box 预测可能导致采样半径错误收缩或扩张。

因此，ACDS-DETR 对原始方案进行两点升级：

```text
CDQ  -> ACQ：
从简单空间排斥升级为分配感知碰撞建模。

SNDS -> R-SNDS：
从直接尺度归一化升级为可靠性引导尺度归一化采样。
```

该升级使方法从“工程式模块组合”转向“针对 DETR 小目标失败机制的显式建模”。

---

## 4. 整体框架

给定输入图像，ACDS-DETR 首先通过 backbone 提取多尺度特征，并将不同尺度特征投影到统一通道维度。随后，多尺度特征被输入 transformer encoder，得到上下文增强后的 memory features。Decoder 使用一组 object queries 通过 multi-scale deformable cross-attention 从 encoder memory 中聚合目标特征，并逐层更新 reference points 和 bounding boxes。

ACDS-DETR 的改动集中在 decoder 侧：

```text
输入图像
    ↓
Backbone 多尺度特征提取
    ↓
多尺度特征投影
    ↓
Transformer encoder
    ↓
Decoder object queries
    ↓
[ACQ] 分配感知碰撞查询解耦
    ↓
[R-SNDS] 可靠性引导尺度归一化可变形采样
    ↓
分类头与边框回归头
    ↓
检测结果
```

其中，ACQ 主要作用于训练阶段的 query 分工约束；R-SNDS 作用于 decoder cross-attention 中 sampling point 的位置生成过程。

---

## 5. 分配感知碰撞查询解耦 ACQ

### 5.1 动机

Deformable DETR 使用一组 object queries 进行目标解码，并通过 Hungarian matching 建立 query 与 ground-truth objects 的一对一匹配。该机制能够避免传统 dense detector 中的 NMS 依赖，但在密集小目标场景中存在 query assignment collapse。

小目标通常尺寸小、距离近、外观相似。在 decoder 早期，多个 queries 的 reference points 和预测框可能聚集到同一小目标附近。由于 Hungarian matching 只能为一个 ground truth 分配一个正样本 query，其他高响应 queries 会变成 unmatched 或 low-quality matched queries。它们仍然可能占用局部响应区域，从而导致相邻小目标缺少有效 query 覆盖。

因此，小目标漏检不只是 query 数量不足，而是 query 利用率不足：

```text
密集小目标
    ↓
多个 queries 响应同一个局部目标
    ↓
相邻目标缺少足够 query 覆盖
    ↓
小目标召回率下降
```

### 5.2 机制变化

ACQ 改变的是 **DETR decoder 中 query assignment 后的 query collision 约束机制**。

原始 Deformable DETR：

```text
Hungarian matching
    ↓
一对一检测损失
    ↓
隐式 query 专门化
```

ACDS-DETR：

```text
Hungarian matching
    ↓
一对一检测损失
    +
分配感知碰撞解耦
    ↓
密集小目标场景下更有效的 query 专门化
```

与简单 query repulsion 不同，ACQ 不惩罚所有距离接近的 queries，而是利用 Hungarian assignment 和预测质量判断哪些 queries 属于冗余竞争响应。

### 5.3 碰撞建模

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

其中 `Omega` 是分配感知碰撞 query pair 集合，只包含满足以下条件的 query pair：

1. `j` 是 matched small-object query；
2. `i` 是 unmatched query 或 low-quality matched query；
3. `IoU(b_i, b_j)` 或 reference point 距离表明二者存在局部竞争；
4. `s_i` 和 `s_j` 高于置信度阈值。

最终训练损失为：

```text
L = L_det + lambda_acq L_acq
```

其中 `L_det` 是 Deformable DETR 原有的分类损失、L1 box loss 和 GIoU loss。

### 5.4 ACQ 相比简单 query repulsion 的优势

简单 query repulsion 的风险是误伤真实相邻小目标，因为相邻小目标本来就可能拥有非常接近的 reference points。

ACQ 通过 Hungarian assignment 限定约束对象：

```text
只解耦冗余或冲突 query 响应。
保留真实相邻目标对应的合法 query。
```

因此，ACQ 更符合 DETR 的一对一分配机制，也更容易通过实验说明其作用不是“让所有 query 分散”，而是“减少无效 query 竞争”。

### 5.5 为什么适合小目标

ACQ 对小目标有效的因果链为：

```text
密集小目标
    ↓
识别分配感知碰撞 query pair
    ↓
解耦同一小目标附近的冗余 queries
    ↓
相邻小目标获得更独立的 query 响应
    ↓
AR_small 和 AP_small 提升
```

该模块尤其适合 VisDrone、AI-TOD 等目标密集、尺寸较小的检测场景。

---

## 6. 可靠性引导尺度归一化可变形采样 R-SNDS

### 6.1 动机

Multi-scale deformable attention 是 Deformable DETR 的核心机制。对于每个 query，模型不再对整张 feature map 执行 dense attention，而是在 reference point 附近预测少量 sampling offsets，从多尺度特征中聚合局部信息。

该机制计算高效，但对小目标存在 sampling deviation 问题：

```text
小目标空间范围有限
    ↓
不受约束的采样偏移可能落到背景或邻近目标上
    ↓
Attention 特征被污染
    ↓
分类置信度和边框定位质量下降
```

直接使用预测框尺度控制 sampling radius 是一个自然选择，但 decoder 早期预测框并不可靠。如果早期 box 预测过小，采样半径可能过度收缩；如果预测过大，采样仍可能被背景污染。因此，R-SNDS 引入 reliability gate，使采样半径在预测可靠时才更多依赖预测尺度。

### 6.2 机制变化

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

### 6.3 带可靠性门控的尺度归一化采样

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

### 6.4 与 Deformable DETR 的差异

原始 Deformable DETR 的 sampling offsets 由 query 自由预测，并没有显式考虑目标尺度和预测可靠性。

R-SNDS 的核心差异是：

```text
采样偏移不仅由 query 决定，
还会根据目标尺度归一化，并由预测可靠性门控。
```

这使小目标 query 的 sampling points 更集中，同时避免 decoder 早期错误预测导致采样范围异常。

### 6.5 为什么适合小目标

R-SNDS 对小目标有效的因果链为：

```text
小目标空间范围有限
    ↓
可靠的小目标预测带来更小的采样半径
    ↓
采样点更集中在目标区域附近
    ↓
减少背景和邻近目标采样
    ↓
Attention 特征纯度提升
    ↓
AP_small 提升
```

相比普通 scale-aware sampling，R-SNDS 额外考虑 decoder 预测可靠性，因此更稳定，也更适合逐层 refinement 的 DETR decoder。

---

## 7. 与 Deformable DETR 的差异

| 组件 | Deformable DETR | ACDS-DETR |
| --- | --- | --- |
| Query 分配 | 依赖 Hungarian matching 实现隐式 query 专门化 | 增加面向密集小目标的分配感知碰撞解耦 |
| Query 正则 | 无 | 仅解耦冗余或冲突 query 响应 |
| 采样偏移 | 自由预测 offsets | 由预测可靠性门控的尺度归一化 offsets |
| Query 数量 | 固定 | 保持不变 |
| 采样点数量 | 固定稀疏采样点 | 数量不变，仅调整空间分布 |
| 额外特征 neck | 不需要 | 不引入 |
| 主要目标 | 通用目标检测 | 小目标与密集目标检测 |

---

## 8. 计算复杂度

ACDS-DETR 保持轻量，原因如下：

1. **不增加 decoder 层数。**  
   Encoder 和 decoder 层数保持不变。

2. **不增加 object queries 数量。**  
   Object queries 数量与基线模型相同，例如 300 queries。

3. **不增加采样点数量。**  
   R-SNDS 只缩放已有 offsets，不引入额外 attention sampling points。

4. **几乎不增加推理参数量。**  
   ACQ 是训练损失，不增加推理开销。R-SNDS 只引入轻量 reliability gate，可由小型 MLP 实现，也可用分类置信度近似。

预期复杂度变化：

```text
Params: +0M 至 +0.2M
FLOPs:  +1% 至 +4%
FPS:    下降 < 5%
```

推理阶段移除 ACQ，因为它只是训练正则项。主要推理开销来自 R-SNDS 的 reliability gate 和标量乘法，相比 transformer attention 计算可以忽略。

---

## 9. 支撑审稿说服力的实验

最重要的实验不仅是 AP 对比，还包括机制验证实验。

### 9.1 Query Collision Rate

定义 query collision rate，用于衡量小目标周围的冗余 query 响应：

```text
QCR = N_collision / N_small
```

其中 `N_collision` 表示匹配小目标附近的冗余高置信 query 数量，`N_small` 表示小目标数量。

预期现象：

```text
Deformable DETR：QCR 较高
+ ACQ：QCR 降低
```

该实验直接支撑“ACQ 能减少 query collision”的论点。

### 9.2 小目标召回率

由于 ACQ 主要面向漏检问题，建议重点报告：

```text
AR_small
Recall@100_small
Recall@300_small
```

预期现象：

```text
+ ACQ 对 AR_small 的提升应明显强于对 AP_large 的提升。
```

### 9.3 采样点可视化

可视化 R-SNDS 前后的 decoder sampling points：

```text
基线模型：
采样点容易落在背景或邻近目标上。

R-SNDS：
采样点更集中在小目标区域附近。
```

该可视化对于证明 R-SNDS 改变 attention 机制非常关键，而不是仅增加一个启发式标量。

### 9.4 密集区域子集评估

从 VisDrone 或 AI-TOD 构造密集小目标子集：

```text
包含超过 N 个小目标的图像，
或平均目标间距低于阈值的区域。
```

预期现象：

```text
ACDS-DETR 在密集小目标子集上的收益更明显。
```

这能直接支撑该方法面向密集小目标检测，而不是普通 AP 调参。

---

## 10. 实验设计

### 10.1 数据集

推荐数据集：

- **COCO 2017**：通用目标检测 benchmark，用于验证整体 AP 和 `AP_small`。
- **VisDrone2019**：密集航拍目标检测数据集，包含大量小目标。
- **AI-TOD**：tiny object detection benchmark，适合验证极小目标检测。

### 10.2 对比方法

推荐对比方法：

- Deformable DETR
- DINO
- DN-DETR
- DAB-DETR
- Faster R-CNN
- RetinaNet

最低必要对比：

- Deformable DETR 基线模型
- DINO
- ACDS-DETR

### 10.3 评价指标

主要指标：

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

### 10.4 消融实验

| 实验 | ACQ | R-SNDS | 目的 |
| --- | --- | --- | --- |
| A0 | 否 | 否 | 基线模型 |
| A1 | 是 | 否 | 验证分配感知查询解耦 |
| A2 | 否 | 是 | 验证可靠性引导尺度归一化采样 |
| A3 | 是 | 是 | 验证完整 ACDS-DETR |

额外敏感性实验：

- `lambda_acq`：分配感知碰撞损失权重。
- `delta`：碰撞 query 之间的最小距离 margin。
- `tau_s`：小目标尺度阈值。
- `sigma`：collision score 中的距离衰减系数。
- `gamma_min` 和 `gamma_max`：采样半径上下界。
- `gamma_base`：预测可靠性较低时的默认采样半径。
- `rho_i`：比较 MLP reliability gate 与分类置信度 reliability gate。
- 比较 R-SNDS 作用于所有 decoder layers 与仅作用于后几层 decoder layers。

---

## 11. 预期结果

在 COCO 2017 的 ResNet-50 Deformable DETR 基线模型上，预期提升为：

```text
AP_small: +2.0 至 +3.5
AP:       +0.8 至 +2.0
FLOPs:    +1% 至 +4%
FPS:      下降 < 5%
```

在 VisDrone2019 和 AI-TOD 这类小目标、密集目标数据集上，预期提升可能更明显：

```text
AP_small 或 AP_tiny: +3.0 至 +5.0
AR_small:            +3.0 至 +6.0
AP:                  +1.0 至 +3.0
计算开销:             < +5%
```

---

## 12. 论文写作重点

论文应避免将 ACDS-DETR 描述为通用特征增强框架，而应围绕 decoder 侧两个机制性失败展开：

```text
密集小目标区域中的 query assignment collapse
deformable attention 中的 sampling deviation
```

推荐英文题目：

```text
ACDS-DETR: Assignment-aware Collision Decoupling and Reliability-guided Scale-normalized Sampling for Small Object Detection
```

推荐中文题目：

```text
面向小目标检测的分配感知查询解耦与可靠性引导尺度采样 Deformable DETR
```

最有说服力的实验可视化包括：

1. ACQ 前后的 query collision rate 对比。
2. ACQ 前后的 reference point 分布。
3. R-SNDS 前后的 sampling point 可视化。
4. 密集小目标子集评估。
5. `AP_small` 和 `AR_small` 消融曲线。

---

## 13. 项目实现

本仓库包含 ACDS-DETR 的完整 PyTorch 实现骨架，包括 VisDrone 数据加载、模型定义、ACQ loss、R-SNDS sampling、训练、评估、推理和 smoke test 配置。

### 13.1 目录结构

```text
ACDS-DETR/
├── configs/              # YAML 实验配置
├── datasets/             # VisDrone 数据集、数据增强、collate 函数
├── models/               # Backbone、transformer、deformable attention、ACQ/R-SNDS 相关模型代码
├── losses/               # DETR 损失与 ACQ 损失
├── engine/               # 训练、评估、推理流程
├── utils/                # box 操作、指标、checkpoint、分布式工具
├── tools/                # train.py、eval.py、infer.py
└── outputs/              # checkpoints 和日志
```

### 13.2 Smoke Test

使用 mini VisDrone 配置在 CPU 或小显存 GPU 上验证完整链路：

```bash
python tools/train.py --config configs/acds_detr_smoke_visdrone_mini.yaml
```

该命令会使用 2 个样本训练 1 个 epoch，并打印训练损失和验证指标。

### 13.3 稳定训练顺序

建议先运行稳定配置，再训练最终完整模型：

```bash
python tools/train.py --config configs/exp_baseline_stable.yaml --gpu 0
python tools/train.py --config configs/exp_rsnds_only_stable.yaml --gpu 0
python tools/train.py --config configs/exp_acq_only_stable.yaml --gpu 0
python tools/train.py --config configs/exp_acds_full_stable.yaml --gpu 0
python tools/train.py --config configs/exp_acds_full_final.yaml --gpu 0
```

稳定配置使用较低学习率、较小辅助损失权重、保守 R-SNDS 半径和 NaN 安全训练策略。

如果训练速度过慢，建议先使用快速配置完成方法验证：

```bash
python tools/train.py --config configs/exp_acds_full_fast.yaml --gpu 0
```

显存或时间更紧张时，可以使用 ResNet-18 快速配置：

```bash
python tools/train.py --config configs/exp_acds_full_fast_r18.yaml --gpu 0
```

快速配置会降低输入尺度、query 数量和 decoder/encoder 层数，并把验证频率改为每 5 个 epoch 一次。该配置适合调参和消融预筛选；最终论文主结果仍建议使用稳定配置或最终配置重新训练。

### 13.4 单卡训练

使用 GPU 0：

```bash
python tools/train.py --config configs/exp_acds_full_stable.yaml --gpu 0
```

使用 GPU 1：

```bash
python tools/train.py --config configs/exp_acds_full_stable.yaml --gpu 1
```

### 13.5 多卡训练

使用 `torchrun` 进行分布式训练：

```bash
torchrun --nproc_per_node=2 tools/train.py --config configs/exp_acds_full_stable.yaml
```

### 13.6 评估

```bash
python tools/eval.py --config configs/exp_acds_full_stable.yaml --checkpoint outputs/exp_acds_full_stable/last.pth --gpu 0
```

每次验证会打印：

```text
loss, loss_ce, loss_bbox, loss_giou, loss_acq,
mAP, AP_small, mAP50_95, AP50, AP75, AR_small, precision, recall, FPS
```

训练还会写入：

```text
config_resolved.yaml
train_log.jsonl
val_metrics.jsonl
metrics_summary.json
last.pth
best_map.pth
best_ap_small.pth
```

### 13.7 可视化与性能分析

Query/reference point 可视化：

```bash
python tools/visualize_queries.py --config configs/exp_acds_full_stable.yaml --checkpoint outputs/exp_acds_full_stable/best_ap_small.pth --gpu 0
```

R-SNDS sampling point 可视化：

```bash
python tools/visualize_sampling.py --config configs/exp_acds_full_stable.yaml --checkpoint outputs/exp_acds_full_stable/best_ap_small.pth --query 0 --gpu 0
```

模型性能分析：

```bash
python tools/profile_model.py --config configs/exp_acds_full_stable.yaml --gpu 0
```

将 VisDrone 标注转换为 COCO JSON：

```bash
python tools/visdrone_to_coco.py --root D:/PythonProjects/SOD/Datasets/VisDrone --split val --output outputs/visdrone_val_coco.json
```

### 13.8 说明

当前 deformable attention 使用纯 PyTorch 实现，优点是可移植、便于阅读，也方便检查 R-SNDS 的采样逻辑。该实现适合方法验证和毕业论文实验，但速度慢于官方 CUDA `MSDeformAttn` 算子。若后续进行大规模最终训练，建议在保持 R-SNDS 接口一致的前提下，将 `models/deformable_attention.py` 替换为 CUDA 算子实现。

---

## 14. 可复现实验流程

### 14.1 最小验证

代码或配置修改后，建议运行：

```bash
python tools/train.py --help
python tools/train.py --config configs/acds_detr_smoke_visdrone_mini.yaml --output-dir outputs/smoke_plan_verify
python tools/eval.py --config configs/acds_detr_smoke_visdrone_mini.yaml --checkpoint outputs/smoke_plan_verify/last.pth
```

如果有多张 GPU：

```bash
torchrun --nproc_per_node=2 tools/train.py --config configs/acds_detr_smoke_visdrone_mini.yaml --output-dir outputs/smoke_ddp_verify
```

### 14.2 命令行覆盖配置

使用 `--seed`、`--device` 和 `--opts` 可以减少 YAML 文件复制：

```bash
python tools/train.py --config configs/exp_acds_full_stable.yaml --seed 42 --gpu 0 --opts train.lr=5e-5 acq.lambda_acq=0.05
```

训练 checkpoint 会保存 resolved config、seed、git summary、命令行参数以及最佳 `mAP` / `AP_small`。每个输出目录中都会生成 `metrics_summary.json`，便于快速整理论文表格。

### 14.3 推荐消融表

| ID | 配置 | ACQ | R-SNDS | 主要指标 |
| --- | --- | --- | --- | --- |
| A0 | `exp_baseline_stable.yaml` | 关闭 | 关闭 | `mAP`, `AP_small`, `AR_small` |
| A1 | `exp_acq_only_stable.yaml` | 开启 | 关闭 | `AP_small`, query collision rate |
| A2 | `exp_rsnds_only_stable.yaml` | 关闭 | 开启 | `AP_small`, FPS |
| A3 | `exp_acds_full_stable.yaml` | 开启 | 开启 | 全部指标 |
| S1 | `ablation_acq_lambda_*.yaml` | 开启 | 开启 | lambda 敏感性 |
| S2 | `ablation_gamma_*.yaml` | 开启 | 开启 | sampling radius 敏感性 |
| S3 | `ablation_sigma_*.yaml` | 开启 | 开启 | collision distance 敏感性 |

只有在相同 seed、相同数据划分、相同训练轮数、相同图像尺度和相同评估阈值下，`AP_small` 与 `AR_small` 同时提升，才建议认为性能提升有效。论文中应同时报告 FPS 和 Params，说明方法收益对应的计算开销。

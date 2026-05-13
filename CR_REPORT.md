# ACDS-DETR 代码缺陷审查报告

## 严重缺陷（会导致训练结果错误或崩溃）

---

### 1. `criterion.py:78-85` — aux_loss 的 weight_dict key 与实际 loss_dict key 不对齐

`weight_dict` 在 `__init__` 中按 `start_layer`（基于 `aux_count - aux_loss_layers`）生成，但在 `forward` 中对 aux_outputs 的遍历也跳过了 `start_layer` 之前的层。问题是两处的 `i` 含义不同：

- `weight_dict` 中的 key 是 `loss_ce_{i}` 其中 `i ∈ [start_layer, aux_count)`
- `forward` 中的 loss_dict key 也是 `loss_ce_{i}`，`i` 是 enumerate 的全局 index

当 `start_layer > 0` 时，`weight_dict["loss_ce_0"]` 不存在，但 `loss_dict["loss_ce_0"]` 却被写入——这些 aux loss 完全不参与反向传播，模型退化为只用最后一层 decoder 输出监督。

**修复方向**：两处 `i` 应用同一套 key，最简单是让 `forward` 里生成的 key 也从 `start_layer` 开始偏移。

---

### 2. `acq_loss.py:38` — `small_thr_norm` 计算单位错误，导致几乎没有小目标被匹配

```python
small_thr_norm = self.small_area_thr / float(targets[b]["size"].prod().item())
```

`self.small_area_thr` 默认是绝对像素面积（如 1024），`targets[b]["size"]` 是 resize 之后的图像尺寸 `[h, w]`。但 `self.small_area_thr` 是配置中的原始像素阈值（比如 1024 = 32×32），而 resize 后小目标面积会按 `scale²` 缩放。如果图像从 1080p resize 到 800px，面积缩放约 0.55，原始 1024px² 的目标 resize 后变为约 562px²，`small_thr_norm` 用原始 1024 除以 resize 后的 `h*w`，比较的是 resize 后面积，但阈值基准是 resize 前的像素数——**单位不统一，small mask 匹配结果会系统偏移**。

---

### 3. `transformer.py:134-138` — encoder 每层更新 `encoder_srcs` 但传给 EncoderLayer 的仍是旧 srcs

```python
for layer in self.encoder_layers:
    memory = layer(memory, memory_pos, encoder_srcs, encoder_masks, encoder_reference_points)
    memories_for_next = memory.split(...)
    encoder_srcs = [...]  # 更新了
```

`EncoderLayer.forward` 里 `srcs` 参数被传给 `MultiScaleDeformableAttention` 作为 value 来源。`reference_points` 只在循环前计算一次，但 `encoder_srcs` 每层都更新。`EncoderLayer` 使用传入的 `srcs` 作为 value，但 `MultiScaleDeformableAttention.forward` 又重新从 `srcs` 中 flatten 提取 value——每层实际上用的是上一层输出重新 reshape 的 `encoder_srcs`，但 `reference_points` 和 `masks` 没有重新计算。这在大多数情况下没有问题，但如果 P2 padding mask 变化会有潜在不一致。

---

## 中等缺陷（影响训练稳定性或指标可信度）

---

### 4. `metrics.py:75` — `AP50` 取的是 `ap_by_thr[0]`，假设第一个阈值是 0.5

```python
"AP50": float(ap_by_thr[0]) if ap_by_thr else 0.0,
```

`iou_thresholds` 默认是 `[0.5, 0.55, ..., 0.95]`，第一个确实是 0.5。但如果配置里改了 `iou_thresholds` 顺序或起点，`AP50` 就是错的。应改为查找 `abs(thr - 0.5) < 1e-6` 对应的 ap 值，与下面 `p50, r50` 的处理保持一致。

---

### 5. `metrics.py:122-124` — `_ap_at_iou` 里 `area_range` 被 `small_only` 无条件覆盖

```python
if small_only:
    area_range = (0, self.small_area_thr)
```

如果有人同时传 `small_only=True, area_range=(x,y)`，外部传入的 `area_range` 被静默忽略。`_dense_subset_metrics` 里的调用 `_ap_at_iou(thr, small_only=True, image_ids=...)` 目前没问题，但接口存在隐患。

---

### 6. `evaluator.py:50` — 评估坐标系与面积基准混用

`postprocess` 里：
```python
bx = denormalize_boxes_xyxy(bx, out_size)
```
`out_size` 取的是 `targets[b].get("size", targets[b]["orig_size"])`，是 resize 后的尺寸，因此预测框是 resize 后的绝对坐标。

而 `coco_evaluate` 里用 `target["area"]` 做 small/medium/large 划分，这个面积是 resize 后的，而 COCO 官方小目标阈值（32²=1024）是基于原始图像尺寸的，**两套 AP_small 数字（自定义 metrics 和 COCO eval）用了不同的面积基准**，会让结果对比产生混淆。

---

### 7. `trainer.py:56` — 非有限 loss 判断时 zero_grad 时机逻辑不清晰

在 `autocast` 块结束后立即判断非有限，此时 grad 来自上一次迭代。非有限路径里（第 63 行）先调用了 `zero_grad`，正常路径里（第 68 行）才调用——如果正常路径走到非有限判断后，上一轮的 grad 还没清除，下一轮的 backward 会**累积**上一轮的梯度（非有限 skip 的那轮没有 backward，所以不会实际累积，但逻辑上不清晰且容易引入 bug）。

---

### 8. `deformable_attention.py:94` — gamma broadcast 维度依赖隐式形状约定

```python
offsets = offsets * gamma[:, :, None, None, None, :]
```

`gamma` 来自 `gamma_vec = gamma.repeat_interleave(2, dim=-1)`，形状为 `[B, Q, 2]`。乘以 `offsets [B, Q, n_heads, n_levels, n_points, 2]` 时 broadcast 到最后一维——这是正确的，但如果 `gamma` 形状出现边界值，整个 broadcast 会静默产生错误结果而不报错。建议加一个 shape assert。

---

## 轻微缺陷

---

### 9. `backbone.py:41-46` — `train_backbone` 参数是死代码

```python
if not train_backbone or not any(layer in pname for layer in train_backbone_layers):
    p.requires_grad_(False)
```

`train_backbone` 参数在 `__init__` 签名里存在，但 `acds_detr.py` 调用时根本没传这个参数，走默认值 `True`。冻结 backbone 的唯一方式是通过 `train_backbone_layers=[]`，但 `train_backbone=False` 这个参数实际上是死代码，会让读代码的人误解。

---

### 10. `acq_loss.py:66` — `m_j` 面积权重区分度低，设计意图未达到

```python
m_j = torch.exp(-tgt_area[small_mask][None, :].float() / (small_thr_norm + 1e-6))
```

当 `tgt_area` 为归一化面积时，实际权重值只在 0.37~0.96 之间浮动，区分度很小，这个权重项几乎不起作用。如果意图是让更小的目标获得更大的排斥权重，建议改为 `1.0 / (tgt_area + eps)` 形式并做归一化。

---

## 优先级汇总

| 优先级 | 位置 | 问题描述 | 影响 |
|--------|------|----------|------|
| P0 | `criterion.py:78-85` | aux_loss key 对齐错误 | aux loss 完全不参与反向传播，训练静默退化 |
| P0 | `acq_loss.py:38` | small_thr_norm 单位不一致 | ACQ 小目标匹配结果系统偏移 |
| P1 | `metrics.py:75` | AP50 硬编码取 `[0]` | 非标准配置下指标值错误 |
| P1 | `evaluator.py:50` | 评估坐标系与面积基准混用 | COCO eval 与自定义 metrics 的 AP_small 不可直接比较 |
| P2 | `metrics.py:122-124` | `small_only` 覆盖外部 `area_range` | 接口隐患 |
| P2 | `trainer.py:56` | zero_grad 时机逻辑不清晰 | 潜在梯度累积风险 |
| P3 | `backbone.py:41-46` | `train_backbone` 死代码 | 可读性问题 |
| P3 | `acq_loss.py:66` | `m_j` 权重区分度低 | 面积权重设计意图未达到 |

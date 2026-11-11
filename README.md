# DINO-ContactSplitSeg

面向金属样品（试样、金相镶嵌块等）的 **DINOv3 + Query 实例分割 + Boundary/Contact/Center + ROI 精修** PyTorch 工程。

> **Status: research prototype / runnable starter.** 代码包含网络、训练、推理、接触切分、标签生成、单元测试与 ONNX 导出脚本；**没有已经训练好的模型权重，也没有在真实金相样品数据上验证精度**。`mock` 只用于 CPU 单测。`MaskQueryHead` 是自包含的 Mask2Former 风格多尺度 masked-attention Decoder，**不是官方 Mask2Former 完整实现**。

## 1. 安装

推荐 Python 3.10+，安装与本机 CUDA 匹配的 PyTorch，然后：

```bash
pip install -e '.[dev]'
pytest -q
```

获取 [DINOv3 官方仓库](https://github.com/facebookresearch/dinov3) 与访问许可下的 ViT-S/16 权重（不要把权重提交 GitHub）：

```bash
git clone https://github.com/facebookresearch/dinov3 external/dinov3
# 从官方申请后将权重放到 weights/dinov3_vits16.pth
```

PyTorch Hub 本地加载：`torch.hub.load('external/dinov3','dinov3_vits16',source='local',weights='weights/dinov3_vits16.pth')`。本项目的 Backbone 使用官方 `get_intermediate_layers(..., reshape=True)`。

## 2. 真实数据

```text
data/
  images/{ID}.png
  instance_masks/{ID}.png
  splits/train.txt
  splits/val.txt
```

Mask 为单通道整数图：`0=背景`，`1..N=独立样品`，可选 `65535=ignore`。**真实接触的两个样品应有不同实例 ID，不要在接缝上标假背景。** 内部斑点、划痕仍属于样品。图像和标注尺寸必须相同。

`boundary` 是样品与背景的外边界；`contact` 只监督不同正实例 ID 直接邻接处，真实黑色间隙不是 contact；`center` 为每实例单个距离变换峰值（高斯热力图）。这些辅助监督均从 ID Mask 自动生成。

```bash
python scripts/validate_dataset.py --root data --split train --size 768
```

## 3. 训练（分阶段）

修改 `configs/full.yaml` 中的 `dinov3_repo`、`dinov3_weights`、`data.root`，并指定 `train.stage`：

- `1`: DINO + Neck + Query Decoder 实例分割
- `2`: 在 stage 1 基础上加入 Boundary + Contact + Center
- `3`: 冻结粗分割，匹配真实实例，训练 ROI Refiner

```bash
python scripts/train.py --config configs/full.yaml
# 下一阶段修改 YAML stage 后：
python scripts/train.py --config configs/full.yaml --resume checkpoints/best.pt
```

注意：每阶段最好设置独立 `checkpoint_dir`，否则可能覆盖上个阶段最佳权重。`stage=3` 会使用匹配的粗预测 mask 生成 ROI 训练样本，而非只使用完美的 GT mask。更严格的实验可增加随机框抖动、预测误差采样与困难样本缓存。

对于 GPU 训练：`amp: true`，实际 Batch Size 按显存调节。当前训练器每个 epoch 重建 AdamW 优化器，便于冻结状态切换，但 **不会恢复优化器/调度器状态**；正式长周期训练建议加入 resume optimizer/scheduler 与 warmup/cosine 策略。

## 4. 推理

```bash
python scripts/predict.py --checkpoint checkpoints/best.pt --image /path/to/example.jpg --output outputs/example
```

结果包括：`instance_map.png`（16 位 ID）、`instances.json`（bbox、面积、轮廓和置信度）、`instance_overlay.png`。

为了保留实例身份，像素使用竞争式归属分配，Contact Split 仅在两个可靠 Center Seed **及** 高可信内部 Contact 响应同时存在时触发；失败时保留原实例，不会凭空刻出背景黑缝。

## 5. CPU 测试与合成数据

```bash
python scripts/make_synthetic_data.py --root data --n 12
pytest -q
python scripts/train.py --config configs/smoke.yaml
python scripts/predict.py --checkpoint checkpoints/best.pt --image data/images/0010.png
```

`configs/smoke.yaml` 使用 **假 Backbone**，输出结果仅用于保证接口可运行，绝不能拿它评价 DINOv3 的实际分割效果。

## 6. ONNX 与部署

```bash
pip install -e '.[export]'
python scripts/export_onnx.py --checkpoint checkpoints/best.pt --size 768 --output coarse_segmenter.onnx
```

脚本只尝试导出粗网络的五个张量输出；ROI 筛选、动态分割及实例融合保持在 Python/C++/C# 侧。导出**尚未完成真实 DINOv3 + ONNX Runtime 的一致性验证**，某些注意力算子可能需要替代实现。ROI Refiner 可通过 `scripts/export_refiner_onnx.py` 单独导出 6 通道动态 Batch 输入，并通过 `scripts/verify_onnx.py` 验证数值一致性；这些功能依赖安装 `onnx`、`onnxruntime`。

## 7. 已知限制与进一步工作

1. 未提供 DINOv3 权重或真实标注；用户示例图仅能用于数据读取 smoke-test，不能充当监督训练集。
2. 当前 Query Decoder 是轻量 Mask2Former-style 实现，不含 MSDeformAttn Pixel Decoder；如果性能不满足要求，可替换为完整实现。
3. 接触分割需要两个有效种子和可见的接触证据；完全无视觉分界的样品不能确保正确分离，应标记供人工复核。
4. 推理优先保留纵横比的整图 letterbox；极大图像可再加切片/跨片实例融合。
5. `scripts/evaluate.py` 提供基础 IoU、匹配 IoU 和计数误差；COCO AP/PQ、HD95/ASSD、速度与多工况 Benchmark 尚待补充。
6. 需要做真实样品训练验证、边界误差分析、模型导出 parity 测试与 C# 包装才能用于生产。

技术参考：[DINOv3](https://github.com/facebookresearch/dinov3)，[Mask2Former](https://github.com/facebookresearch/Mask2Former)。请单独核对第三方代码、权重及商业使用许可。

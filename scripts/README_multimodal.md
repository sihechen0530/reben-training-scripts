# Multimodal Training Script

训练脚本 `train_multimodal.py` 用于训练多模态分类模型，该模型结合了：
- **S2 RGB数据 (3通道)** → DINOv3 backbone
- **S2非RGB通道 (11通道) + S1数据 (2通道)** → ResNet101 backbone

## 基本用法

### 1. 从头开始训练（使用预训练权重）

```bash
python train_multimodal.py \
    --seed 42 \
    --lr 0.001 \
    --epochs 100 \
    --bs 32 \
    --dinov3-model-name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --dinov3-pretrained \
    --resnet-pretrained
```

### 2. 从checkpoint加载backbone权重（冻结参数）

从已有的checkpoint文件加载DINOv3 backbone权重并冻结：

```bash
python train_multimodal.py \
    --seed 42 \
    --lr 0.001 \
    --epochs 100 \
    --bs 32 \
    --dinov3-checkpoint "/home/chen.sihe1/documents/cs6140/reben-training-scripts/scripts/checkpoints_b.2/dinov3-base-42-10-unfreeze-val_mAP_macro-0.58.ckpt" \
    --dinov3-freeze \
    --resnet-pretrained
```

**注意**：脚本会自动从checkpoint文件名或内容中推断DINOv3模型大小（base/large/small/giant）。如果无法推断，请显式指定 `--dinov3-model-name`：

```bash
# 如果checkpoint是large版本
python train_multimodal.py \
    --dinov3-checkpoint "checkpoints/dinov3-large-42-10-unfreeze-val_mAP_macro-0.58.ckpt" \
    --dinov3-model-name "facebook/dinov3-vitl16-pretrain-lvd1689m" \
    --dinov3-freeze
```

**Checkpoint路径支持：**
- 绝对路径：`/path/to/checkpoint.ckpt`
- 相对路径（相对于 `scripts/checkpoints/`）：`dinov3-base-42-10-unfreeze-val_mAP_macro-0.58.ckpt`

### 3. 同时加载两个backbone的checkpoint

```bash
python train_multimodal.py \
    --seed 42 \
    --lr 0.001 \
    --epochs 100 \
    --bs 32 \
    --dinov3-checkpoint "checkpoints_b.2/dinov3-base-42-10-unfreeze-val_mAP_macro-0.58.ckpt" \
    --dinov3-freeze \
    --resnet-checkpoint "checkpoints_b.2/resnet101-42-12-val_mAP_macro-0.75.ckpt" \
    --resnet-freeze
```

### 4. 使用不同的融合策略

```bash
# 使用加权融合
python train_multimodal.py \
    --fusion-type "weighted" \
    --dinov3-checkpoint "checkpoints/dinov3-base-42-10-unfreeze-val_mAP_macro-0.58.ckpt" \
    --dinov3-freeze

# 使用线性投影融合
python train_multimodal.py \
    --fusion-type "linear_projection" \
    --fusion-output-dim 512 \
    --dinov3-checkpoint "checkpoints/dinov3-base-42-10-unfreeze-val_mAP_macro-0.58.ckpt" \
    --dinov3-freeze
```

### 5. 使用MLP分类器

```bash
python train_multimodal.py \
    --classifier-type "mlp" \
    --classifier-hidden-dim 512 \
    --dinov3-checkpoint "checkpoints/dinov3-base-42-10-unfreeze-val_mAP_macro-0.58.ckpt" \
    --dinov3-freeze
```

## 主要参数

### DINOv3配置
- `--dinov3-model-name`: DINOv3模型名称（默认：`facebook/dinov3-vitb16-pretrain-lvd1689m`）
- `--dinov3-pretrained`: 是否使用预训练权重
- `--dinov3-freeze`: 是否冻结DINOv3 backbone
- `--dinov3-lr`: DINOv3 backbone的学习率（默认：1e-4）
- `--dinov3-checkpoint`: 从checkpoint文件加载DINOv3权重

### ResNet101配置
- `--resnet-pretrained`: 是否使用预训练ImageNet权重
- `--resnet-freeze`: 是否冻结ResNet101 backbone
- `--resnet-lr`: ResNet101 backbone的学习率（默认：1e-4）
- `--resnet-checkpoint`: 从checkpoint文件加载ResNet权重

### 融合配置
- `--fusion-type`: 融合策略（`concat`, `weighted`, `linear_projection`）
- `--fusion-output-dim`: 线性投影融合的输出维度（可选）

### 分类器配置
- `--classifier-type`: 分类器类型（`linear`, `mlp`）
- `--classifier-hidden-dim`: MLP分类器的隐藏层维度

### 训练配置
- `--seed`: 随机种子
- `--lr`: 主学习率
- `--epochs`: 训练轮数
- `--bs`: 批次大小
- `--drop-rate`: Dropout率
- `--warmup`: Warmup步数（-1表示自动计算）
- `--workers`: 数据加载器工作进程数
- `--resume-from`: 恢复训练的checkpoint路径

## Checkpoint格式

脚本支持从以下格式的checkpoint文件加载权重：
- PyTorch Lightning checkpoint文件（`.ckpt`）
- 包含 `state_dict` 的checkpoint文件
- BigEarthNetv2_0_ImageClassifier保存的checkpoint

Checkpoint文件路径示例：
```
/home/chen.sihe1/documents/cs6140/reben-training-scripts/scripts/checkpoints_b.2/dinov3-base-42-10-unfreeze-val_mAP_macro-0.58.ckpt
```

## 数据格式

模型期望的输入数据格式：
- **S2数据**: 前N个通道（N = S2波段数，通常为12-14）
  - 前3个通道：RGB（B04, B03, B02）
  - 其余通道：非RGB波段
- **S1数据**: 最后2个通道（VV, VH）

数据加载器会自动处理通道顺序和分离。

## 注意事项

1. **Checkpoint加载**：如果指定了checkpoint路径，脚本会尝试从checkpoint中提取backbone权重。如果找不到匹配的键，会显示错误信息。

2. **冻结参数**：如果设置了 `--dinov3-freeze` 或 `--resnet-freeze`，相应的backbone参数将被冻结，不会在训练中更新。

3. **学习率**：可以为不同的backbone设置不同的学习率。如果从checkpoint加载并冻结，学习率设置将被忽略。

4. **数据路径**：确保数据路径配置正确（在 `scripts/utils.py` 中配置）。

## 示例：完整训练命令

```bash
python train_multimodal.py \
    --seed 42 \
    --lr 0.001 \
    --epochs 100 \
    --bs 32 \
    --drop-rate 0.15 \
    --warmup 1000 \
    --workers 8 \
    --dinov3-model-name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --dinov3-checkpoint "checkpoints_b.2/dinov3-base-42-10-unfreeze-val_mAP_macro-0.58.ckpt" \
    --dinov3-freeze \
    --dinov3-lr 1e-4 \
    --resnet-pretrained \
    --resnet-lr 1e-4 \
    --fusion-type "concat" \
    --classifier-type "linear" \
    --use-wandb \
    --test-run False
```


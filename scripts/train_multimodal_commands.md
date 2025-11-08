# Multimodal Training Commands

## 使用 DINOv3-large checkpoint 训练多模态模型

### Checkpoint 信息
- **路径**: `/home/chen.sihe1/documents/cs6140/reben-training-scripts/scripts/checkpoints/dinov3-large-42-3-dinov3-large_rgb_42_20251107_120819-val_mAP_macro-0.79.ckpt`
- **模型**: DINOv3-large (1024维)
- **训练数据**: RGB (3通道)
- **验证mAP**: 0.79
- **Seed**: 42

### 基本训练命令

```bash
cd /home/chen.sihe1/documents/cs6140/reben-training-scripts/scripts

python train_multimodal.py \
    --seed 42 \
    --lr 0.001 \
    --epochs 100 \
    --bs 32 \
    --drop-rate 0.15 \
    --warmup 1000 \
    --workers 8 \
    --dinov3-checkpoint "checkpoints/dinov3-large-42-3-dinov3-large_rgb_42_20251107_120819-val_mAP_macro-0.79.ckpt" \
    --dinov3-freeze \
    --dinov3-lr 1e-4 \
    --resnet-pretrained \
    --resnet-lr 1e-4 \
    --fusion-type "concat" \
    --classifier-type "linear" \
    --use-wandb False \
    --test-run False
```

### 使用绝对路径

```bash
python train_multimodal.py \
    --seed 42 \
    --lr 0.001 \
    --epochs 100 \
    --bs 32 \
    --dinov3-checkpoint "/home/chen.sihe1/documents/cs6140/reben-training-scripts/scripts/checkpoints/dinov3-large-42-3-dinov3-large_rgb_42_20251107_120819-val_mAP_macro-0.79.ckpt" \
    --dinov3-freeze \
    --resnet-pretrained
```

### 不同配置选项

#### 1. 冻结两个backbone（只训练fusion和classifier）

```bash
python train_multimodal.py \
    --seed 42 \
    --lr 0.001 \
    --epochs 100 \
    --bs 32 \
    --dinov3-checkpoint "checkpoints/dinov3-large-42-3-dinov3-large_rgb_42_20251107_120819-val_mAP_macro-0.79.ckpt" \
    --dinov3-freeze \
    --resnet-freeze
```

#### 2. 使用加权融合

```bash
python train_multimodal.py \
    --seed 42 \
    --lr 0.001 \
    --epochs 100 \
    --bs 32 \
    --dinov3-checkpoint "checkpoints/dinov3-large-42-3-dinov3-large_rgb_42_20251107_120819-val_mAP_macro-0.79.ckpt" \
    --dinov3-freeze \
    --fusion-type "weighted"
```

#### 3. 使用MLP分类器

```bash
python train_multimodal.py \
    --seed 42 \
    --lr 0.001 \
    --epochs 100 \
    --bs 32 \
    --dinov3-checkpoint "checkpoints/dinov3-large-42-3-dinov3-large_rgb_42_20251107_120819-val_mAP_macro-0.79.ckpt" \
    --dinov3-freeze \
    --classifier-type "mlp" \
    --classifier-hidden-dim 512
```

#### 4. 使用线性投影融合

```bash
python train_multimodal.py \
    --seed 42 \
    --lr 0.001 \
    --epochs 100 \
    --bs 32 \
    --dinov3-checkpoint "checkpoints/dinov3-large-42-3-dinov3-large_rgb_42_20251107_120819-val_mAP_macro-0.79.ckpt" \
    --dinov3-freeze \
    --fusion-type "linear_projection" \
    --fusion-output-dim 512
```

#### 5. 使用wandb记录

```bash
python train_multimodal.py \
    --seed 42 \
    --lr 0.001 \
    --epochs 100 \
    --bs 32 \
    --dinov3-checkpoint "checkpoints/dinov3-large-42-3-dinov3-large_rgb_42_20251107_120819-val_mAP_macro-0.79.ckpt" \
    --dinov3-freeze \
    --use-wandb True
```

#### 6. 快速测试（少量epochs和batches）

```bash
python train_multimodal.py \
    --seed 42 \
    --lr 0.001 \
    --epochs 100 \
    --bs 32 \
    --dinov3-checkpoint "checkpoints/dinov3-large-42-3-dinov3-large_rgb_42_20251107_120819-val_mAP_macro-0.79.ckpt" \
    --dinov3-freeze \
    --test-run True
```

### 说明

1. **自动推断模型大小**: 脚本会自动从checkpoint文件名中检测到 `dinov3-large`，并使用 `facebook/dinov3-vitl16-pretrain-lvd1689m` 模型。

2. **默认行为**:
   - DINOv3: 从checkpoint加载并冻结
   - ResNet101: 使用ImageNet预训练权重并fine-tune
   - Fusion: 简单拼接（concat）
   - Classifier: 线性分类器

3. **数据**: 自动使用所有S2波段（RGB在前）+ S1波段

4. **Metrics和划分**: 与baseline完全一致


# 多模态训练脚本使用说明

## 概述

`train_multimodal.py` 是一个用于训练多模态分类模型的训练脚本。该模型结合了：
- **S2 RGB数据 (3通道)**：通过 DINOv3 backbone 处理
- **S2非RGB通道 (9通道) + 可选的S1数据 (2通道)**：通过 ResNet101 backbone 处理
- **特征融合**：使用可配置的融合策略合并两个backbone的特征
- **分类头**：输出最终的分类预测

## 数据流说明

### 当 `use_s1=True` 时：
1. 数据加载器加载的数据通道顺序：`[RGB (3), S2_non-RGB (9), S1 (2)]`
2. Lightning模块分割：`rgb_data = x[:, :3]` 和 `non_rgb_s1_data = x[:, 3:]`
3. 模型处理：`DINOv3(rgb_data) + ResNet(non_rgb_s1_data) -> 融合 -> 分类器`

### 当 `use_s1=False` 时（默认）：
1. 数据加载器加载的数据通道顺序：`[RGB (3), S2_non-RGB (9)]`
2. Lightning模块分割：`rgb_data = x[:, :3]` 和 `non_rgb_data = x[:, 3:]`
3. 模型处理：`DINOv3(rgb_data) + ResNet(non_rgb_data) -> 融合 -> 分类器`

## 参数详解

### 1. 基本训练参数

#### `--seed` (默认: 42)
- **类型**: `int`
- **语义**: 随机种子，用于保证实验的可重复性
- **用法**: `--seed 42`
- **说明**: 设置随机种子后，PyTorch、NumPy等库的随机数生成器都会使用该种子，确保每次运行结果一致

#### `--lr` (默认: 0.001)
- **类型**: `float`
- **语义**: 主学习率，用于训练融合层和分类头
- **用法**: `--lr 0.001`
- **说明**: 这是全局学习率，用于优化融合层和分类头的参数。backbone的学习率通过 `--dinov3-lr` 和 `--resnet-lr` 单独设置

#### `--epochs` (默认: 100)
- **类型**: `int`
- **语义**: 训练的总轮数
- **用法**: `--epochs 100`
- **说明**: 每个epoch会遍历一次完整的训练数据集

#### `--bs` (默认: 32)
- **类型**: `int`
- **语义**: 批次大小（batch size），即每次训练使用的样本数量
- **用法**: `--bs 32`
- **说明**: 较大的batch size可以提高训练速度，但需要更多显存。较小的batch size可以提供更稳定的梯度估计

#### `--drop-rate` (默认: 0.15)
- **类型**: `float`
- **语义**: Dropout率，用于防止过拟合
- **用法**: `--drop-rate 0.15`
- **说明**: 在分类头中使用Dropout层，随机丢弃一定比例的神经元。取值范围通常在0.0-0.5之间

#### `--warmup` (默认: 1000)
- **类型**: `int`
- **语义**: 学习率预热步数，设置为-1表示自动计算
- **用法**: `--warmup 1000` 或 `--warmup -1`
- **说明**: 在训练初期，学习率从0线性增加到目标学习率，这个过程称为warmup。设置为-1时，会自动根据训练步数计算合适的warmup步数

#### `--workers` (默认: 8)
- **类型**: `int`
- **语义**: 数据加载器的工作进程数
- **用法**: `--workers 8`
- **说明**: 更多的worker可以提高数据加载速度，但也会占用更多内存。通常设置为CPU核心数的一半到全部

#### `--use-wandb` / `--no-wandb` (默认: False)
- **类型**: `bool`
- **语义**: 是否使用Weights & Biases (wandb) 进行实验日志记录
- **用法**: `--use-wandb` 或 `--no-wandb`
- **说明**: 如果启用wandb，训练过程中的指标、损失、学习率等信息会被记录到wandb平台，便于可视化和对比实验

#### `--test-run` / `--no-test-run` (默认: True)
- **类型**: `bool`
- **语义**: 是否运行测试模式（使用更少的epoch和batch）
- **用法**: `--test-run` 或 `--no-test-run`
- **说明**: 测试模式会减少训练的epoch数和batch数，用于快速验证代码是否正确运行。正式训练时应使用 `--no-test-run`

---

### 2. DINOv3 Backbone 配置参数

#### `--dinov3-hidden-size` (默认: 768)
- **类型**: `int`
- **语义**: DINOv3模型的隐藏层维度（嵌入维度），决定使用哪个DINOv3模型变体
- **用法**: `--dinov3-hidden-size 384` 或 `--dinov3-hidden-size 768`
- **可选值**:
  - `384`: Small模型 (`facebook/dinov3-vits16-pretrain-lvd1689m`)
  - `768`: Base模型 (`facebook/dinov3-vitb16-pretrain-lvd1689m`) - 默认
  - `1024`: Large模型 (`facebook/dinov3-vitl16-pretrain-lvd1689m`)
  - `1536`: Giant模型 (`facebook/dinov3-vitg16-pretrain-lvd1689m`)
- **说明**: **重要**：当从checkpoint加载权重时，必须确保 `--dinov3-hidden-size` 与checkpoint中模型的隐藏层维度匹配。例如，如果checkpoint是384维的，必须使用 `--dinov3-hidden-size 384`

#### `--dinov3-pretrained` / `--no-dinov3-pretrained` (默认: True)
- **类型**: `bool`
- **语义**: 是否使用预训练的DINOv3权重（从HuggingFace加载）
- **用法**: `--dinov3-pretrained` 或 `--no-dinov3-pretrained`
- **说明**: 如果启用，会从HuggingFace加载预训练的DINOv3权重。如果指定了 `--dinov3-checkpoint`，该参数会被忽略（优先使用checkpoint）

#### `--dinov3-freeze` / `--no-dinov3-freeze` (默认: False)
- **类型**: `bool`
- **语义**: 是否冻结DINOv3 backbone的参数（不进行梯度更新）
- **用法**: `--dinov3-freeze` 或 `--no-dinov3-freeze`
- **说明**: 如果冻结，DINOv3的参数在训练过程中不会被更新，只有融合层和分类头会被训练。这通常用于迁移学习场景，当DINOv3已经在一个相关任务上训练好时

#### `--dinov3-lr` (默认: 1e-4)
- **类型**: `float`
- **语义**: DINOv3 backbone的学习率
- **用法**: `--dinov3-lr 1e-4`
- **说明**: 如果DINOv3未被冻结，该学习率会用于优化DINOv3的参数。通常backbone的学习率比全局学习率小一个数量级

#### `--dinov3-checkpoint` (默认: None)
- **类型**: `str`
- **语义**: 从checkpoint文件加载DINOv3 backbone权重的路径
- **用法**: `--dinov3-checkpoint "path/to/checkpoint.ckpt"` 或 `--dinov3-checkpoint "checkpoint_a.2/dinov3-base-42-3-base_0.00005-val_mAP_macro-0.77.ckpt"`
- **路径支持**:
  - **绝对路径**: `/home/user/checkpoints/dinov3.ckpt`
  - **相对路径**: 相对于 `scripts/checkpoints/` 目录，如 `checkpoint_a.2/dinov3.ckpt`
- **说明**: 
  - 如果指定了checkpoint，会从该文件加载DINOv3的权重
  - **必须确保 `--dinov3-hidden-size` 与checkpoint中模型的维度匹配**，否则会出现维度不匹配的错误
  - 如果checkpoint是384维的，必须使用 `--dinov3-hidden-size 384`
  - 如果checkpoint是768维的，必须使用 `--dinov3-hidden-size 768`

---

### 3. ResNet101 Backbone 配置参数

#### `--resnet-pretrained` / `--no-resnet-pretrained` (默认: True)
- **类型**: `bool`
- **语义**: 是否使用预训练的ResNet101权重（从ImageNet加载）
- **用法**: `--resnet-pretrained` 或 `--no-resnet-pretrained`
- **说明**: 如果启用，会加载在ImageNet上预训练的ResNet101权重。由于输入通道数可能不是3（9或11），第一层卷积会被重新初始化

#### `--resnet-freeze` / `--no-resnet-freeze` (默认: False)
- **类型**: `bool`
- **语义**: 是否冻结ResNet101 backbone的参数（不进行梯度更新）
- **用法**: `--resnet-freeze` 或 `--no-resnet-freeze`
- **说明**: 如果冻结，ResNet的参数在训练过程中不会被更新，只有融合层和分类头会被训练

#### `--resnet-lr` (默认: 1e-4)
- **类型**: `float`
- **语义**: ResNet101 backbone的学习率
- **用法**: `--resnet-lr 1e-4`
- **说明**: 如果ResNet未被冻结，该学习率会用于优化ResNet的参数

#### `--resnet-checkpoint` (默认: None)
- **类型**: `str`
- **语义**: 从checkpoint文件加载ResNet101 backbone权重的路径
- **用法**: `--resnet-checkpoint "path/to/checkpoint.ckpt"`
- **路径支持**: 同 `--dinov3-checkpoint`，支持绝对路径和相对于 `scripts/checkpoints/` 的相对路径
- **说明**: 如果指定了checkpoint，会从该文件加载ResNet的权重

---

### 4. 特征融合配置参数

#### `--fusion-type` (默认: "concat")
- **类型**: `str`
- **语义**: 特征融合策略
- **用法**: `--fusion-type "concat"` 或 `--fusion-type "weighted"` 或 `--fusion-type "linear_projection"`
- **可选值**:
  - `concat`: 简单拼接两个backbone的特征向量
  - `weighted`: 使用可学习的权重对两个backbone的特征进行加权求和
  - `linear_projection`: 使用线性投影层将两个backbone的特征投影到统一维度后拼接
- **说明**: 不同的融合策略适用于不同的场景。`concat`最简单，`weighted`可以让模型学习不同backbone的重要性，`linear_projection`可以减少特征维度

#### `--fusion-output-dim` (默认: None)
- **类型**: `int`
- **语义**: 线性投影融合的输出维度（仅当 `--fusion-type` 为 `linear_projection` 时有效）
- **用法**: `--fusion-output-dim 512`
- **说明**: 如果使用 `linear_projection` 融合，需要指定投影后的特征维度。如果未指定，会使用两个backbone特征维度之和

---

### 5. 分类器配置参数

#### `--classifier-type` (默认: "linear")
- **类型**: `str`
- **语义**: 分类器类型
- **用法**: `--classifier-type "linear"` 或 `--classifier-type "mlp"`
- **可选值**:
  - `linear`: 线性分类器（单个全连接层）
  - `mlp`: 多层感知机分类器（包含隐藏层的全连接网络）
- **说明**: `linear`分类器参数少、训练快，但表达能力有限。`mlp`分类器参数多、表达能力更强，但可能更容易过拟合

#### `--classifier-hidden-dim` (默认: 512)
- **类型**: `int`
- **语义**: MLP分类器的隐藏层维度（仅当 `--classifier-type` 为 `mlp` 时有效）
- **用法**: `--classifier-hidden-dim 512`
- **说明**: 如果使用MLP分类器，需要指定隐藏层的维度。较大的隐藏层维度可以提高模型的表达能力，但也会增加参数量和计算量

---

### 6. 数据配置参数

#### `--use-s1` / `--no-use-s1` (默认: False)
- **类型**: `bool`
- **语义**: 是否包含S1（Sentinel-1）数据
- **用法**: `--use-s1` 或 `--no-use-s1`
- **说明**: 
  - 如果启用（`--use-s1`），ResNet会处理9个S2非RGB通道 + 2个S1通道（VV, VH），共11个通道
  - 如果禁用（`--no-use-s1`，默认），ResNet只处理9个S2非RGB通道
  - **重要**：该参数必须在训练脚本中正确设置，因为它决定了ResNet的输入通道数

---

### 7. 训练恢复参数

#### `--resume-from` (默认: None)
- **类型**: `str`
- **语义**: 恢复训练的checkpoint路径
- **用法**: 
  - `--resume-from "path/to/checkpoint.ckpt"`: 从指定checkpoint恢复
  - `--resume-from "best"`: 从最佳checkpoint恢复（Lightning会自动查找）
  - `--resume-from "last"`: 从最后一个checkpoint恢复（Lightning会自动查找）
- **路径支持**: 支持绝对路径和相对于 `scripts/checkpoints/` 的相对路径
- **说明**: 恢复训练会加载模型权重、优化器状态、学习率调度器状态等，可以从中断的地方继续训练

---

## 使用示例

### 示例1: 从头开始训练（使用预训练权重，不包含S1数据）

```bash
python train_multimodal.py \
    --seed 42 \
    --lr 0.001 \
    --epochs 100 \
    --bs 32 \
    --dinov3-hidden-size 768 \
    --dinov3-pretrained \
    --dinov3-lr 1e-4 \
    --resnet-pretrained \
    --resnet-lr 1e-4 \
    --no-use-s1 \
    --no-test-run
```

### 示例2: 从DINOv3 checkpoint加载权重并冻结（训练ResNet，不包含S1数据）

```bash
python train_multimodal.py \
    --seed 42 \
    --lr 0.001 \
    --epochs 100 \
    --bs 32 \
    --dinov3-hidden-size 384 \
    --dinov3-checkpoint "checkpoint_a.2/dinov3-base-42-3-base_0.00005-val_mAP_macro-0.77.ckpt" \
    --dinov3-freeze \
    --resnet-pretrained \
    --resnet-lr 1e-4 \
    --no-use-s1 \
    --no-test-run
```

**重要提示**: 如果checkpoint是384维的，必须使用 `--dinov3-hidden-size 384`。如果checkpoint是768维的，必须使用 `--dinov3-hidden-size 768`。

### 示例3: 包含S1数据的完整训练

```bash
python train_multimodal.py \
    --seed 42 \
    --lr 0.001 \
    --epochs 100 \
    --bs 32 \
    --dinov3-hidden-size 768 \
    --dinov3-pretrained \
    --dinov3-lr 1e-4 \
    --resnet-pretrained \
    --resnet-lr 1e-4 \
    --use-s1 \
    --no-test-run
```

### 示例4: 使用加权融合和MLP分类器

```bash
python train_multimodal.py \
    --seed 42 \
    --lr 0.001 \
    --epochs 100 \
    --bs 32 \
    --dinov3-hidden-size 384 \
    --dinov3-checkpoint "checkpoint_a.2/dinov3-base-42-3-base_0.00005-val_mAP_macro-0.77.ckpt" \
    --dinov3-freeze \
    --resnet-pretrained \
    --resnet-lr 1e-4 \
    --fusion-type "weighted" \
    --classifier-type "mlp" \
    --classifier-hidden-dim 512 \
    --no-use-s1 \
    --no-test-run
```

### 示例5: 从checkpoint恢复训练

```bash
python train_multimodal.py \
    --seed 42 \
    --lr 0.001 \
    --epochs 100 \
    --bs 32 \
    --dinov3-hidden-size 768 \
    --dinov3-pretrained \
    --resnet-pretrained \
    --resume-from "best" \
    --no-use-s1 \
    --no-test-run
```

### 示例6: 使用wandb记录实验

```bash
python train_multimodal.py \
    --seed 42 \
    --lr 0.001 \
    --epochs 100 \
    --bs 32 \
    --dinov3-hidden-size 768 \
    --dinov3-pretrained \
    --resnet-pretrained \
    --use-wandb \
    --no-test-run
```

---

## 常见问题

### Q1: 如何确定checkpoint的隐藏层维度？

**方法1**: 查看checkpoint文件名。如果文件名包含 `small`、`base`、`large`、`giant`，可以推断：
- `small` → 384维
- `base` → 768维
- `large` → 1024维
- `giant` → 1536维

**方法2**: 如果文件名无法确定，可以尝试加载checkpoint并查看错误信息。如果出现维度不匹配错误，错误信息会显示期望的维度和实际的维度。

**方法3**: 查看训练该checkpoint时使用的参数。如果知道训练时使用的 `--dinov3-hidden-size`，就可以确定维度。

### Q2: 为什么会出现维度不匹配错误？

如果出现 `size mismatch` 错误，通常是因为：
1. `--dinov3-hidden-size` 与checkpoint中模型的维度不匹配
2. 例如，checkpoint是384维的，但使用了 `--dinov3-hidden-size 768`

**解决方法**: 确保 `--dinov3-hidden-size` 与checkpoint的维度匹配。如果checkpoint是384维的，使用 `--dinov3-hidden-size 384`。

### Q3: 如何使用S1数据？

使用 `--use-s1` 参数启用S1数据：
```bash
python train_multimodal.py --use-s1 ...
```

**注意**: 
- 如果启用S1，ResNet的输入通道数会是11（9个S2非RGB + 2个S1）
- 如果禁用S1，ResNet的输入通道数会是9（只有S2非RGB）
- 确保数据集包含S1数据

### Q4: 如何冻结backbone？

使用 `--dinov3-freeze` 或 `--resnet-freeze` 参数：
```bash
python train_multimodal.py --dinov3-freeze --resnet-freeze ...
```

**说明**: 冻结后，backbone的参数不会更新，只有融合层和分类头会被训练。这通常用于迁移学习场景。

### Q5: 如何设置不同的学习率？

使用 `--lr`、`--dinov3-lr`、`--resnet-lr` 参数：
```bash
python train_multimodal.py \
    --lr 0.001 \
    --dinov3-lr 1e-4 \
    --resnet-lr 1e-4 \
    ...
```

**说明**: 
- `--lr`: 全局学习率，用于融合层和分类头
- `--dinov3-lr`: DINOv3 backbone的学习率
- `--resnet-lr`: ResNet backbone的学习率

### Q6: 如何从checkpoint恢复训练？

使用 `--resume-from` 参数：
```bash
python train_multimodal.py --resume-from "best" ...
```

**说明**: 
- `--resume-from "best"`: 从最佳checkpoint恢复
- `--resume-from "last"`: 从最后一个checkpoint恢复
- `--resume-from "path/to/checkpoint.ckpt"`: 从指定checkpoint恢复

---

## 检查点（Checkpoint）格式

脚本支持从以下格式的checkpoint文件加载权重：
- PyTorch Lightning checkpoint文件（`.ckpt`）
- 包含 `state_dict` 的checkpoint文件
- BigEarthNetv2_0_ImageClassifier保存的checkpoint

Checkpoint文件路径示例：
```
/home/user/documents/cs6140/reben-training-scripts/scripts/checkpoint_a.2/dinov3-base-42-3-base_0.00005-val_mAP_macro-0.77.ckpt
```

---

## 数据格式

模型期望的输入数据格式：
- **S2数据**: 
  - 前3个通道：RGB（B04, B03, B02）→ DINOv3
  - 其余通道：非RGB波段 → ResNet
- **S1数据**（可选）: 
  - 最后2个通道：VV, VH → ResNet

数据加载器会自动处理通道顺序和分离。

---

## 注意事项

1. **Checkpoint加载**: 如果指定了checkpoint路径，脚本会尝试从checkpoint中提取backbone权重。如果找不到匹配的键，会显示错误信息。

2. **维度匹配**: **重要**：当从checkpoint加载DINOv3权重时，必须确保 `--dinov3-hidden-size` 与checkpoint中模型的隐藏层维度匹配。否则会出现维度不匹配错误。

3. **冻结参数**: 如果设置了 `--dinov3-freeze` 或 `--resnet-freeze`，相应的backbone参数将被冻结，不会在训练中更新。

4. **学习率**: 可以为不同的backbone设置不同的学习率。如果从checkpoint加载并冻结，学习率设置将被忽略。

5. **数据路径**: 确保数据路径配置正确（在 `scripts/utils.py` 中配置）。

6. **S1数据**: 如果使用 `--use-s1`，确保数据集包含S1数据，否则会出现数据加载错误。

---

## 完整训练命令示例

```bash
python train_multimodal.py \
    --seed 42 \
    --lr 0.001 \
    --epochs 100 \
    --bs 32 \
    --drop-rate 0.15 \
    --warmup 1000 \
    --workers 8 \
    --dinov3-hidden-size 384 \
    --dinov3-checkpoint "checkpoint_a.2/dinov3-base-42-3-base_0.00005-val_mAP_macro-0.77.ckpt" \
    --dinov3-freeze \
    --dinov3-lr 1e-4 \
    --resnet-pretrained \
    --resnet-lr 1e-4 \
    --fusion-type "concat" \
    --classifier-type "linear" \
    --no-use-s1 \
    --use-wandb \
    --no-test-run
```

---

## 联系与支持

如有问题或建议，请联系项目维护者。

# SLURM Training Infrastructure

这个目录包含了用于在SLURM集群上提交BigEarthNet训练任务的基础设施代码。

## 📁 文件说明

- **`config.yaml`** - 主配置文件，包含作业参数、环境变量、数据路径和训练参数
- **`submit.py`** - Python提交脚本，读取配置并提交SLURM作业
- **`sbatch_train.sbatch`** - SLURM批处理脚本模板
- **`env_setup.sh`** - 环境设置脚本（加载CUDA、conda等）

## 🚀 快速开始

### 1. 配置环境

编辑 `config.yaml` 文件，设置你的集群参数：

```yaml
job:
  partition: gpu          # 你的GPU分区名
  account: your_account   # 你的账号
  chdir: "/path/to/reben-training-scripts"
  mail_user: "your@email.com"

data:
  benv2_data_dir: "/path/to/BigEarthNet/data"

train:
  args:
    architecture: "resnet18"
    batch_size: 32
    epochs: 100
    # ... 其他训练参数
```

### 2. 提交单个训练任务

```bash
cd train_scripts
python submit.py
```

### 3. 提交超参数搜索任务

在 `config.yaml` 中配置sweep网格：

```yaml
train:
  sweep:
    grid:
      lr: [0.001, 0.0003, 0.0001]
      batch_size: [32, 64, 128]
      seed: [42, 123, 456]
```

然后提交：

```bash
python submit.py --sweep
```

这会自动生成所有参数组合（笛卡尔积）并提交独立的作业。

### 4. 预览不提交（Dry Run）

```bash
# 预览单个作业
python submit.py --dry-run

# 预览sweep作业
python submit.py --sweep --dry-run
```

## 📋 配置文件详解

### Job配置
```yaml
job:
  name: bigearthnet-ft     # 作业名称
  partition: gpu           # SLURM分区
  qos: normal             # QoS队列
  time: "08:00:00"        # 最大运行时间
  nodes: 1                # 节点数
  gpus_per_task: 1        # 每个任务的GPU数
  cpus_per_task: 8        # 每个任务的CPU核心数
  mem: "32G"              # 内存
  constraint: "a100|v100" # GPU型号限制
```

### 数据配置
```yaml
data:
  benv2_data_dir: "/path/to/data"  # BigEarthNet v2.0 数据目录
```

### 训练参数
```yaml
train:
  args:
    architecture: "resnet18"    # 模型架构
    bandconfig: "all"           # 波段配置：all, s2, s1, rgb
    batch_size: 32              # 批大小
    epochs: 100                 # 训练轮数
    lr: 0.001                   # 学习率
    seed: 42                    # 随机种子
    use_wandb: false            # 是否使用W&B
    config: "../train_scripts/config.yaml"  # 指向此配置文件
```

### 超参数搜索

**方式1: 网格搜索（笛卡尔积）**
```yaml
sweep:
  grid:
    lr: [0.001, 0.0003]
    batch_size: [32, 64]
    seed: [42, 123]
```
这会生成 2×2×2 = 8 个作业

**方式2: 列表文件**
```yaml
sweep:
  list_file: "sweeps.txt"
```

`sweeps.txt` 内容示例：
```yaml
{lr: 0.001, batch_size: 32, seed: 42}
{lr: 0.0003, batch_size: 64, seed: 123}
{lr: 0.0001, batch_size: 128, seed: 456}
```

## 🔧 高级用法

### 使用DINOv3模型

```yaml
train:
  args:
    architecture: "dinov3-base"
    linear_probe: true  # 冻结backbone，只训练分类头
```

### 从检查点恢复

```yaml
train:
  args:
    resume_from: "best"  # 或 "last" 或 "/path/to/checkpoint.ckpt"
```

### 上传到HuggingFace Hub

```yaml
train:
  args:
    upload_to_hub: true
    hf_entity: "your-hf-username"
    use_wandb: true
```

### 使用自定义配置文件

```bash
python submit.py --config my_config.yaml
python submit.py --config my_config.yaml --sweep
```

### 使用自定义sbatch模板

```bash
python submit.py --template my_template.sbatch
```

### 多模态训练（Multimodal Training）

多模态训练支持多个backbone、多种融合策略和分类器类型。要使用多模态训练：

1. **切换到多模态训练脚本**：
```yaml
train:
  script: "train_multimodal.py"  # 从 train_BigEarthNetv2_0.py 改为 train_multimodal.py
```

2. **配置多模态参数**（在 `multimodal_args` 部分）：
```yaml
train:
  script: "train_multimodal.py"
  multimodal_args:
    # 基础参数
    seed: 42
    lr: 0.001
    epochs: 100
    bs: 512
    
    # DINOv3 backbone（处理RGB，3通道）
    dinov3_hidden_size: 768      # 384 (small), 768 (base), 1024 (large), 1536 (giant)
    dinov3_pretrained: true
    dinov3_freeze: false         # true = 冻结（线性探测）, false = 微调
    dinov3_lr: 0.0001
    
    # ResNet101 backbone（处理S2非RGB + 可选S1）
    resnet_pretrained: true
    resnet_freeze: false         # true = 冻结, false = 微调
    resnet_lr: 0.0001
    
    # 融合策略
    fusion_type: "concat"        # concat (默认), weighted, linear_projection
    # fusion_output_dim: 512     # 仅用于 linear_projection
    
    # 分类器
    classifier_type: "linear"    # linear (默认，线性探测), mlp
    classifier_hidden_dim: 512  # 仅用于 MLP
    
    # 数据配置
    use_s1: false               # true = 包含S1（11通道）, false = 仅S2非RGB（9通道）
```

3. **多模态训练示例**：

**示例1：线性探测（冻结所有backbone）**
```yaml
train:
  script: "train_multimodal.py"
  multimodal_args:
    dinov3_freeze: true
    resnet_freeze: true
    fusion_type: "concat"
    classifier_type: "linear"
```

**示例2：加权融合 + MLP分类器**
```yaml
train:
  script: "train_multimodal.py"
  multimodal_args:
    fusion_type: "weighted"
    classifier_type: "mlp"
    classifier_hidden_dim: 512
```

**示例3：包含S1数据**
```yaml
train:
  script: "train_multimodal.py"
  multimodal_args:
    use_s1: true  # ResNet将处理11通道（9 S2非RGB + 2 S1）
```

4. **多模态超参搜索**：
```yaml
train:
  script: "train_multimodal.py"
  multimodal_args:
    # ... 基础配置 ...
  sweep:
    grid:
      fusion_type: ["concat", "weighted", "linear_projection"]
      classifier_type: ["linear", "mlp"]
      dinov3_freeze: [true, false]
      resnet_freeze: [true, false]
```

### 多GPU训练（Multi-GPU Training）

PyTorch Lightning支持多GPU训练，可以显著加速训练过程。

#### 配置多GPU训练

**方式1：通过配置文件（推荐）**

在 `config.yaml` 中配置：

```yaml
job:
  gres: "gpu:v100-sxm2:4"  # 请求4个GPU
  mem: "64G"                # 多GPU时增加内存

train:
  args:
    devices: 4              # 使用4个GPU
    strategy: "ddp"         # 使用DDP策略（推荐）
    bs: 512                 # 每个GPU的batch size
    # 总batch size = bs * num_gpus = 512 * 4 = 2048
```

**方式2：通过命令行参数**

```bash
python scripts/train_BigEarthNetv2_0.py \
    --devices 4 \
    --strategy ddp \
    --bs 512
```

#### 多GPU训练策略

- **`ddp`** (推荐): DistributedDataParallel，单节点多GPU训练的最佳选择
- **`ddp_spawn`**: DDP with spawn，适用于某些环境（Windows、Jupyter）
- **`deepspeed`**: DeepSpeed策略，需要安装DeepSpeed库
- **`fsdp`**: Fully Sharded Data Parallel，适用于大模型

#### 重要注意事项

1. **Batch Size**: 
   - 配置的 `bs` 是每个GPU的batch size
   - 总batch size = `bs * num_gpus`
   - 例如：4个GPU，`bs=512` → 总batch size = 2048

2. **学习率调整**:
   - 多GPU训练时，通常需要按GPU数量线性缩放学习率
   - 例如：单GPU `lr=0.001`，4个GPU时建议 `lr=0.004`
   - 或者使用学习率调度器自动调整

3. **Workers数量**:
   - 建议 `workers = num_gpus * 2-4`
   - 例如：4个GPU → `workers=8` 或 `workers=16`

4. **内存需求**:
   - 多GPU训练需要更多系统内存
   - 建议：`mem = "32G" * num_gpus`（至少）

5. **SLURM配置**:
   ```yaml
   job:
     gres: "gpu:v100-sxm2:4"  # 请求4个GPU
     nodes: 1                  # 单节点多GPU
     cpus_per_task: 16         # 增加CPU核心数
     mem: "64G"                # 增加内存
   ```

#### 多GPU训练示例

**示例1：4个GPU训练**
```yaml
job:
  gres: "gpu:v100-sxm2:4"
  mem: "64G"
  cpus_per_task: 16

train:
  args:
    devices: 4
    strategy: "ddp"
    bs: 512
    lr: 0.004  # 线性缩放：0.001 * 4
    workers: 16
```

**示例2：8个GPU训练**
```yaml
job:
  gres: "gpu:a100:8"
  mem: "128G"
  cpus_per_task: 32

train:
  args:
    devices: 8
    strategy: "ddp"
    bs: 256   # 每个GPU的batch size
    lr: 0.008  # 线性缩放：0.001 * 8
    workers: 32
```

#### 验证多GPU训练

训练开始时会看到类似输出：
```
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
```

#### 故障排查

1. **CUDA out of memory**: 
   - 减少 `bs`（每个GPU的batch size）
   - 增加 `mem`（系统内存）

2. **NCCL错误**:
   - 确保所有GPU在同一节点上
   - 检查网络配置（InfiniBand等）

3. **训练速度没有提升**:
   - 检查数据加载是否成为瓶颈（增加 `workers`）
   - 确保batch size足够大
   - 检查GPU利用率（`nvidia-smi`）

## 📊 监控作业

```bash
# 查看作业队列
squeue -u $USER

# 查看特定作业
squeue -j JOB_ID

# 取消作业
scancel JOB_ID

# 查看作业输出
tail -f logs/bigearthnet-ft-JOBID.out
```

## 🐛 故障排查

### 作业失败
1. 检查日志文件：`logs/bigearthnet-ft-JOBID.err`
2. 验证数据路径：确保 `benv2_data_dir` 指向正确的数据集
3. 检查环境：确保 `env_setup.sh` 正确加载了conda环境

### 数据路径问题
训练脚本会按以下顺序查找数据目录：
1. 如果提供了 `--config` 参数，从配置文件读取 `data.benv2_data_dir`
2. 否则，根据hostname自动选择（mars, erde, pluto等）
3. 最后回退到默认路径

### 提交脚本找不到sbatch
如果看到 "sbatch command not found"，确保：
1. 你在SLURM集群上运行
2. SLURM模块已加载（可能需要 `module load slurm`）

## 📚 示例工作流

### 单次实验
```bash
# 1. 编辑config.yaml设置参数
vim config.yaml

# 2. 预览
python submit.py --dry-run

# 3. 提交
python submit.py

# 4. 监控
squeue -u $USER
tail -f logs/bigearthnet-ft-*.out
```

### 超参数搜索
```bash
# 1. 配置sweep网格
vim config.yaml  # 编辑 train.sweep.grid

# 2. 预览所有作业
python submit.py --sweep --dry-run

# 3. 提交全部
python submit.py --sweep

# 4. 监控全部作业
watch -n 10 squeue -u $USER
```

## 🔗 相关文件

- 训练脚本: `../scripts/train_BigEarthNetv2_0.py`
- 工具函数: `../scripts/utils.py`
- 项目README: `../README.md`

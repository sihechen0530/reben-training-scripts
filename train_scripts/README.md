# SLURM Training Infrastructure

这个目录包含了在 SLURM 集群上启动 BigEarthNet 以及多模态训练任务所需的一切：配置模板、提交脚本、sbatch 模板、环境脚本等。本文档整合了 `MULTI_CONFIG_USAGE.md` 与 `SETUP_DINOV3.md` 的所有信息，只保留一个权威的 README。

## 🧭 快速导航

- [📁 关键文件](#-关键文件)
- [🚀 快速开始](#-快速开始)
- [📋 配置文件详解](#-配置文件详解)
- [🧩 多配置文件批量提交](#-多配置文件批量提交)
- [🦾 DINOv3 设置指南](#-dinov3-设置指南)
- [🔀 多模态训练](#-多模态训练)
- [⚡️ 多-GPU-训练](#️-多-gpu-训练)
- [📊 监控与作业管理](#-监控与作业管理)
- [🐛 常见问题](#-常见问题)
- [📚 示例工作流](#-示例工作流)
- [🔗 相关文件](#-相关文件)

## 📁 关键文件

- **`config.yaml`**：主配置文件，统一管理作业参数、环境变量、数据路径与训练参数
- **`submit.py`**：Python 提交脚本，读取配置并生成/提交 sbatch 作业
- **`sbatch_train.sbatch`**：SLURM 批处理脚本模板
- **`env_setup.sh`**：环境初始化脚本（加载 CUDA、conda、HF token 等）

## 🚀 快速开始

### 1. 准备配置

编辑 `config.yaml` 以匹配你的集群与数据环境：

```yaml
job:
  partition: gpu
  account: your_account
  chdir: "/path/to/reben-training-scripts"
  mail_user: "you@email.com"

data:
  benv2_data_dir: "/path/to/BigEarthNet/data"

train:
  args:
    architecture: "resnet18"
    batch_size: 32
    epochs: 100
    lr: 0.001
```

### 2. 提交单个训练作业

```bash
cd train_scripts
python submit.py
```

### 3. 启用超参搜索（Sweep）

```yaml
train:
  sweep:
    grid:
      lr: [0.001, 0.0003, 0.0001]
      batch_size: [32, 64, 128]
      seed: [42, 123, 456]
```

```bash
python submit.py --sweep
```

### 4. 预览不提交（Dry Run）

```bash
python submit.py --dry-run
python submit.py --sweep --dry-run
```

## 📋 配置文件详解

### Job 配置
```yaml
job:
  name: bigearthnet-ft
  partition: gpu
  qos: normal
  time: "08:00:00"
  nodes: 1
  gpus_per_task: 1
  cpus_per_task: 8
  mem: "32G"
  constraint: "a100|v100"
```

### 数据配置
```yaml
data:
  benv2_data_dir: "/path/to/data"
```

### 训练参数
```yaml
train:
  args:
    architecture: "resnet18"
    bandconfig: "all"   # all, s2, s1, rgb
    batch_size: 32
    epochs: 100
    lr: 0.001
    seed: 42
    use_wandb: false
    config: "../train_scripts/config.yaml"
```

### 超参搜索写法

**方式 1：网格搜索**
```yaml
sweep:
  grid:
    lr: [0.001, 0.0003]
    batch_size: [32, 64]
    seed: [42, 123]
```

**方式 2：列表文件**
```yaml
sweep:
  list_file: "sweeps.txt"
```

`sweeps.txt` 示例：
```yaml
{lr: 0.001, batch_size: 32, seed: 42}
{lr: 0.0003, batch_size: 64, seed: 123}
{lr: 0.0001, batch_size: 128, seed: 456}
```

## 🧩 多配置文件批量提交

`submit.py` 支持一次性提交多个配置并且每个配置都可以自带 sweep。

### 基本命令

```bash
# 单配置（旧方式）
python submit.py --config config.yaml

# 多配置（新方式）
python submit.py --config cfg_small.yaml cfg_base.yaml cfg_large.yaml

# 使用通配符
python submit.py --config config_dinov3_*.yaml

# 为每个配置启用 sweep
python submit.py --config cfg_small.yaml cfg_base.yaml --sweep

# 仅预览
python submit.py --config cfg_*.yaml --dry-run
```

### 典型场景

**场景 1：不同架构的对比实验**

```bash
python submit.py --config \
  config_dinov3_small_lp.yaml \
  config_dinov3_base_lp.yaml \
  config_dinov3_large_lp.yaml
```

输出摘要示例：
```
Using template: sbatch_train.sbatch
Using 3 config file(s):
  - config_dinov3_small_lp.yaml
  - config_dinov3_base_lp.yaml
  - config_dinov3_large_lp.yaml

============================================================
Processing config [1/3]: config_dinov3_small_lp.yaml
... 2760501
Processing config [2/3]: config_dinov3_base_lp.yaml
... 2760502
Processing config [3/3]: config_dinov3_large_lp.yaml
... 2760503
============================================================
SUBMISSION SUMMARY
Total jobs submitted: 3
  [config_dinov3_small_lp.yaml] → Job 2760501
  [config_dinov3_base_lp.yaml] → Job 2760502
  [config_dinov3_large_lp.yaml] → Job 2760503
============================================================
```

**场景 2：每个配置自行 sweep**

```bash
python submit.py --config config_dinov3_small_lp.yaml config_dinov3_base_lp.yaml --sweep
```

- small 配置自带 `lr × bs` 共 4 个任务
- base 配置自带 `lr × drop_rate` 共 4 个任务
- 总计 8 个 sbatch

**场景 3：Dry Run 校验**

```bash
python submit.py --config config_*.yaml --dry-run
```

### 输出目录结构

```
/scratch/zhou.lihan/experiments/
├── small_model/
│   ├── 2760501/
│   │   ├── checkpoints/
│   │   ├── config.yaml
│   │   ├── slurm.out
│   │   └── slurm.err
├── base_model/
│   └── 2760502/
└── large_model/
    └── 2760503/
```

### 关键参数

- `--config`: 多个文件或通配符，默认 `config.yaml`
- `--sweep`: 对每个配置启用 `train.sweep`
- `--dry-run`: 仅生成 sbatch，不提交
- `--template`: 指定 `sbatch` 模板，默认 `sbatch_train.sbatch`

### 最佳实践

1. **命名规范**：`config_dinov3_small_lp.yaml`、`config_multimodal_s2s1.yaml` 等
2. **版本控制**：`git add train_scripts/config_*.yaml`
3. **输出隔离**：为每类实验设置独立 `job.output_dir`
4. **Dry Run 优先**：批量提交前以 `less` 检查生成的脚本

### 故障排查

- 配置文件不存在：脚本会直接报错并终止，保证不会提交任何作业
- job name 重复：SLURM 允许，但建议在配置中区分 `job.name`
- 批量取消：使用总结里的 Job ID → `scancel 2760501 2760502 2760503`

## 🦾 DINOv3 设置指南

### 1. 获取 Hugging Face Token
1. 登录 [Hugging Face](https://huggingface.co/)
2. Settings → Access Tokens → New token
3. 复制形如 `hf_xxxxxxxxx` 的 token

### 2. 配置 Token

**方式 1：`env_setup.sh`（本地/交互式推荐）**
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"
```

**方式 2：`config.yaml`（SLURM 作业推荐）**
```yaml
env:
  HF_TOKEN: "hf_xxxxxxxxxxxxxxxxxxxxx"
  OMP_NUM_THREADS: 8
  TOKENIZERS_PARALLELISM: "false"
```

### 3. 启用 DINOv3 模型

```yaml
train:
  args:
    architecture: "dinov3-base"   # small | base | large | giant
    bandconfig: "s2"
    bs: 32
    epochs: 10
    lr: 0.001
    # linear_probe: true  # 仅训练分类头
```

### 4. 可用模型列表

- `dinov3-small`  → facebook/dinov3-vits16-pretrain-lvd1689m (384 dim)
- `dinov3-base`   → facebook/dinov3-vitb16-pretrain-lvd1689m (768 dim)
- `dinov3-large`  → facebook/dinov3-vitl16-pretrain-lvd1689m (1024 dim)
- `dinov3-giant`  → facebook/dinov3-vitg16-pretrain-lvd1689m (1536 dim)

### 5. 提交命令参考

```bash
python submit.py              # 单次训练
python submit.py --sweep      # 超参搜索
python submit.py --dry-run    # 预览
```

### 6. 注意事项

- ⚠️ 不要把 token commit 到 Git
- token 建议放入环境变量或私有 `.env`
- 认证失败时先确认 token 是否生效
- DINOv3 参数量大，对 GPU/显存要求显著高于 ResNet

### ResNet vs DINOv3

| 特性 | ResNet18 | DINOv3-base |
|------|----------|-------------|
| 参数量 | ~11M | ~86M |
| Token 需求 | ❌ | ✅ |
| GPU 内存 | ~2-4GB | ~8-12GB |
| 训练速度 | 快 | 相对慢 |
| 性能 | 基线 | 通常更好 |

## 🔀 多模态训练

支持多 backbones、融合策略与分类器。关键步骤：

1. 切换脚本：
   ```yaml
   train:
     script: "train_multimodal.py"
   ```
2. 在 `multimodal_args` 中配置 backbone、融合、分类器、数据设置（见下例）。
3. 根据需要添加 sweep：
   ```yaml
   sweep:
     grid:
       fusion_type: ["concat", "weighted", "linear_projection"]
       classifier_type: ["linear", "mlp"]
       dinov3_freeze: [true, false]
       resnet_freeze: [true, false]
   ```

常用片段：

```yaml
multimodal_args:
  dinov3_freeze: true
  resnet_freeze: true
  fusion_type: "concat"
  classifier_type: "linear"
```

```yaml
multimodal_args:
  fusion_type: "weighted"
  classifier_type: "mlp"
  classifier_hidden_dim: 512
```

```yaml
multimodal_args:
  use_s1: true  # 让 ResNet 同时处理 S1 + S2
```

## ⚡️ 多 GPU 训练

借助 PyTorch Lightning：

```yaml
job:
  gres: "gpu:v100-sxm2:4"
  mem: "64G"
  cpus_per_task: 16

train:
  args:
    devices: 4
    strategy: "ddp"
    bs: 512          # 每 GPU batch
    lr: 0.004        # 线性缩放
    workers: 16
```

要点：

1. `bs` 是每块 GPU 的 batch size，总 batch = `bs * devices`
2. 学习率通常按 GPU 数量线性放大
3. `workers ≈ devices * (2~4)`
4. 提前在 `job.gres` / `job.mem` 中申请足够资源
5. 常见策略：`ddp`（推荐）、`ddp_spawn`、`deepspeed`、`fsdp`

常见问题：OOM → 降低 `bs` / 增加 `mem`；NCCL 错误 → 保证单节点 + 网络配置；提速不明显 → 增大 batch、提升数据加载能力。

## 📊 监控与作业管理

```bash
squeue -u $USER          # 作业队列
squeue -j JOB_ID         # 单个作业
scancel JOB_ID           # 取消作业
tail -f logs/*.out       # 实时查看输出
tail -f logs/*.err       # 查看错误
```

## 🐛 常见问题

1. **作业失败**：检查 `logs/<job>.err`；确认数据路径；确保 `env_setup.sh` 启动了正确的 conda 环境。
2. **数据路径错误**：优先读取配置里的 `data.benv2_data_dir`，否则根据 hostname 推断，最后才使用默认值。
3. **`sbatch` 找不到**：确认处于 SLURM 节点并 `module load slurm`。
4. **多配置提交出错**：若任一配置文件不存在脚本会立即报错；确保所有配置可读且 `job.output_dir` 不冲突。
5. **批量取消**：使用总结里打印出的 Job IDs → `scancel {id1..idN}`。

## 📚 示例工作流

### 单次实验
```bash
vim config.yaml
python submit.py --dry-run
python submit.py
squeue -u $USER
tail -f logs/bigearthnet-ft-*.out
```

### 超参数搜索
```bash
vim config.yaml                # 编辑 train.sweep.grid
python submit.py --sweep --dry-run
python submit.py --sweep
watch -n 10 squeue -u $USER
```

## 🔗 相关文件

- 训练脚本：`../scripts/train_BigEarthNetv2_0.py`
- 多模态脚本：`../scripts/train_multimodal.py`
- 实用函数：`../scripts/utils.py`
- 项目总 README：`../README.md`

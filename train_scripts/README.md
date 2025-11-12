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

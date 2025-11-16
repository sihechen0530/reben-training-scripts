# 多配置文件提交功能

`submit.py` 现在支持一次性提交多个不同配置文件的训练任务。

## 功能说明

### 单配置提交（原有方式）
```bash
python submit.py --config config.yaml
```

### 多配置提交（新功能）
```bash
# 提交多个独立配置
python submit.py --config config_dinov3_small_lp.yaml config_dinov3_base_lp.yaml config_dinov3_large_lp.yaml

# 使用通配符提交所有匹配的配置
python submit.py --config config_dinov3_*.yaml

# 每个配置都可以包含自己的 sweep
python submit.py --config config_small.yaml config_base.yaml --sweep
```

## 使用场景

### 场景 1: 不同模型架构的对比实验
创建多个配置文件，每个对应不同的 DINOv3 模型：

```bash
# 一次性提交 small、base、large 三个模型的训练
python submit.py --config \
  config_dinov3_small_lp.yaml \
  config_dinov3_base_lp.yaml \
  config_dinov3_large_lp.yaml
```

**输出示例：**
```
Using template: sbatch_train.sbatch
Using 3 config file(s):
  - config_dinov3_small_lp.yaml
  - config_dinov3_base_lp.yaml
  - config_dinov3_large_lp.yaml

============================================================
Processing config [1/3]: config_dinov3_small_lp.yaml
============================================================
Submitting job from config_dinov3_small_lp.yaml...
✓ Submitted job from config_dinov3_small_lp.yaml: 2760501

============================================================
Processing config [2/3]: config_dinov3_base_lp.yaml
============================================================
Submitting job from config_dinov3_base_lp.yaml...
✓ Submitted job from config_dinov3_base_lp.yaml: 2760502

============================================================
Processing config [3/3]: config_dinov3_large_lp.yaml
============================================================
Submitting job from config_dinov3_large_lp.yaml...
✓ Submitted job from config_dinov3_large_lp.yaml: 2760503

============================================================
SUBMISSION SUMMARY
============================================================
Total jobs submitted: 3
  [config_dinov3_small_lp.yaml] → Job 2760501
  [config_dinov3_base_lp.yaml] → Job 2760502
  [config_dinov3_large_lp.yaml] → Job 2760503
============================================================
```

### 场景 2: 配合 sweep 进行超参数搜索
每个配置文件都可以定义自己的 sweep grid：

**config_dinov3_small_lp.yaml:**
```yaml
train:
  multimodal_args:
    architecture: "dinov3-small"
    # ... 其他参数
  sweep:
    grid:
      lr: [0.0001, 0.0003]
      bs: [32, 64]
```

**config_dinov3_base_lp.yaml:**
```yaml
train:
  multimodal_args:
    architecture: "dinov3-base"
    # ... 其他参数
  sweep:
    grid:
      lr: [0.0001, 0.0003]
      drop_rate: [0.1, 0.15]
```

提交命令：
```bash
python submit.py --config config_dinov3_small_lp.yaml config_dinov3_base_lp.yaml --sweep
```

这会为 small 模型提交 4 个任务（2 lr × 2 bs），为 base 模型提交 4 个任务（2 lr × 2 drop_rate），总共 8 个任务。

### 场景 3: 预览模式（Dry Run）
在实际提交前预览所有配置：

```bash
python submit.py --config config_*.yaml --dry-run
```

这会显示每个配置将生成的 sbatch 脚本，但不会真正提交到 SLURM。

## 输出目录结构

每个配置文件都有自己的 `output_dir` 设置，所以不同配置的输出会分别保存：

```
/scratch/zhou.lihan/experiments/
├── small_model/
│   ├── 2760501/
│   │   ├── checkpoints/
│   │   ├── config.yaml      # config_dinov3_small_lp.yaml 的快照
│   │   ├── slurm.out
│   │   └── slurm.err
│   └── 2760504/
│       └── ...
├── base_model/
│   ├── 2760502/
│   │   ├── checkpoints/
│   │   ├── config.yaml      # config_dinov3_base_lp.yaml 的快照
│   │   ├── slurm.out
│   │   └── slurm.err
│   └── 2760505/
│       └── ...
└── large_model/
    └── 2760503/
        └── ...
```

## 参数说明

```bash
python submit.py --help
```

**`--config`**: 可以指定一个或多个配置文件路径（支持通配符）
- 默认值: `config.yaml`
- 示例: `--config cfg1.yaml cfg2.yaml cfg3.yaml`
- 支持通配符: `--config config_*.yaml`

**`--sweep`**: 为每个配置文件启用超参数搜索
- 每个配置文件需要定义自己的 `train.sweep` 部分

**`--dry-run`**: 预览模式，不实际提交任务

**`--template`**: 指定 sbatch 模板文件
- 默认值: `sbatch_train.sbatch`

## 最佳实践

1. **命名规范**: 使用有意义的配置文件名
   - `config_dinov3_small_lp.yaml` (linear probe)
   - `config_dinov3_base_ft.yaml` (full fine-tuning)
   - `config_multimodal_s2s1.yaml` (multimodal with S1+S2)

2. **版本控制**: 将所有配置文件提交到 git
   ```bash
   git add train_scripts/config_*.yaml
   git commit -m "Add experiment configs for model comparison"
   ```

3. **输出隔离**: 为不同类型的实验设置不同的 `output_dir`
   ```yaml
   # config_dinov3_small_lp.yaml
   job:
     output_dir: "/scratch/user/exp_small_lp"
   
   # config_dinov3_base_lp.yaml
   job:
     output_dir: "/scratch/user/exp_base_lp"
   ```

4. **干跑验证**: 在提交大量任务前先 dry-run 检查
   ```bash
   python submit.py --config config_*.yaml --sweep --dry-run | less
   ```

## 故障排查

**Q: 某个配置文件不存在，会怎样？**  
A: `submit.py` 会在开始前检查所有配置文件，如果有任何一个不存在，会报错并退出，不会提交任何任务。

**Q: 不同配置的 job name 重复了怎么办？**  
A: SLURM 允许相同的 job name，会通过 job ID 区分。但建议在每个配置中设置不同的 `job.name`。

**Q: 如何取消所有提交的任务？**  
A: 可以使用提交总结中的 Job IDs：
```bash
scancel 2760501 2760502 2760503
# 或批量取消
scancel {2760501..2760503}
```

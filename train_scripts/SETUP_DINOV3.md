# 配置 DINOv3 训练

## 1. 获取 Hugging Face Token

DINOv3 模型托管在 Hugging Face 上，需要认证才能访问。

1. 登录 [Hugging Face](https://huggingface.co/)
2. 进入 Settings → Access Tokens
3. 创建一个新的 token（或使用现有的）
4. 复制 token（格式类似：`hf_xxxxxxxxxxxxxxxxxxxxx`）

## 2. 配置 Token

有两种方式配置 token：

### 方式 1：通过环境脚本（推荐用于本地开发）

编辑 `env_setup.sh`：

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"  # 替换为你的实际 token
```

### 方式 2：通过 config.yaml（推荐用于 SLURM 任务）

编辑 `config.yaml`：

```yaml
env:
  OMP_NUM_THREADS: 8
  TOKENIZERS_PARALLELISM: "false"
  HF_TOKEN: "hf_xxxxxxxxxxxxxxxxxxxxx"  # 替换为你的实际 token
```

## 3. 使用 DINOv3 模型

在 `config.yaml` 中修改训练参数：

```yaml
train:
  args:
    architecture: "dinov3-base"  # 可选: dinov3-small, dinov3-base, dinov3-large, dinov3-giant
    bandconfig: "s2"
    bs: 32
    epochs: 10
    lr: 0.001
    # linear_probe: true  # 可选：只训练分类头，冻结 backbone
```

## 4. 可用的 DINOv3 模型

- `dinov3-small` - facebook/dinov3-vits16-pretrain-lvd1689m (384 dim)
- `dinov3-base` - facebook/dinov3-vitb16-pretrain-lvd1689m (768 dim)
- `dinov3-large` - facebook/dinov3-vitl16-pretrain-lvd1689m (1024 dim)
- `dinov3-giant` - facebook/dinov3-vitg16-pretrain-lvd1689m (1536 dim)

## 5. 提交训练任务

```bash
# 单个任务
python submit.py

# 超参数搜索
python submit.py --sweep

# 预览而不提交
python submit.py --dry-run
```

## 注意事项

- ⚠️ **不要将 token 提交到 Git 仓库**
- 建议将 token 存储在环境变量中或使用 `.env` 文件
- 如果遇到认证错误，检查 token 是否正确配置
- DINOv3 模型比 ResNet 大很多，需要更多 GPU 内存

## ResNet vs DINOv3 对比

| 特性 | ResNet18 | DINOv3-base |
|------|----------|-------------|
| 参数量 | ~11M | ~86M |
| Token 需求 | ❌ 不需要 | ✅ 需要 HF token |
| GPU 内存 | ~2-4GB | ~8-12GB |
| 训练速度 | 快 | 较慢 |
| 性能 | 基线 | 通常更好 |


#!/bin/bash
#SBATCH --job-name=train_s2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/train_s2_%j.out
#SBATCH --error=logs/train_s2_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# 创建日志目录
mkdir -p logs

# 激活环境
source ~/.bashrc
conda activate /home/chen.sihe1/.conda/envs/dinov3/

# 打印作业信息
echo "========================================="
echo "作业名称: train_s2"
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Conda 环境: $CONDA_PREFIX"
echo "开始时间: $(date)"
echo "========================================="
echo ""

# 运行训练
python train_BigEarthNetv2_0.py \
    --no-test-run \
    --bandconfig=s2 \
    --lr=0.0001 \
    --bs=512 \
    --resume-from /home/liu.guoy/reben-training-scripts/scripts/checkpoints

# 打印完成信息
echo ""
echo "========================================="
echo "训练完成"
echo "完成时间: $(date)"
echo "========================================="

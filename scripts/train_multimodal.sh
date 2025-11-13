#!/bin/bash
#SBATCH --job-name=dinov3_frozen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/dinov3_frozen_%j.out
#SBATCH --error=logs/dinov3_frozen_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

mkdir -p logs
source ~/.bashrc
conda activate /home/chen.sihe1/.conda/envs/dinov3/

echo "========================================="
echo "作业ID: $SLURM_JOB_ID | 节点: $SLURM_NODELIST"
echo "开始时间: $(date)"
echo "========================================="

python train_multimodal.py \
    --dinov3-hidden-size 1024 \
    --dinov3-checkpoint /home/chen.sihe1/documents/cs6140/reben-training-scripts/scripts/checkpoint_b.1/dinov3-large-42-3-dinov3-large_rgb_42_20251107_120819-val_mAP_macro-0.79.ckpt \
    --dinov3-freeze \
    --no-test-run \
    --seed 42 \
    --lr 0.0001 \
    --use-s1 \
    --resume-from /home/liu.guoy/reben-training-scripts/scripts/checkpoints

echo "完成时间: $(date)"

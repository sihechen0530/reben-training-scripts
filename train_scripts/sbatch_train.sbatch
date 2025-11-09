#!/bin/bash
#SBATCH --job-name={JOB_NAME}
#SBATCH --partition={PARTITION}
#SBATCH --qos={QOS}
#SBATCH --account={ACCOUNT}
#SBATCH --time={TIME}
#SBATCH --nodes={NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --gres={GRES}
#SBATCH --mem={MEM}
#SBATCH --mail-user={MAIL_USER}
#SBATCH --mail-type={MAIL_TYPE}
#SBATCH --output={OUTPUT_DIR}/{JOB_NAME}-%j.out
#SBATCH --error={OUTPUT_DIR}/{JOB_NAME}-%j.err

# Print job information
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "========================================="

# Source environment setup
source {ENV_SETUP_SCRIPT}

# Set environment variables
{ENV_VARS}

# Change to working directory
cd {CHDIR}

# Print Python and CUDA info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "========================================="

# Run training script
echo "Running training command:"
echo "{PYTHON} {TRAIN_SCRIPT} {TRAIN_ARGS}"
echo "========================================="

{PYTHON} {TRAIN_SCRIPT} {TRAIN_ARGS}

EXIT_CODE=$?

echo "========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================="

exit $EXIT_CODE

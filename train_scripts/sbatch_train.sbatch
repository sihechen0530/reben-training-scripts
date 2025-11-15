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
#SBATCH --output={OUTPUT_DIR}/.tmp-{JOB_NAME}-%j.out
#SBATCH --error={OUTPUT_DIR}/.tmp-{JOB_NAME}-%j.err

# Create run directory and move log files immediately
RUN_DIR="{OUTPUT_DIR}/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

# Move SLURM output files to job directory
TEMP_STDOUT="{OUTPUT_DIR}/.tmp-{JOB_NAME}-$SLURM_JOB_ID.out"
TEMP_STDERR="{OUTPUT_DIR}/.tmp-{JOB_NAME}-$SLURM_JOB_ID.err"
FINAL_STDOUT="$RUN_DIR/slurm.out"
FINAL_STDERR="$RUN_DIR/slurm.err"

# Move files (they should exist by now, but SLURM may still be writing)
[ -f "$TEMP_STDOUT" ] && mv "$TEMP_STDOUT" "$FINAL_STDOUT"
[ -f "$TEMP_STDERR" ] && mv "$TEMP_STDERR" "$FINAL_STDERR"

# From now on, redirect all output to the final location
exec 1>>"$FINAL_STDOUT" 2>>"$FINAL_STDERR"

# Print job information
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Run artifacts directory: $RUN_DIR"
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
echo "All artifacts stored in: $RUN_DIR"
echo "  - Checkpoints: $RUN_DIR/checkpoints/"
echo "  - Logs: $RUN_DIR/slurm.out & slurm.err"
echo "  - Config: $RUN_DIR/config.yaml"
echo "========================================="

exit $EXIT_CODE

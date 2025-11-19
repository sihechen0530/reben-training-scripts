#!/usr/bin/env python3
"""
Submit training jobs to SLURM using configuration from config.yaml
Supports both single jobs and hyperparameter sweeps.

Usage:
    python submit.py                    # Submit single job with default config
    python submit.py --sweep            # Submit sweep jobs
    python submit.py --config custom.yaml --sweep
    python submit.py --dry-run          # Preview without submitting
"""

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import re
import getpass

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def format_sbatch_script(
    template_path: str,
    config: Dict[str, Any],
    train_args: Optional[Dict[str, Any]] = None,
    job_suffix: str = ""
) -> str:
    """
    Fill in the sbatch template with values from config.
    
    Args:
        template_path: Path to sbatch template file
        config: Full configuration dictionary
        train_args: Training arguments to override default config
        job_suffix: Suffix to add to job name (for sweep jobs)
    
    Returns:
        Formatted sbatch script as string
    """
    with open(template_path, 'r') as f:
        template = f.read()
    
    job_config = config.get('job', {})
    env_config = config.get('env', {})
    data_config = config.get('data', {})
    train_config = config.get('train', {})

    use_multimodal = train_config.get('use_multimodal', False)
    args_key = 'multimodal_args' if use_multimodal else 'args'

    # Merge default train args with sweep-specific overrides
    final_train_args = train_config.get(args_key, {}).copy()

    # Provide user-friendly aliases for backbone checkpoint loading in multimodal configs
    # Users can specify dino_resume_from/resnet_resume_from in YAML and we'll forward them
    # to the underlying train_multimodal.py CLI flags (dinov3-checkpoint/resnet-checkpoint).
    alias_map = {
        "dino_resume_from": "dinov3_checkpoint",
        "dinov3_resume_from": "dinov3_checkpoint",
        "resnet_resume_from": "resnet_checkpoint",
    }
    for alias_key, target_key in alias_map.items():
        if alias_key in final_train_args:
            value = final_train_args.pop(alias_key)
            if value is not None and value != "":
                # Don't override an explicitly provided target option
                final_train_args.setdefault(target_key, value)
    if train_args:
        final_train_args.update(train_args)
    
    # Get script directory for resolving paths
    script_dir = Path(__file__).parent.resolve()
    
    # Resolve chdir
    chdir = Path(job_config.get('chdir', str(script_dir.parent))).resolve()
    
    # Resolve output_dir - support both absolute and relative paths
    output_dir_config = job_config.get('output_dir', '../ckpt_logs')
    output_dir_path = Path(output_dir_config)
    if output_dir_path.is_absolute():
        # Already absolute, use as-is
        output_dir = output_dir_path.resolve()
    else:
        # Relative path, resolve relative to chdir
        output_dir = (chdir / output_dir_config).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    env_setup_script = script_dir / 'env_setup.sh'
    
    # Determine train script path based on chdir and `train.script` in config
    # Priority:
    # 1. If `train.script` is an absolute path -> use it
    # 2. If `train.script` contains a slash -> treat as relative to chdir
    # 3. If it's a plain filename -> use chdir/scripts/<name> unless chdir already is the scripts dir
    script_key = 'multimodal_script' if use_multimodal else 'script'
    default_script = 'train_multimodal.py' if use_multimodal else 'train_BigEarthNetv2_0.py'
    train_script_name = train_config.get(script_key, default_script)
    train_script_path = Path(train_script_name)
    if train_script_path.is_absolute():
        train_script = train_script_path
    elif '/' in train_script_name:
        train_script = chdir / train_script_name
    else:
        if chdir.name == 'scripts':
            train_script = chdir / train_script_name
        else:
            train_script = chdir / 'scripts' / train_script_name
    
    # Prepare environment variables
    env_vars = []
    for key, value in env_config.items():
        env_vars.append(f'export {key}="{value}"')
    env_vars_str = '\n'.join(env_vars) if env_vars else '# No additional env vars'
    
    # Prepare training arguments
    train_args_list = []
    
    # Add no-test-run flag (disable test mode for actual training)
    train_args_list.append('--no-test-run')
    
    # Add data directory from config
    if 'benv2_data_dir' in data_config:
        # Note: We'll pass this via config file to the training script
        # The training script should be updated to read from config
        pass
    
    # Add all training arguments
    for key, value in final_train_args.items():
        # Map config -> config-path because the training script expects --config-path
        if key == 'config':
            key_formatted = 'config-path'
        else:
            key_formatted = key.replace('_', '-')

        if isinstance(value, bool):
            if value:
                train_args_list.append(f'--{key_formatted}')
        else:
            train_args_list.append(f'--{key_formatted}={value}')
    
    train_args_str = ' '.join(train_args_list)
    
    # Job name with suffix for sweeps
    job_name = job_config.get('name', 'bigearthnet-ft')
    if job_suffix:
        job_name = f"{job_name}-{job_suffix}"
    
    # Handle GRES (GPU resource specification)
    # Support both simple format (gpus_per_task: 1) and full format (gres: "gpu:v100-sxm2:1")
    if 'gres' in job_config:
        gres = job_config['gres']
    elif 'gpus_per_task' in job_config:
        gpus = job_config['gpus_per_task']
        # If gpus_per_task contains ':', it's already in full format (e.g., "v100-sxm2:1")
        if isinstance(gpus, str) and ':' in gpus:
            gres = f"gpu:{gpus}"
        else:
            gres = f"gpu:{gpus}"
    else:
        gres = "gpu:1"
    
    # Fill in template placeholders
    # Treat account as optional; if omitted or equals current login, don't include it
    account_val = job_config.get('account', '')
    # If account was set to the current unix username, assume it's not an SLURM account name
    if account_val == getpass.getuser():
        account_val = ''

    replacements = {
        'JOB_NAME': job_name,
        'PARTITION': job_config.get('partition', 'gpu'),
        'QOS': job_config.get('qos', 'normal'),
        'ACCOUNT': account_val,
        'TIME': job_config.get('time', '08:00:00'),
        'NODES': str(job_config.get('nodes', 1)),
        'CPUS_PER_TASK': str(job_config.get('cpus_per_task', 8)),
        'GRES': gres,
        'MEM': job_config.get('mem', '32G'),
        'MAIL_USER': job_config.get('mail_user', ''),
        'MAIL_TYPE': job_config.get('mail_type', 'END,FAIL'),
        'OUTPUT_DIR': str(output_dir),
        'ENV_SETUP_SCRIPT': str(env_setup_script),
        'ENV_VARS': env_vars_str,
        'CHDIR': str(chdir),
        'PYTHON': train_config.get('python', 'python'),
        'TRAIN_SCRIPT': str(train_script),
        'TRAIN_ARGS': train_args_str,
    }
    
    script = template
    for key, value in replacements.items():
        script = script.replace(f'{{{key}}}', value)

    # Remove any SBATCH lines that ended up with empty assignments, e.g. "#SBATCH --account="
    cleaned_lines = []
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith('#SBATCH') and re.search(r'--[A-Za-z0-9_-]+=\s*$', stripped):
            # skip lines like "#SBATCH --account=" or "#SBATCH --constraint=" when no value
            continue
        cleaned_lines.append(line)

    script = '\n'.join(cleaned_lines)
    return script


def generate_sweep_configs(config: Dict[str, Any]) -> List[tuple[Dict[str, Any], str]]:
    """
    Generate sweep configurations from grid or list file.
    
    Returns:
        List of (config_dict, job_suffix) tuples
    """
    sweep_config = config.get('train', {}).get('sweep', {})
    
    if not sweep_config:
        print("No sweep configuration found. Use --sweep with a sweep config in config.yaml")
        return []
    
    configs = []
    
    # Method 1: Grid search (Cartesian product)
    if 'grid' in sweep_config:
        grid = sweep_config['grid']
        keys = list(grid.keys())
        values = list(grid.values())
        
        # Generate all combinations
        for i, combination in enumerate(itertools.product(*values)):
            sweep_args = dict(zip(keys, combination))
            # Create descriptive suffix
            suffix = '-'.join([f"{k}_{v}" for k, v in sweep_args.items()])
            configs.append((sweep_args, suffix))
        
        print(f"Generated {len(configs)} sweep configurations from grid")
    
    # Method 2: List file (each line is a configuration)
    elif 'list_file' in sweep_config:
        list_file = Path(sweep_config['list_file'])
        if not list_file.exists():
            print(f"Error: Sweep list file not found: {list_file}")
            return []
        
        with open(list_file, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Parse line as YAML or key=value pairs
                try:
                    sweep_args = yaml.safe_load(line)
                    suffix = f"sweep_{i}"
                    configs.append((sweep_args, suffix))
                except:
                    print(f"Warning: Could not parse line {i}: {line}")
        
        print(f"Generated {len(configs)} sweep configurations from list file")
    
    return configs


def submit_job(script_content: str, job_name: str, dry_run: bool = False) -> Optional[str]:
    """
    Submit a job to SLURM.
    
    Args:
        script_content: The sbatch script content
        job_name: Name of the job
        dry_run: If True, only print the script without submitting
    
    Returns:
        Job ID if submitted, None otherwise
    """
    if dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN - Job: {job_name}")
        print(f"{'='*60}")
        print(script_content)
        print(f"{'='*60}\n")
        return None
    
    # Submit via sbatch
    try:
        result = subprocess.run(
            ['sbatch'],
            input=script_content.encode(),
            capture_output=True,
            check=True
        )
        output = result.stdout.decode().strip()
        # Extract job ID from output like "Submitted batch job 12345"
        job_id = output.split()[-1] if output else "unknown"
        print(f"✓ Submitted job: {job_name} (Job ID: {job_id})")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to submit job {job_name}")
        print(f"  Error: {e.stderr.decode()}")
        return None
    except FileNotFoundError:
        print("✗ Error: sbatch command not found. Are you on a SLURM cluster?")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Submit BigEarthNet training jobs to SLURM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit single job with default config
  python submit.py
  
  # Submit multiple jobs with different configs
  python submit.py --config config_small.yaml config_base.yaml config_large.yaml
  
  # Submit hyperparameter sweep
  python submit.py --sweep
  
  # Submit sweeps for multiple configs
  python submit.py --config config_dinov3_small_lp.yaml config_dinov3_base_lp.yaml --sweep
  
  # Use custom config file
  python submit.py --config my_config.yaml
  
  # Dry run (preview without submitting)
  python submit.py --dry-run
  python submit.py --config config_*.yaml --sweep --dry-run
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        nargs='+',
        default=['config.yaml'],
        help='Path to config YAML file(s). Can specify multiple files to submit multiple jobs. (default: config.yaml)'
    )
    parser.add_argument(
        '--template',
        type=str,
        default='sbatch_train.sbatch',
        help='Path to sbatch template file (default: sbatch_train.sbatch)'
    )
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='Run hyperparameter sweep instead of single job'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print sbatch scripts without submitting them'
    )
    parser.add_argument(
        '--follow',
        '-f',
        action='store_true',
        help='Follow (tail -f) the job output after submission'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    
    # Handle multiple config files
    config_paths = [script_dir / cfg for cfg in args.config]
    template_path = script_dir / args.template
    
    # Check if template exists
    if not template_path.exists():
        print(f"Error: Template file not found: {template_path}")
        sys.exit(1)
    
    # Check if all config files exist
    for config_path in config_paths:
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
    
    print(f"Using template: {template_path}")
    print(f"Using {len(config_paths)} config file(s):")
    for cfg_path in config_paths:
        print(f"  - {cfg_path}")
    print()
    
    # Track all submitted jobs across all configs
    all_submitted_jobs = []
    
    # Process each config file
    for config_idx, config_path in enumerate(config_paths):
        # Load configuration
        config = load_config(str(config_path))
        
        if len(config_paths) > 1:
            print(f"\n{'='*60}")
            print(f"Processing config [{config_idx + 1}/{len(config_paths)}]: {config_path.name}")
            print(f"{'='*60}")
        
        if args.sweep:
            # Generate and submit sweep jobs
            sweep_configs = generate_sweep_configs(config)
            
            if not sweep_configs:
                print(f"No sweep configurations to submit for {config_path.name}")
                continue
            
            print(f"Submitting {len(sweep_configs)} sweep jobs from {config_path.name}...")
            
            submitted_jobs = []
            for sweep_args, job_suffix in sweep_configs:
                script = format_sbatch_script(
                    str(template_path),
                    config,
                    train_args=sweep_args,
                    job_suffix=job_suffix
                )
                job_name = f"{config.get('job', {}).get('name', 'job')}-{job_suffix}"
                job_id = submit_job(script, job_name, dry_run=args.dry_run)
                if job_id:
                    submitted_jobs.append(job_id)
                    all_submitted_jobs.append((config_path.name, job_id))
            
            if not args.dry_run:
                print(f"✓ Submitted {len(submitted_jobs)} jobs from {config_path.name}")
                print(f"  Job IDs: {', '.join(submitted_jobs)}")
        else:
            # Submit single job per config
            print(f"Submitting job from {config_path.name}...")
            script = format_sbatch_script(str(template_path), config)
            job_name = config.get('job', {}).get('name', 'bigearthnet-ft')
            job_id = submit_job(script, job_name, dry_run=args.dry_run)
            
            if job_id and not args.dry_run:
                print(f"✓ Submitted job from {config_path.name}: {job_id}")
                all_submitted_jobs.append((config_path.name, job_id))
    
    # Summary
    if not args.dry_run and all_submitted_jobs:
        print(f"\n{'='*60}")
        print(f"SUBMISSION SUMMARY")
        print(f"{'='*60}")
        print(f"Total jobs submitted: {len(all_submitted_jobs)}")
        for cfg_name, job_id in all_submitted_jobs:
            print(f"  [{cfg_name}] → Job {job_id}")
        print(f"{'='*60}")
    elif args.dry_run:
        print("\n(Dry run - no jobs were actually submitted)")


if __name__ == '__main__':
    main()

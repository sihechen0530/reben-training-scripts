#!/usr/bin/env python3
"""
Update old checkpoint files to be compatible with new model parameters.

This script adds missing hyperparameters to old checkpoints so they can be 
loaded with the updated model code.

Usage:
    python update_checkpoint.py <checkpoint_path> [--output <output_path>]
"""

import argparse
import torch
from pathlib import Path


def update_checkpoint(
    checkpoint_path: str,
    output_path: str = None,
    head_type: str = "linear",
    mlp_hidden_dims: list = None,
    head_dropout: float = 0.15,
):
    """
    Update a checkpoint file with new hyperparameters.
    
    Args:
        checkpoint_path: Path to the original checkpoint
        output_path: Path to save updated checkpoint (defaults to <original>_updated.ckpt)
        head_type: Classification head type ("linear" or "mlp")
        mlp_hidden_dims: Hidden dimensions for MLP head
        head_dropout: Dropout rate for classification head
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if hyperparameters exist
    if 'hyper_parameters' not in checkpoint:
        print("Warning: No hyper_parameters found in checkpoint")
        checkpoint['hyper_parameters'] = {}
    
    hparams = checkpoint['hyper_parameters']
    
    # Add new parameters if they don't exist
    updates_made = []
    
    if 'head_type' not in hparams:
        hparams['head_type'] = head_type
        updates_made.append(f"head_type={head_type}")
    
    if 'mlp_hidden_dims' not in hparams:
        hparams['mlp_hidden_dims'] = mlp_hidden_dims
        updates_made.append(f"mlp_hidden_dims={mlp_hidden_dims}")
    
    if 'head_dropout' not in hparams:
        hparams['head_dropout'] = head_dropout
        updates_made.append(f"head_dropout={head_dropout}")
    
    if not updates_made:
        print("No updates needed - checkpoint already has all required parameters")
        return
    
    print(f"Adding parameters: {', '.join(updates_made)}")
    
    # Save updated checkpoint
    if output_path is None:
        output_path = checkpoint_path.parent / f"{checkpoint_path.stem}_updated.ckpt"
    else:
        output_path = Path(output_path)
    
    print(f"Saving updated checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    print("✓ Checkpoint updated successfully!")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Update old checkpoint with new hyperparameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to the checkpoint file to update'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Output path for updated checkpoint (default: <original>_updated.ckpt)'
    )
    parser.add_argument(
        '--head-type',
        type=str,
        default='linear',
        choices=['linear', 'mlp'],
        help='Classification head type (default: linear)'
    )
    parser.add_argument(
        '--mlp-dims',
        type=str,
        default=None,
        help='Comma-separated MLP hidden dimensions (e.g., "1024,512")'
    )
    parser.add_argument(
        '--head-dropout',
        type=float,
        default=0.15,
        help='Dropout rate for classification head (default: 0.15)'
    )
    
    args = parser.parse_args()
    
    # Parse MLP dimensions
    mlp_dims = None
    if args.mlp_dims:
        mlp_dims = [int(d.strip()) for d in args.mlp_dims.split(',')]
    
    try:
        update_checkpoint(
            args.checkpoint,
            args.output,
            args.head_type,
            mlp_dims,
            args.head_dropout,
        )
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

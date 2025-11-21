#!/usr/bin/env python3
"""
Script to analyze BigEarthNet v2.0 dataset statistics.

This script loads the BigEarthNet metadata and provides comprehensive analysis:
- Dataset size and splits
- Label distribution
- Class distribution
- Multi-label statistics
- Visualizations
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# BigEarthNet v2.0 class labels (19 classes)
BIGEARTHNET_LABELS = {
    'Urban fabric': 0,
    'Industrial or commercial units': 1,
    'Arable land': 2,
    'Permanent crops': 3,
    'Pastures': 4,
    'Complex cultivation patterns': 5,
    'Land principally occupied by agriculture': 6,
    'Broad-leaved forest': 7,
    'Coniferous forest': 8,
    'Mixed forest': 9,
    'Natural grassland and sparsely vegetated areas': 10,
    'Moors, heathland and sclerophyllous vegetation': 11,
    'Transitional woodland, shrub': 12,
    'Beaches, dunes, sands': 13,
    'Bare rock': 14,
    'Sparsely vegetated areas': 15,
    'Burnt areas': 16,
    'Inland waters': 17,
    'Marine waters': 18
}

LABEL_NAMES = list(BIGEARTHNET_LABELS.keys())
NUM_CLASSES = len(BIGEARTHNET_LABELS)


def extract_labels_from_row(row) -> List[str]:
    """
    Extract labels from a metadata row.
    
    Returns:
        List of label strings
    """
    if 'labels' not in row:
        return []
    
    labels = row['labels']
    
    if labels is None or (isinstance(labels, float) and pd.isna(labels)):
        return []
    
    # Handle different label formats
    label_list = []
    if isinstance(labels, list):
        label_list = labels
    elif isinstance(labels, str):
        try:
            import ast
            label_list = ast.literal_eval(labels)
        except:
            label_list = [l.strip() for l in labels.split(',') if l.strip()]
    elif isinstance(labels, (np.ndarray, pd.Series)):
        label_list = labels.tolist()
    
    # Filter to valid labels only
    valid_labels = []
    for label in label_list:
        if isinstance(label, str):
            label = label.strip()
        else:
            label = str(label).strip()
        
        if label in BIGEARTHNET_LABELS:
            valid_labels.append(label)
    
    return valid_labels


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """Load BigEarthNet metadata from parquet file."""
    print(f"Loading metadata from: {metadata_path}")
    df = pd.read_parquet(metadata_path)
    print(f"Loaded {len(df)} patches")
    return df


def analyze_splits(df: pd.DataFrame) -> Dict:
    """Analyze dataset splits."""
    if 'split' not in df.columns:
        print("Warning: 'split' column not found in metadata")
        return {}
    
    split_counts = df['split'].value_counts().to_dict()
    total = len(df)
    
    print("\n" + "="*80)
    print("DATASET SPLITS")
    print("="*80)
    for split, count in sorted(split_counts.items()):
        percentage = 100 * count / total
        print(f"{split:15s}: {count:8d} patches ({percentage:5.2f}%)")
    print(f"{'Total':15s}: {total:8d} patches")
    
    return split_counts


def analyze_labels(df: pd.DataFrame) -> Dict:
    """Analyze label distribution and statistics."""
    print("\n" + "="*80)
    print("LABEL ANALYSIS")
    print("="*80)
    
    # Extract labels for all patches
    all_labels = []
    labels_per_patch = []
    class_counts = Counter()
    
    print("Extracting labels from metadata...")
    for idx, row in df.iterrows():
        labels = extract_labels_from_row(row)
        all_labels.extend(labels)
        labels_per_patch.append(len(labels))
        
        for label in labels:
            class_counts[label] += 1
    
    # Statistics
    total_patches = len(df)
    patches_with_labels = sum(1 for count in labels_per_patch if count > 0)
    patches_without_labels = total_patches - patches_with_labels
    
    print(f"\nTotal patches: {total_patches}")
    print(f"Patches with labels: {patches_with_labels} ({100*patches_with_labels/total_patches:.2f}%)")
    print(f"Patches without labels: {patches_without_labels} ({100*patches_without_labels/total_patches:.2f}%)")
    
    # Labels per patch statistics
    labels_per_patch_array = np.array(labels_per_patch)
    print(f"\nLabels per patch statistics:")
    print(f"  Mean: {labels_per_patch_array.mean():.2f}")
    print(f"  Median: {np.median(labels_per_patch_array):.2f}")
    print(f"  Min: {labels_per_patch_array.min()}")
    print(f"  Max: {labels_per_patch_array.max()}")
    print(f"  Std: {labels_per_patch_array.std():.2f}")
    
    # Distribution of number of labels
    label_count_dist = Counter(labels_per_patch)
    print(f"\nDistribution of number of labels per patch:")
    for num_labels in sorted(label_count_dist.keys()):
        count = label_count_dist[num_labels]
        percentage = 100 * count / total_patches
        print(f"  {num_labels} label(s): {count:6d} patches ({percentage:5.2f}%)")
    
    # Class frequency
    print(f"\nClass frequency (total occurrences):")
    sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])
    for label, count in sorted_classes:
        percentage = 100 * count / total_patches
        print(f"  {label:50s}: {count:6d} ({percentage:5.2f}%)")
    
    return {
        'class_counts': class_counts,
        'labels_per_patch': labels_per_patch,
        'patches_with_labels': patches_with_labels,
        'patches_without_labels': patches_without_labels,
    }


def analyze_by_split(df: pd.DataFrame) -> Dict:
    """Analyze label distribution by split."""
    if 'split' not in df.columns:
        return {}
    
    print("\n" + "="*80)
    print("LABEL DISTRIBUTION BY SPLIT")
    print("="*80)
    
    split_analyses = {}
    
    for split in df['split'].unique():
        split_df = df[df['split'] == split]
        print(f"\n{split.upper()} Split ({len(split_df)} patches):")
        
        class_counts = Counter()
        labels_per_patch = []
        
        for idx, row in split_df.iterrows():
            labels = extract_labels_from_row(row)
            labels_per_patch.append(len(labels))
            for label in labels:
                class_counts[label] += 1
        
        # Top classes
        sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])[:10]
        print(f"  Top 10 classes:")
        for label, count in sorted_classes:
            percentage = 100 * count / len(split_df)
            print(f"    {label:50s}: {count:6d} ({percentage:5.2f}%)")
        
        # Labels per patch stats
        if labels_per_patch:
            labels_array = np.array(labels_per_patch)
            print(f"  Labels per patch - Mean: {labels_array.mean():.2f}, "
                  f"Median: {np.median(labels_array):.2f}, "
                  f"Min: {labels_array.min()}, Max: {labels_array.max()}")
        
        split_analyses[split] = {
            'class_counts': class_counts,
            'labels_per_patch': labels_per_patch,
        }
    
    return split_analyses


def analyze_class_cooccurrence(df: pd.DataFrame, top_n: int = 10) -> Dict:
    """Analyze which classes frequently co-occur."""
    print("\n" + "="*80)
    print(f"CLASS CO-OCCURRENCE ANALYSIS (Top {top_n} pairs)")
    print("="*80)
    
    cooccurrence = defaultdict(int)
    
    for idx, row in df.iterrows():
        labels = extract_labels_from_row(row)
        # Count all pairs
        for i, label1 in enumerate(labels):
            for label2 in labels[i+1:]:
                pair = tuple(sorted([label1, label2]))
                cooccurrence[pair] += 1
    
    # Sort by frequency
    sorted_pairs = sorted(cooccurrence.items(), key=lambda x: -x[1])[:top_n]
    
    print(f"\nMost frequent class pairs:")
    for (label1, label2), count in sorted_pairs:
        percentage = 100 * count / len(df)
        print(f"  {label1:40s} + {label2:40s}: {count:6d} ({percentage:5.2f}%)")
    
    return dict(cooccurrence)


def create_visualizations(df: pd.DataFrame, output_dir: Optional[str] = None):
    """Create visualization plots."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    
    # 1. Split distribution pie chart
    if 'split' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 8))
        split_counts = df['split'].value_counts()
        ax.pie(split_counts.values, labels=split_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('Dataset Split Distribution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'split_distribution.png'), dpi=300, bbox_inches='tight')
            print(f"Saved: {os.path.join(output_dir, 'split_distribution.png')}")
        plt.close()
    
    # 2. Class frequency bar chart
    class_counts = Counter()
    for idx, row in df.iterrows():
        labels = extract_labels_from_row(row)
        for label in labels:
            class_counts[label] += 1
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])
    classes = [c[0] for c in sorted_classes]
    counts = [c[1] for c in sorted_classes]
    
    bars = ax.barh(range(len(classes)), counts)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel('Number of Patches', fontsize=12)
    ax.set_title('Class Frequency Distribution', fontsize=16, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        percentage = 100 * count / len(df)
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{count} ({percentage:.1f}%)', va='center', fontsize=9)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'class_frequency.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(output_dir, 'class_frequency.png')}")
    plt.close()
    
    # 3. Labels per patch distribution
    labels_per_patch = []
    for idx, row in df.iterrows():
        labels = extract_labels_from_row(row)
        labels_per_patch.append(len(labels))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    label_count_dist = Counter(labels_per_patch)
    counts = [label_count_dist.get(i, 0) for i in range(max(labels_per_patch) + 1)]
    ax.bar(range(len(counts)), counts, alpha=0.7, color='steelblue')
    ax.set_xlabel('Number of Labels per Patch', fontsize=12)
    ax.set_ylabel('Number of Patches', fontsize=12)
    ax.set_title('Distribution of Labels per Patch', fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(counts)))
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, count in enumerate(counts):
        if count > 0:
            ax.text(i, count + max(counts) * 0.01, str(count), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'labels_per_patch.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(output_dir, 'labels_per_patch.png')}")
    plt.close()
    
    # 4. Class distribution by split (if split column exists)
    if 'split' in df.columns:
        fig, axes = plt.subplots(1, len(df['split'].unique()), figsize=(18, 8))
        if len(df['split'].unique()) == 1:
            axes = [axes]
        
        for ax, split in zip(axes, sorted(df['split'].unique())):
            split_df = df[df['split'] == split]
            split_class_counts = Counter()
            
            for idx, row in split_df.iterrows():
                labels = extract_labels_from_row(row)
                for label in labels:
                    split_class_counts[label] += 1
            
            sorted_classes = sorted(split_class_counts.items(), key=lambda x: -x[1])
            classes = [c[0] for c in sorted_classes]
            counts = [c[1] for c in sorted_classes]
            
            bars = ax.barh(range(len(classes)), counts)
            ax.set_yticks(range(len(classes)))
            ax.set_yticklabels(classes, fontsize=8)
            ax.set_xlabel('Number of Patches', fontsize=10)
            ax.set_title(f'{split.upper()} Split\n({len(split_df)} patches)', fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'class_distribution_by_split.png'), dpi=300, bbox_inches='tight')
            print(f"Saved: {os.path.join(output_dir, 'class_distribution_by_split.png')}")
        plt.close()
    
    print("\nVisualizations created successfully!")


def save_statistics(df: pd.DataFrame, output_file: str):
    """Save detailed statistics to a text file."""
    print(f"\nSaving statistics to: {output_file}")
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BIGEARTHNET V2.0 DATASET ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        # Basic info
        f.write(f"Total patches: {len(df)}\n")
        f.write(f"Number of classes: {NUM_CLASSES}\n\n")
        
        # Columns
        f.write(f"Metadata columns: {list(df.columns)}\n\n")
        
        # Splits
        if 'split' in df.columns:
            f.write("Dataset Splits:\n")
            split_counts = df['split'].value_counts()
            for split, count in sorted(split_counts.items()):
                percentage = 100 * count / len(df)
                f.write(f"  {split:15s}: {count:8d} ({percentage:5.2f}%)\n")
            f.write("\n")
        
        # Label analysis
        class_counts = Counter()
        labels_per_patch = []
        
        for idx, row in df.iterrows():
            labels = extract_labels_from_row(row)
            labels_per_patch.append(len(labels))
            for label in labels:
                class_counts[label] += 1
        
        f.write("Class Frequency:\n")
        sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])
        for label, count in sorted_classes:
            percentage = 100 * count / len(df)
            f.write(f"  {label:50s}: {count:6d} ({percentage:5.2f}%)\n")
        
        f.write(f"\nLabels per patch statistics:\n")
        labels_array = np.array(labels_per_patch)
        f.write(f"  Mean: {labels_array.mean():.2f}\n")
        f.write(f"  Median: {np.median(labels_array):.2f}\n")
        f.write(f"  Min: {labels_array.min()}\n")
        f.write(f"  Max: {labels_array.max()}\n")
        f.write(f"  Std: {labels_array.std():.2f}\n")
    
    print("Statistics saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze BigEarthNet v2.0 dataset statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyze_bigearthnet.py --metadata metadata.parquet
  
  # Analysis with visualizations
  python analyze_bigearthnet.py --metadata metadata.parquet --output-dir ./analysis_output
  
  # Save statistics to file
  python analyze_bigearthnet.py --metadata metadata.parquet --save-stats stats.txt
        """
    )
    
    parser.add_argument(
        '--metadata',
        type=str,
        required=True,
        help='Path to metadata.parquet file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save visualization plots (optional)'
    )
    
    parser.add_argument(
        '--save-stats',
        type=str,
        default=None,
        help='Path to save detailed statistics text file (optional)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip creating visualization plots'
    )
    
    args = parser.parse_args()
    
    # Check if metadata file exists
    if not os.path.exists(args.metadata):
        print(f"Error: Metadata file not found: {args.metadata}")
        sys.exit(1)
    
    # Load metadata
    df = load_metadata(args.metadata)
    
    # Basic info
    print("\n" + "="*80)
    print("DATASET BASIC INFORMATION")
    print("="*80)
    print(f"Total patches: {len(df)}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Metadata columns: {list(df.columns)}")
    
    # Analyze splits
    split_analysis = analyze_splits(df)
    
    # Analyze labels
    label_analysis = analyze_labels(df)
    
    # Analyze by split
    split_label_analysis = analyze_by_split(df)
    
    # Co-occurrence analysis
    cooccurrence = analyze_class_cooccurrence(df, top_n=15)
    
    # Create visualizations
    if not args.no_plots:
        create_visualizations(df, output_dir=args.output_dir)
    
    # Save statistics
    if args.save_stats:
        save_statistics(df, args.save_stats)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()


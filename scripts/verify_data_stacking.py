"""
Verification script to confirm that data is stacked according to band order.

This script verifies that:
1. BENv2DataSet uses channel_configurations to determine band order
2. stack_and_interpolate stacks bands in the order specified
3. RGB channels (B04, B03, B02) are at indices [0, 1, 2]
"""
import sys
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from configilm.extra.BENv2_utils import STANDARD_BANDS, resolve_data_dir
from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet
from configilm.extra.DataModules.BENv2_DataModule import BENv2DataModule
from scripts.utils import get_benv2_dir_dict


def verify_data_stacking(include_s1: bool = False):
    """
    Verify that data is stacked according to the band order in channel_configurations.
    
    Args:
        include_s1: If False, only verify S2 data (RGB + non-RGB). If True, include S1 data.
    """
    print("=" * 80)
    print("VERIFYING DATA STACKING ORDER")
    print("=" * 80)
    
    if include_s1:
        print("Mode: S2 + S1 (multimodal)")
    else:
        print("Mode: S2 only (RGB + non-RGB)")
    print()
    
    # Get S2 bands
    s2_bands = STANDARD_BANDS.get("S2", STANDARD_BANDS.get("s2_full", []))
    rgb_bands = ["B04", "B03", "B02"]
    s2_non_rgb = [b for b in s2_bands if b not in rgb_bands]
    s2_ordered = rgb_bands + s2_non_rgb
    
    if include_s1:
        s1_bands = STANDARD_BANDS.get("S1", [])
        test_bands = s2_ordered + s1_bands
        config_name = "Multimodal (S2 ordered + S1)"
    else:
        s1_bands = []
        test_bands = s2_ordered
        config_name = "S2 only (RGB + non-RGB)"
    
    num_channels = len(test_bands)
    
    print(f"Expected band order: {test_bands}")
    print(f"Number of channels: {num_channels}")
    print(f"  - RGB bands (indices 0-2): {test_bands[:3]}")
    print(f"  - S2 non-RGB ({len(s2_ordered)-3} channels): {test_bands[3:len(s2_ordered)]}")
    if include_s1:
        print(f"  - S1 bands (last 2): {test_bands[-2:]}")
    print()
    
    # Register configuration
    STANDARD_BANDS[num_channels] = test_bands
    BENv2DataSet.channel_configurations[num_channels] = test_bands
    BENv2DataSet.avail_chan_configs[num_channels] = config_name
    
    print(f"Registered channel_configurations[{num_channels}] = {test_bands}")
    print()
    
    # Get data directories
    hostname, data_dirs = get_benv2_dir_dict()
    data_dirs = resolve_data_dir(data_dirs, allow_mock=True)
    
    print(f"Using data directories: {data_dirs}")
    print()
    
    # Create data module
    dm = BENv2DataModule(
        data_dirs=data_dirs,
        batch_size=1,
        num_workers_dataloader=0,
        img_size=(num_channels, 120, 120),
    )
    
    # Setup dataloader
    dm.setup("fit")
    
    # Get a sample
    print("Loading a sample from the dataloader...")
    train_loader = dm.train_dataloader()
    sample = next(iter(train_loader))
    x, y = sample
    
    print(f"Loaded tensor shape: {x.shape}")
    print(f"Expected shape: (1, {num_channels}, 120, 120)")
    print()
    
    # Verify channel count
    channel_count_match = x.shape[1] == num_channels
    print(f"Channel count verification: {'✓' if channel_count_match else '✗'}")
    print(f"  Tensor has {x.shape[1]} channels (expected {num_channels})")
    print()
    
    # Check dataset bands attribute
    dataset_bands = None
    if hasattr(dm.train_ds, 'bands'):
        dataset_bands = dm.train_ds.bands
        print(f"Dataset bands attribute: {dataset_bands}")
    elif hasattr(dm.train_ds, 'channel_configuration'):
        dataset_bands = dm.train_ds.channel_configuration
        print(f"Dataset channel_configuration: {dataset_bands}")
    else:
        # Try to find bands in dataset attributes
        print("Checking dataset internals...")
        if hasattr(dm.train_ds, '__dict__'):
            for key, value in dm.train_ds.__dict__.items():
                if 'band' in key.lower() and isinstance(value, list):
                    if len(value) == num_channels:
                        dataset_bands = value
                        print(f"  Found: {key} = {value}")
                        break
    
    print()
    
    # Verify band order
    if dataset_bands is not None:
        print("Band Order Verification:")
        print(f"  Expected: {test_bands}")
        print(f"  Dataset:  {dataset_bands}")
        
        if dataset_bands == test_bands:
            print("  ✓ BAND ORDER MATCHES!")
            print("  ✓ Data is stacked according to channel_configurations order")
            print("  ✓ RGB channels (B04, B03, B02) are at indices [0, 1, 2]")
            print("  ✓ rgb_data = x[:, :3, :, :] will correctly extract RGB")
            order_match = True
        else:
            print("  ✗ BAND ORDER MISMATCH!")
            print("  Differences:")
            for i, (exp, act) in enumerate(zip(test_bands, dataset_bands)):
                if exp != act:
                    print(f"    Channel {i}: expected {exp}, got {act}")
            order_match = False
    else:
        print("Band Order Verification:")
        print("  ⚠ Cannot directly verify - dataset bands attribute not accessible")
        print("  However, based on configilm library implementation:")
        print("    - BENv2DataSet uses channel_configurations to get bands list")
        print("    - stack_and_interpolate uses order=bands parameter")
        print("    - Bands are stacked in the order of the bands list")
        print("  Therefore, if channel count matches, order should be correct.")
        order_match = None
    
    # ============================================================================
    # VERIFY RGB EXTRACTION BY COMPARING WITH DIRECT RGB LOADING
    # ============================================================================
    print()
    print("=" * 80)
    print("RGB EXTRACTION VERIFICATION")
    print("=" * 80)
    print("\nMethod: Compare RGB extracted from multi-channel data vs direct RGB loading")
    print("-" * 80)
    
    # Step 1: Extract RGB from the multi-channel data (as done in Lightning module)
    rgb_extracted = x[:, :3, :, :]
    print(f"\n1. Extracted RGB from multi-channel data:")
    print(f"   Shape: {rgb_extracted.shape}")
    print(f"   Mean: {rgb_extracted.mean().item():.4f}, Std: {rgb_extracted.std().item():.4f}")
    
    # Step 2: Load RGB data directly (only RGB bands)
    print(f"\n2. Loading RGB data directly (only B04, B03, B02)...")
    
    # Register RGB-only configuration
    rgb_only_bands = ["B04", "B03", "B02"]
    rgb_num_channels = 3
    STANDARD_BANDS[rgb_num_channels] = rgb_only_bands
    BENv2DataSet.channel_configurations[rgb_num_channels] = rgb_only_bands
    BENv2DataSet.avail_chan_configs[rgb_num_channels] = "RGB only (B04, B03, B02)"
    
    # Create a new data module for RGB-only data
    dm_rgb = BENv2DataModule(
        data_dirs=data_dirs,
        batch_size=1,
        num_workers_dataloader=0,
        img_size=(rgb_num_channels, 120, 120),
    )
    dm_rgb.setup("fit")
    
    # Get the same sample (using same index)
    train_loader_rgb = dm_rgb.train_dataloader()
    # Get the same sample by iterating to the same position
    # Note: We'll use the first sample, which should be the same if data is deterministic
    sample_rgb = next(iter(train_loader_rgb))
    x_rgb, y_rgb = sample_rgb
    
    print(f"   Shape: {x_rgb.shape}")
    print(f"   Mean: {x_rgb.mean().item():.4f}, Std: {x_rgb.std().item():.4f}")
    
    # Step 3: Compare the two RGB tensors
    print(f"\n3. Comparing extracted RGB vs direct RGB loading:")
    print("-" * 80)
    
    # Check shapes
    shape_match = rgb_extracted.shape == x_rgb.shape
    print(f"   Shape match: {'✓' if shape_match else '✗'} (extracted: {rgb_extracted.shape}, direct: {x_rgb.shape})")
    
    if shape_match:
        # Compare values
        if torch.allclose(rgb_extracted, x_rgb, atol=1e-5):
            print(f"   ✓✓✓ VALUES MATCH EXACTLY! ✓✓✓")
            print(f"   ✓ RGB extraction is CORRECT")
            print(f"   ✓ Channel order is CORRECT (B04, B03, B02 at indices 0, 1, 2)")
            rgb_extraction_verified = True
        else:
            # Check if difference is due to normalization/transforms
            diff = torch.abs(rgb_extracted - x_rgb)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            print(f"   Values differ:")
            print(f"     Max difference: {max_diff:.6f}")
            print(f"     Mean difference: {mean_diff:.6f}")
            
            if max_diff < 0.01:  # Very small difference, likely numerical precision
                print(f"   ⚠ Very small difference (likely numerical precision)")
                print(f"   ✓ RGB extraction is LIKELY CORRECT")
                rgb_extraction_verified = True
            elif max_diff < 0.1:  # Small difference, might be transforms
                print(f"   ⚠ Small difference (may be due to transforms/normalization)")
                print(f"   ⚠ RGB extraction is PROBABLY CORRECT")
                rgb_extraction_verified = None
            else:
                print(f"   ✗ Significant difference detected")
                print(f"   ✗ RGB extraction may be INCORRECT")
                rgb_extraction_verified = False
    else:
        print(f"   ✗ Shape mismatch - cannot compare values")
        rgb_extraction_verified = False
    
    
    # ============================================================================
    # FINAL VERIFICATION SUMMARY
    # ============================================================================
    print()
    print("=" * 80)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 80)
    
    # Combine verification results
    verification_passed = []
    verification_failed = []
    verification_warnings = []
    
    if channel_count_match:
        verification_passed.append("Channel count matches")
    else:
        verification_failed.append("Channel count mismatch")
    
    if order_match is True:
        verification_passed.append("Band order matches expected configuration")
    elif order_match is False:
        verification_failed.append("Band order does NOT match")
    else:
        verification_warnings.append("Cannot directly verify band order")
    
    # RGB extraction verification (the key check)
    if rgb_extraction_verified is True:
        verification_passed.append("RGB extraction verified (extracted RGB matches direct RGB loading)")
    elif rgb_extraction_verified is False:
        verification_failed.append("RGB extraction FAILED (extracted RGB does NOT match direct RGB)")
    else:
        verification_warnings.append("RGB extraction uncertain (small differences detected)")
    
    # Print summary
    if verification_passed:
        print("\n✓ PASSED CHECKS:")
        for check in verification_passed:
            print(f"  ✓ {check}")
    
    if verification_warnings:
        print("\n⚠ WARNINGS:")
        for warning in verification_warnings:
            print(f"  ⚠ {warning}")
    
    if verification_failed:
        print("\n✗ FAILED CHECKS:")
        for failure in verification_failed:
            print(f"  ✗ {failure}")
    
    # Final verdict
    print()
    if rgb_extraction_verified is True:
        print("✓✓✓ RGB EXTRACTION VERIFIED ✓✓✓")
        print("  ✓ Extracted RGB from multi-channel data matches direct RGB loading")
        print("  ✓ rgb_data = x[:, :3, :, :] correctly extracts RGB channels")
        print("  ✓ Channel order is CORRECT (B04, B03, B02 at indices 0, 1, 2)")
        print("  ✓ Pipeline is ready for training!")
        extraction_verified = True
    elif rgb_extraction_verified is False:
        print("✗✗✗ RGB EXTRACTION VERIFICATION FAILED ✗✗✗")
        print("  ✗ Extracted RGB does NOT match direct RGB loading")
        print("  ✗ rgb_data = x[:, :3, :, :] may extract wrong channels")
        print("  ✗ Please check channel_configurations registration")
        extraction_verified = False
    else:
        print("⚠⚠⚠ RGB EXTRACTION UNCERTAIN ⚠⚠⚠")
        print("  ⚠ Small differences detected (may be due to transforms/normalization)")
        print("  ⚠ RGB extraction is PROBABLY correct, but cannot be fully verified")
        extraction_verified = None
    
    print("=" * 80)
    
    # Return True only if RGB extraction is verified
    return (channel_count_match and rgb_extraction_verified is True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify data stacking order")
    parser.add_argument(
        "--include-s1",
        action="store_true",
        help="Include S1 data in verification (default: S2 only)"
    )
    args = parser.parse_args()
    
    try:
        success = verify_data_stacking(include_s1=args.include_s1)
        if success:
            print("\n✓ Verification passed!")
        else:
            print("\n✗ Verification failed!")
    except Exception as e:
        print(f"\n✗ Error during verification: {e}")
        import traceback
        traceback.print_exc()


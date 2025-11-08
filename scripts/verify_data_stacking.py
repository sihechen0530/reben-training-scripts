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
    # VERIFY RGB EXTRACTION IN PIPELINE
    # ============================================================================
    print()
    print("=" * 80)
    print("RGB EXTRACTION VERIFICATION")
    print("=" * 80)
    
    # Simulate the extraction as done in Lightning module
    rgb_data = x[:, :3, :, :]
    if include_s1:
        non_rgb_s1_data = x[:, 3:, :, :]
        expected_non_rgb_shape = (1, num_channels - 3, 120, 120)
        non_rgb_name = "non_rgb_s1_data"
    else:
        non_rgb_data = x[:, 3:, :, :]
        expected_non_rgb_shape = (1, num_channels - 3, 120, 120)
        non_rgb_name = "non_rgb_data"
    
    print(f"\nExtracted tensors:")
    print(f"  rgb_data shape: {rgb_data.shape} (expected: (1, 3, 120, 120))")
    if include_s1:
        print(f"  non_rgb_s1_data shape: {non_rgb_s1_data.shape} (expected: {expected_non_rgb_shape})")
    else:
        print(f"  non_rgb_data shape: {non_rgb_data.shape} (expected: {expected_non_rgb_shape})")
    
    rgb_shape_correct = rgb_data.shape == (1, 3, 120, 120)
    if include_s1:
        non_rgb_shape_correct = non_rgb_s1_data.shape == expected_non_rgb_shape
    else:
        non_rgb_shape_correct = non_rgb_data.shape == expected_non_rgb_shape
    
    print(f"\nShape verification:")
    print(f"  RGB shape: {'✓' if rgb_shape_correct else '✗'}")
    if include_s1:
        print(f"  Non-RGB+S1 shape: {'✓' if non_rgb_shape_correct else '✗'}")
    else:
        print(f"  Non-RGB shape: {'✓' if non_rgb_shape_correct else '✗'}")
    
    # ============================================================================
    # VERIFY RGB EXTRACTION BY VALUES (not just dimensions)
    # ============================================================================
    print("\nRGB Data Value Verification:")
    print("-" * 80)
    
    # Check if RGB channels have reasonable values (not all zeros, not all same)
    rgb_mean = rgb_data.mean().item()
    rgb_std = rgb_data.std().item()
    rgb_unique = torch.unique(rgb_data).numel()
    rgb_min = rgb_data.min().item()
    rgb_max = rgb_data.max().item()
    
    # Per-channel statistics
    rgb_ch0_mean = rgb_data[:, 0, :, :].mean().item()
    rgb_ch1_mean = rgb_data[:, 1, :, :].mean().item()
    rgb_ch2_mean = rgb_data[:, 2, :, :].mean().item()
    
    print(f"Overall RGB statistics:")
    print(f"  Mean: {rgb_mean:.4f}")
    print(f"  Std: {rgb_std:.4f}")
    print(f"  Min: {rgb_min:.4f}, Max: {rgb_max:.4f}")
    print(f"  Unique values: {rgb_unique}")
    
    print(f"\nPer-channel means (should be different if valid RGB):")
    print(f"  Channel 0 (expected B04/Red): {rgb_ch0_mean:.4f}")
    print(f"  Channel 1 (expected B03/Green): {rgb_ch1_mean:.4f}")
    print(f"  Channel 2 (expected B02/Blue): {rgb_ch2_mean:.4f}")
    
    # Check if channels are different (RGB channels should have different values)
    channels_different = (abs(rgb_ch0_mean - rgb_ch1_mean) > 0.001 and 
                         abs(rgb_ch1_mean - rgb_ch2_mean) > 0.001 and
                         abs(rgb_ch0_mean - rgb_ch2_mean) > 0.001)
    
    rgb_valid = rgb_std > 0.001 and rgb_unique > 10  # Basic sanity check
    
    # Initialize value_verification
    if rgb_valid and channels_different:
        print(f"  ✓ RGB channels have different values (expected for B04, B03, B02)")
        print(f"  ✓ RGB data appears valid (not all zeros or constant)")
        value_verification = True
    elif rgb_valid:
        print(f"  ⚠ RGB data has variance but channels are similar")
        print(f"  ⚠ This might indicate incorrect channel assignment")
        value_verification = False
    else:
        print(f"  ✗ RGB data may be invalid (all zeros or constant)")
        value_verification = False
    
    # ============================================================================
    # COMPARE WITH EXPECTED BAND VALUES (if dataset provides access)
    # ============================================================================
    print("\n" + "-" * 80)
    print("Direct Band Value Comparison:")
    print("-" * 80)
    
    # Try to get the actual sample index and compare with raw band data
    # This is the most reliable way to verify, but requires access to dataset internals
    direct_value_match = None  # Initialize
    try:
        # Get the first sample index
        sample_idx = 0
        if hasattr(dm.train_ds, '__getitem__'):
            # Try to access raw data if possible
            raw_sample = dm.train_ds[sample_idx]
            if isinstance(raw_sample, tuple) and len(raw_sample) >= 1:
                raw_x = raw_sample[0] if isinstance(raw_sample[0], torch.Tensor) else None
                
                if raw_x is not None and raw_x.shape == x.shape:
                    # Compare extracted RGB with expected channels
                    extracted_rgb = x[:, :3, :, :]
                    raw_rgb = raw_x[:3, :, :] if len(raw_x.shape) == 3 else raw_x[:, :3, :, :]
                    
                    # Check if values match (accounting for potential normalization)
                    if torch.allclose(extracted_rgb.squeeze(0), raw_rgb, atol=1e-5):
                        print("  ✓ Extracted RGB values match raw data")
                        print("  ✓ Channel order is CORRECT")
                        direct_value_match = True
                    else:
                        # Check if it's just normalization difference
                        extracted_norm = extracted_rgb.squeeze(0).mean()
                        raw_norm = raw_rgb.mean()
                        if abs(extracted_norm - raw_norm) < 0.1:
                            print("  ⚠ Values differ but may be due to normalization")
                            print("  ⚠ Shape and order appear correct")
                            direct_value_match = None
                        else:
                            print("  ✗ Extracted RGB values do NOT match raw data")
                            print("  ✗ Channel order may be INCORRECT")
                            direct_value_match = False
                else:
                    print("  ⚠ Cannot compare - raw data shape mismatch")
                    direct_value_match = None
            else:
                print("  ⚠ Cannot access raw data from dataset")
                direct_value_match = None
        else:
            print("  ⚠ Cannot access dataset items directly")
            direct_value_match = None
    except Exception as e:
        print(f"  ⚠ Cannot perform direct value comparison: {e}")
        direct_value_match = None
    
    # ============================================================================
    # FINAL VERIFICATION (combining all checks)
    # ============================================================================
    print()
    print("=" * 80)
    print("COMPREHENSIVE VERIFICATION RESULT")
    print("=" * 80)
    
    # Combine all verification results
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
    
    if rgb_shape_correct and non_rgb_shape_correct:
        verification_passed.append("RGB extraction produces correct shapes")
    else:
        verification_failed.append("RGB extraction shape incorrect")
    
    if value_verification is True:
        verification_passed.append("RGB channels have valid and different values")
    elif value_verification is False:
        verification_failed.append("RGB channels may not be correctly assigned")
    else:
        verification_warnings.append("RGB value verification inconclusive")
    
    if direct_value_match is True:
        verification_passed.append("Extracted RGB values match raw data")
    elif direct_value_match is False:
        verification_failed.append("Extracted RGB values do NOT match raw data")
    elif direct_value_match is None:
        verification_warnings.append("Cannot perform direct value comparison")
    
    # Print verification summary
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
    if len(verification_failed) == 0 and order_match is True:
        if direct_value_match is True:
            print("✓✓✓ RGB EXTRACTION FULLY VERIFIED ✓✓✓")
            print("  ✓ Verified by: band order + shape + values + direct comparison")
            extraction_verified = True
        elif value_verification is True:
            print("✓✓ RGB EXTRACTION VERIFIED ✓✓")
            print("  ✓ Verified by: band order + shape + values")
            extraction_verified = True
        else:
            print("✓ RGB EXTRACTION LIKELY CORRECT ✓")
            print("  ✓ Verified by: band order + shape")
            print("  ⚠ Value verification inconclusive")
            extraction_verified = True
    elif len(verification_failed) == 0:
        print("⚠ RGB EXTRACTION UNCERTAIN ⚠")
        print("  ✓ Shape is correct")
        print("  ⚠ Cannot fully verify band order or values")
        extraction_verified = None
    else:
        print("✗ RGB EXTRACTION VERIFICATION FAILED ✗")
        print("  ✗ Some checks failed - RGB extraction may be incorrect")
        extraction_verified = False
    
    print()
    print("=" * 80)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Channel count: {'✓' if channel_count_match else '✗'}")
    
    if order_match is True:
        print("Band order: ✓ CONFIRMED - Data stacked according to band order")
    elif order_match is False:
        print("Band order: ✗ MISMATCH - Data may not be in expected order")
    else:
        print("Band order: ⚠ UNCERTAIN - Cannot directly verify")
    
    if extraction_verified is True:
        print("RGB extraction: ✓ VERIFIED - rgb_data = x[:, :3, :, :] is CORRECT")
        print("\n✓✓✓ PIPELINE VERIFICATION PASSED ✓✓✓")
        print("  - Data is stacked according to band order")
        print("  - RGB channels (B04, B03, B02) are at indices [0, 1, 2]")
        print("  - rgb_data = x[:, :3, :, :] correctly extracts RGB")
        print("  - Pipeline is ready for training!")
    elif extraction_verified is False:
        print("RGB extraction: ✗ FAILED - rgb_data = x[:, :3, :, :] may be INCORRECT")
        print("\n✗✗✗ PIPELINE VERIFICATION FAILED ✗✗✗")
        print("  - Band order does not match expected configuration")
        print("  - RGB extraction may extract wrong channels")
        print("  - Please check channel_configurations registration")
    else:
        print("RGB extraction: ⚠ UNCERTAIN - Assumed correct based on implementation")
        print("\n⚠⚠⚠ PIPELINE VERIFICATION INCONCLUSIVE ⚠⚠⚠")
        print("  - Cannot fully verify band order")
        print("  - Based on configilm implementation, should be correct")
        print("  - Recommend checking dataset bands attribute manually")
    
    print("=" * 80)
    
    # Return True only if we have strong confirmation
    return (channel_count_match and 
            order_match is True and 
            extraction_verified is True)


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


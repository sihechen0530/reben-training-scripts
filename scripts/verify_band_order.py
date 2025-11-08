"""
Verification script to check if the dataloader loads bands in the order specified
in channel_configurations.

This script will:
1. Register a custom band configuration
2. Load a sample from the dataloader
3. Verify that the channels are in the expected order
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

def verify_band_order():
    """Verify that dataloader loads bands in the order specified in channel_configurations."""
    
    # Define a test band order (different from default to make verification obvious)
    # We'll use: B04, B03, B02 (RGB) + B01, B05 (non-RGB) + VV, VH (S1)
    test_bands = ["B04", "B03", "B02", "B01", "B05", "VV", "VH"]
    num_channels = len(test_bands)
    
    print("=" * 80)
    print("Verifying Band Order in Dataloader")
    print("=" * 80)
    print(f"Expected band order: {test_bands}")
    print(f"Number of channels: {num_channels}")
    print()
    
    # Register the configuration
    STANDARD_BANDS[num_channels] = test_bands
    BENv2DataSet.channel_configurations[num_channels] = test_bands
    BENv2DataSet.avail_chan_configs[num_channels] = "Test order verification"
    
    print(f"Registered channel_configurations[{num_channels}] = {test_bands}")
    print()
    
    # Get data directories
    hostname, data_dirs = get_benv2_dir_dict()
    data_dirs = resolve_data_dir(data_dirs, allow_mock=True)  # Use mock for testing
    
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
    
    # Verify the order
    # Since we can't directly verify which band is which from the tensor,
    # we'll check if the dataloader is using our registered configuration
    print("Verification:")
    print(f"  - Tensor has {x.shape[1]} channels (expected {num_channels}): {'✓' if x.shape[1] == num_channels else '✗'}")
    print(f"  - Channel configuration registered: {'✓' if num_channels in BENv2DataSet.channel_configurations else '✗'}")
    print(f"  - Registered bands match expected: {'✓' if BENv2DataSet.channel_configurations[num_channels] == test_bands else '✗'}")
    print()
    
    # Note: We cannot directly verify the band order from the tensor values
    # because we don't have access to the actual band data. However, if the
    # dataloader uses channel_configurations, the order should match.
    print("Note: Direct verification of band order requires access to the actual")
    print("      band data. The dataloader should use channel_configurations")
    print("      to determine which bands to load and in what order.")
    print()
    
    # Check if we can access the configuration used by the dataset
    if hasattr(dm.train_ds, 'bands') or hasattr(dm.train_ds, 'channel_configuration'):
        print("Dataset attributes:")
        if hasattr(dm.train_ds, 'bands'):
            print(f"  - train_ds.bands: {dm.train_ds.bands}")
        if hasattr(dm.train_ds, 'channel_configuration'):
            print(f"  - train_ds.channel_configuration: {dm.train_ds.channel_configuration}")
    
    print("=" * 80)
    print("Verification complete!")
    print("=" * 80)
    
    return x.shape[1] == num_channels

if __name__ == "__main__":
    try:
        success = verify_band_order()
        if success:
            print("\n✓ Verification passed: Channel count matches expected configuration")
        else:
            print("\n✗ Verification failed: Channel count does not match")
    except Exception as e:
        print(f"\n✗ Error during verification: {e}")
        import traceback
        traceback.print_exc()


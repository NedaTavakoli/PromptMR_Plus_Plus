"""
Test script for verifying data types in CMR2025 dataset
This script tests the dataset loading functionality and explicitly checks data types
"""
import os
import sys
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.getcwd())

# Try to import the CMR2025Dataset
try:
    from data.cmr2025 import CMR2025Dataset
    print("Successfully imported CMR2025Dataset!")
except ImportError as e:
    print(f"Error importing CMR2025Dataset: {e}")
    sys.exit(1)

# Find data files
train_path = './h5_dataset_simplified/train'
val_path = './h5_dataset_simplified/val'

print(f"Looking for files in {train_path}")
train_files = sorted(glob.glob(os.path.join(train_path, '*.h5')))
print(f"Found {len(train_files)} training files")

if len(train_files) == 0:
    print("No training files found. Checking current directory structure:")
    try:
        print(f"Current directory: {os.getcwd()}")
        if os.path.exists('./h5_dataset_simplified'):
            print("h5_dataset_simplified directory exists")
            print(f"Contents: {os.listdir('./h5_dataset_simplified')}")
        else:
            print("h5_dataset_simplified directory does not exist")
    except Exception as e:
        print(f"Error checking directories: {e}")
    sys.exit(1)

# Create the dataset
dataset = CMR2025Dataset(
    files=train_files[:3],  # Use only first 3 files for debugging
    acceleration=8,
    crop_size=(320, 320),
    mode='train',
    sample_rate=1.0
)

print(f"Dataset size: {len(dataset)}")

# Try to load a few samples
num_samples = min(2, len(dataset))
for i in range(num_samples):
    try:
        print(f"\n{'-'*50}")
        print(f"Loading sample {i}:")
        sample = dataset[i]
        
        # Print sample information
        print(f"File: {os.path.basename(sample['fname'])}")
        print(f"Slice ID: {sample['slice_id']}")
        
        # Print tensor shapes and data types in detail
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}:")
                print(f"  - Shape: {value.shape}")
                print(f"  - Data type: {value.dtype}")
                print(f"  - Device: {value.device}")
                print(f"  - Min/Max: {value.min().item():.4f}/{value.max().item():.4f}")
                
                # Check for NaN or Inf values
                has_nan = torch.isnan(value).any()
                has_inf = torch.isinf(value).any()
                if has_nan or has_inf:
                    print(f"  - WARNING: {'NaN ' if has_nan else ''}{'Inf ' if has_inf else ''}values detected")
                
                # Check complex structure for kspace
                if key == 'kspace':
                    print(f"  - Last dimension size: {value.shape[-1]}")
                    if value.shape[-1] == 2:
                        print("  - Complex representation detected (last dim = 2)")
                        # Check ratio of real to imaginary part magnitude
                        real_mag = torch.abs(value[..., 0]).mean().item()
                        imag_mag = torch.abs(value[..., 1]).mean().item()
                        print(f"  - Real/Imag magnitude ratio: {real_mag:.4f}/{imag_mag:.4f}")
        
        # Plot the target image and mask
        plt.figure(figsize=(15, 5))
        
        # Target image
        plt.subplot(1, 3, 1)
        if 'target' in sample:
            plt.imshow(sample['target'][0].numpy(), cmap='gray')
            plt.title(f"Target Image (Sample {i})")
            plt.colorbar()
        
        # Mask
        plt.subplot(1, 3, 2)
        if 'mask' in sample:
            plt.imshow(sample['mask'].numpy(), cmap='gray')
            plt.title(f"Sampling Mask (Acceleration {sample['acceleration']})")
            plt.colorbar()
        
        # K-space magnitude (sum of real and imag squares)
        plt.subplot(1, 3, 3)
        if 'kspace' in sample and sample['kspace'].shape[-1] == 2:
            kspace_mag = torch.sqrt(sample['kspace'][0, 0, :, :, 0]**2 + sample['kspace'][0, 0, :, :, 1]**2)
            plt.imshow(np.log(kspace_mag.numpy() + 1e-9), cmap='viridis')
            plt.title(f"K-space Magnitude (log scale)")
            plt.colorbar()
        
        # Save figure
        os.makedirs('debug_plots', exist_ok=True)
        plt.savefig(f"debug_plots/datatypes_sample_{i}.png")
        plt.close()
        print(f"Saved plot to debug_plots/datatypes_sample_{i}.png")
        
    except Exception as e:
        print(f"Error loading sample {i}: {e}")
        import traceback
        traceback.print_exc()

print("\nTest completed!")

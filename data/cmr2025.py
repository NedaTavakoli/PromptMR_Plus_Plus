import os
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import lightning.pytorch as pl  # Make sure to use this import

# Import the necessary functions for the patched forward method
from mri_utils.coil_combine import sens_expand, sens_reduce

# In data/cmr2025.py (fork)
from mri_utils import rss, fft2c, ifft2c, complex_center_crop, complex_random_crop, get_mask_func 
# (import only the functions that exist, omitting 'sense')
# ... define CMR2025DataModule that computes coil sensitivity maps or RSS internally ...


def complex_center_crop(data, crop_size):
    """
    Apply center crop to complex-valued tensor, with handling for crop sizes larger than input.
    
    Args:
        data: Input complex data
        crop_size: Desired crop size
        
    Returns:
        Cropped data
    """
    if not isinstance(crop_size, (list, tuple)):
        crop_size = (crop_size, crop_size)
    
    # Get dimensions   
    h, w = data.shape[-2], data.shape[-1]
    
    # Check if crop size is larger than input size and adjust
    crop_h = min(h, crop_size[0])
    crop_w = min(w, crop_size[1])
    
    # Calculate cropping indices
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    
    # Perform the crop based on tensor dimensionality
    if len(data.shape) == 3:  # 2D case
        return data[start_h:start_h + crop_h, start_w:start_w + crop_w, :]
    elif len(data.shape) == 4:  # 2D with batch
        return data[:, start_h:start_h + crop_h, start_w:start_w + crop_w, :]
    elif len(data.shape) == 5:  # 3D case
        return data[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w, :]
    elif len(data.shape) == 6:  # 3D with batch
        return data[:, :, :, start_h:start_h + crop_h, start_w:start_w + crop_w, :]
    else:
        raise ValueError(f"Unsupported tensor shape: {data.shape}")

def complex_abs(data):
    """
    Compute the absolute value of a complex tensor.
    
    Args:
        data: Complex data with last dimension being 2 (real and imaginary)
        
    Returns:
        Absolute value of complex data
    """
    assert data.size(-1) == 2
    return torch.sqrt((data ** 2).sum(dim=-1))

def create_mask(shape, acc_factor):
    """
    Create a 1D sampling mask for given acceleration factor.
    
    Args:
        shape: Shape of the mask
        acc_factor: Acceleration factor
        
    Returns:
        Mask tensor
    """
    num_cols = shape[1]
    center_fraction = 0.08
    
    # Create the mask
    num_low_frequencies = int(round(num_cols * center_fraction))
    prob = (num_cols / acc_factor - num_low_frequencies) / (num_cols - num_low_frequencies)
    mask = np.random.random(num_cols) < prob
    
    # Always include the center
    pad = (num_cols - num_low_frequencies + 1) // 2
    mask[pad:pad + num_low_frequencies] = True
    
    # Reshape the mask
    mask_shape = [1 for _ in shape]
    mask_shape[1] = num_cols
    mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
    
    # Create mask for complex data (both real and imaginary)
    mask = torch.cat([mask, mask], dim=-1)
    
    return mask

def normalize(data, mean=None, std=None, eps=0.0):
    """
    Normalize complex data by mean and standard deviation.
    
    Args:
        data: Input data
        mean: Mean value (if None, computed from data)
        std: Standard deviation (if None, computed from data)
        eps: Small constant to avoid division by zero
        
    Returns:
        Normalized data, mean, std
    """
    if mean is None:
        mean = data.mean()
    if std is None:
        std = data.std()
    
    return (data - mean) / (std + eps), mean, std

def create_batch(kspace, mask, sensitivity_maps, target, attrs, crop_size):
    """
    Create a batch from raw data components, ensuring no None values.
    """
    batch = {
        'kspace': kspace,
        'masked_kspace': kspace,  # Include both keys to be safe
        'mask': mask,
        'mask_type': 'cartesian',  # Always include a valid mask_type
        'attrs': attrs if attrs is not None else {},
        'crop_size': crop_size
    }
    
    # Add low frequencies if available in attrs
    if attrs and 'num_low_frequencies' in attrs:
        batch['num_low_frequencies'] = attrs['num_low_frequencies']
    else:
        # A reasonable default, adjust as needed
        batch['num_low_frequencies'] = 8
    
    # Only add target if it's not None
    if target is not None:
        batch['target'] = target
    
    # Only add sensitivity_maps if it's not None
    if sensitivity_maps is not None:
        batch['sensitivity_maps'] = sensitivity_maps
    
    return batch

def apply_mask_to_kspace(kspace, mask):
    """
    Apply a sampling mask to kspace data with proper broadcasting.
    
    Args:
        kspace: K-space data with shape [batch, coils, height, width, complex]
        mask: Sampling mask
        
    Returns:
        Masked k-space data
    """
    print(f"Inside apply_mask_to_kspace - kspace: {kspace.shape}, mask: {mask.shape}")
    
    # Get the shapes
    kspace_shape = kspace.shape
    mask_shape = mask.shape
    
    try:
        # For mask with shape [1, 10, 1, 2] and kspace with shape [batch, 10, height, width, 2]
        if len(mask_shape) == 4 and len(kspace_shape) == 5:
            if mask_shape[2] == 1:  # Height is 1 in mask
                # Expand mask to match kspace dimensions
                expanded_mask = mask.expand(-1, -1, kspace_shape[2], -1)  # Expand height
                # Now mask is [1, 10, height, 2]
                
                # Further reshape if needed
                if kspace_shape[3] > 1 and expanded_mask.shape[3] == 2:
                    # We need to expand the width dimension too
                    expanded_mask = expanded_mask.unsqueeze(3).expand(-1, -1, -1, kspace_shape[3], -1)
                    # Now mask is [1, 10, height, width, 2]
                
                # Finally expand batch if needed
                if kspace_shape[0] > 1 and expanded_mask.shape[0] == 1:
                    expanded_mask = expanded_mask.expand(kspace_shape[0], -1, -1, -1, -1)
                
                print(f"Expanded mask shape: {expanded_mask.shape}")
                return kspace * expanded_mask
        
        # If we got here, try a simpler approach - direct multiplication with broadcasting
        return kspace * mask
        
    except RuntimeError as e:
        print(f"Error in masking: {e}")
        
        # Create a new mask with the same shape as kspace
        new_mask = torch.ones_like(kspace)
        
        # Copy the mask values into the appropriate dimensions
        for i in range(kspace_shape[0]):  # Batch dimension
            for j in range(min(kspace_shape[1], mask_shape[1])):  # Coil dimension (limited by smaller one)
                # Get the mask value for this coil
                mask_val = mask[0, j, 0, :] if mask_shape[0] == 1 else mask[i, j, 0, :]
                
                # Apply it across all spatial dimensions
                new_mask[i, j, :, :, :] = mask_val.view(1, 1, 2)
        
        print(f"Created custom mask with shape: {new_mask.shape}")
        return kspace * new_mask

def convert_paths_to_strings(item):
    """
    Recursively convert Path objects to strings in a nested data structure.
    
    Args:
        item: The item to convert (can be a dict, list, Path, or other type)
        
    Returns:
        The converted item with Path objects replaced by strings
    """
    if isinstance(item, (Path, type(Path()))):
        return str(item)
    elif isinstance(item, dict):
        return {k: convert_paths_to_strings(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [convert_paths_to_strings(x) for x in item]
    elif isinstance(item, tuple):
        return tuple(convert_paths_to_strings(x) for x in item)
    else:
        return item



class CMR2025Dataset(Dataset):
    """
    Dataset for CMR 2025 challenge.
    """
    
    def __init__(
        self,
        data_path,
        transform=None,
        challenge='multicoil',
        sample_rate=1.0,
        volume_sample_rate=None,
        use_dataset_cache=True,
        crop_size=(320, 190),  # Adjusted to be smaller than smallest dimension
        seed=42,
        train_accelerations=[4, 8, 12],
        pad_sides=False,
        **kwargs
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset
            transform: Optional transform
            challenge: Type of challenge
            sample_rate: Fraction of slices to include
            volume_sample_rate: Fraction of volumes to include
            use_dataset_cache: Whether to use dataset cache
            crop_size: Size to crop input data
            seed: Random seed
            train_accelerations: List of acceleration factors for training
            pad_sides: Whether to pad sides of kspace
            **kwargs: Additional arguments
        """
        self.data_path = data_path
        self.transform = transform
        self.challenge = challenge
        self.sample_rate = sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.use_dataset_cache = use_dataset_cache
        self.crop_size = crop_size
        self.seed = seed
        self.train_accelerations = train_accelerations
        self.pad_sides = pad_sides
        self.recons_key = 'reconstruction_esc' if challenge == 'multicoil' else 'reconstruction_rss'
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Get file paths
        self.examples = []
        
        if not Path(data_path).exists():
            raise ValueError(f"Data path {data_path} does not exist")
        
        # Find all h5 files in the data directory
        files = list(Path(data_path).glob('*.h5'))
        
        # Apply volume sampling if needed
        if volume_sample_rate is not None and volume_sample_rate < 1.0:
            random.shuffle(files)
            num_files = round(len(files) * volume_sample_rate)
            files = files[:num_files]
        
        # Process each file
        for fname in sorted(files):
            with h5py.File(fname, 'r') as hf:
                num_slices = hf['kspace'].shape[0]
                self.examples += [(fname, slice_id) for slice_id in range(num_slices)]
        
        # Apply slice sampling if needed
        if sample_rate < 1.0:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
        
        print(f"Found {len(self.examples)} examples in {data_path}")
    
    def __len__(self):
        """Return the number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Get a dataset example."""
        fname, slice_id = self.examples[idx]
        
        # Initialize sensitivity_maps to None by default
        sensitivity_maps = None
        
        # Load the data
        with h5py.File(fname, 'r') as hf:
            # Load kspace data
            kspace = hf['kspace'][slice_id]
            kspace = torch.from_numpy(kspace).to(torch.float32)  # Ensure it's float32
            
            # Load target image if available
            if self.recons_key in hf:
                target = hf[self.recons_key][slice_id]
                target = torch.from_numpy(target).to(torch.float32)  # Ensure it's float32
                if len(target.shape) == 2:  # Add channel dimension if needed
                    target = target.unsqueeze(0)
            else:
                target = None
            
            # Load attributes
            attrs = dict(hf.attrs) if hasattr(hf, 'attrs') else {}
            
            # Load masks or create if not available
            if 'mask' in hf:
                mask = hf['mask'][slice_id]
                mask = torch.from_numpy(mask).to(torch.float32)  # Ensure it's float32
            else:
                # Create mask based on acceleration factor
                accel = random.choice(self.train_accelerations)
                mask = create_mask(kspace.shape, accel)
            
            # Load sensitivity maps if available
            if 'sensitivity_maps' in hf:
                sensitivity_maps = hf['sensitivity_maps'][slice_id]
                sensitivity_maps = torch.from_numpy(sensitivity_maps).to(torch.float32)  # Ensure it's float32
        
            # Print shapes for debugging
            print(f"kspace shape: {kspace.shape}")
            print(f"mask_tensor shape: {mask.shape}")
            
            # Check if data is already complex (ComplexDouble or ComplexFloat)
            if torch.is_complex(kspace):
                # Convert to real tensor with last dimension of size 2
                kspace = torch.stack([kspace.real, kspace.imag], dim=-1).to(torch.float32)
                print(f"Converted complex tensor to real tensor with complex dim: {kspace.shape}")
            # Ensure kspace has complex dimension (last dim of size 2)
            elif len(kspace.shape) == 0 or kspace.shape[-1] != 2:
                zeros = torch.zeros_like(kspace)
                kspace = torch.stack([kspace, zeros], dim=-1)
                print(f"Added complex dimension to kspace: {kspace.shape}")
            
            # Check if mask is complex
            if torch.is_complex(mask):
                # Convert to real tensor with last dimension of size 2
                mask = torch.stack([mask.real, mask.imag], dim=-1).to(torch.float32)
                print(f"Converted complex mask to real tensor with complex dim: {mask.shape}")
            # Ensure mask has complex dimension if it doesn't already
            elif len(mask.shape) > 0 and mask.shape[-1] != 2:
                mask_real = mask  # Original mask values become real part
                mask_imag = torch.zeros_like(mask)  # Zeros for imaginary part
                mask = torch.stack([mask_real, mask_imag], dim=-1)
                print(f"Added complex dimension to mask: {mask.shape}")
            
            # Apply mask to kspace
            kspace = apply_mask_to_kspace(kspace, mask)
            
            # Normalize the data if target is available
            if target is not None:
                target, mean, std = normalize(target)
            else:
                mean, std = None, None
            
            # Crop to desired size
            if self.crop_size is not None:
                kspace = complex_center_crop(kspace, self.crop_size)
                if target is not None:
                    # Handle target cropping
                    if len(target.shape) > 0 and target.shape[-1] != 2:
                        target_complex = target.unsqueeze(-1).repeat(1, 1, 1, 2)
                    else:
                        target_complex = target
                    
                    target = complex_abs(complex_center_crop(target_complex, self.crop_size))
                    
                    # Remove extra dimensions if needed
                    if len(target.shape) > 3:
                        target = target.squeeze(-1)
                
                # Ensure sensitivity_maps exists before trying to crop it
                if sensitivity_maps is not None:
                    # Check if sensitivity_maps is complex
                    if torch.is_complex(sensitivity_maps):
                        sensitivity_maps = torch.stack([sensitivity_maps.real, sensitivity_maps.imag], dim=-1).to(torch.float32)
                    
                    # Handle sensitivity maps cropping
                    if len(sensitivity_maps.shape) > 0 and sensitivity_maps.shape[-1] != 2:
                        sens_complex = sensitivity_maps.unsqueeze(-1).repeat(1, 1, 1, 2)
                        sensitivity_maps = complex_center_crop(sens_complex, self.crop_size)
                    else:
                        sensitivity_maps = complex_center_crop(sensitivity_maps, self.crop_size)
            
            # Convert fname to string if it's a Path object
            fname_str = str(fname) if isinstance(fname, (Path, type(Path()))) else fname
            
            # Create batch dictionary
            batch = {
                'kspace': kspace,
                'masked_kspace': kspace,  # Include both keys to handle variations in model
                'mask': mask,
                'mask_type': 'cartesian',  # Always provide a valid mask_type
                'attrs': attrs if attrs is not None else {},
                'crop_size': self.crop_size,
                'fname': fname_str,  # Use string version of fname
                'slice_num': slice_id
            }
            
            # Add low frequencies if available in attrs
            if attrs and 'num_low_frequencies' in attrs:
                batch['num_low_frequencies'] = attrs['num_low_frequencies']
            else:
                # A reasonable default, adjust as needed
                batch['num_low_frequencies'] = 8
            
            # Add max_value for normalization
            if target is not None:
                max_value = attrs.get('max_value', target.max().item())
                batch['max_value'] = max_value
                batch['target'] = target
            else:
                batch['max_value'] = 1.0
            
            # Only add sensitivity_maps if it's not None
            if sensitivity_maps is not None:
                batch['sensitivity_maps'] = sensitivity_maps
            
            # Apply transform if available
            if self.transform:
                batch = self.transform(batch)
            
            return batch

class CMR2025DataModule(pl.LightningDataModule):
    """Data module for CMR 2025 challenge."""
    
    def __init__(
        self,
        data_path='./h5_dataset',
        train_path='./h5_dataset/train',
        val_path='./h5_dataset/val',
        test_path=None,
        challenge='multicoil',
        train_transform=None,
        val_transform=None,
        test_transform=None,
        sample_rate=1.0,
        volume_sample_rate=None,
        train_accelerations=[4, 8, 12],
        val_accelerations=[4, 8, 12],
        test_accelerations=None,
        crop_size=(320, 190),  # Adjusted to be smaller than smallest dimension
        batch_size=1,
        num_workers=4,
        distributed_sampler=False,
        use_dataset_cache=True,
        use_seed=True,
        seed=42,
        val_split=None,
        test_split=None,
        pad_sides=False,
        fix_acceleration_val=None,
        **kwargs
    ):
        """
        Initialize the data module.
        
        Args:
            data_path: Path to the dataset
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data
            challenge: Type of challenge
            train_transform: Transform for training data
            val_transform: Transform for validation data
            test_transform: Transform for test data
            sample_rate: Fraction of slices to include
            volume_sample_rate: Fraction of volumes to include
            train_accelerations: List of acceleration factors for training
            val_accelerations: List of acceleration factors for validation
            test_accelerations: List of acceleration factors for testing
            crop_size: Size to crop input data
            batch_size: Batch size
            num_workers: Number of workers for data loading
            distributed_sampler: Whether to use distributed sampler
            use_dataset_cache: Whether to use dataset cache
            use_seed: Whether to use fixed seed
            seed: Random seed
            val_split: Validation split
            test_split: Test split
            pad_sides: Whether to pad sides of kspace
            fix_acceleration_val: Fixed acceleration factor for validation
            **kwargs: Additional arguments
        """
        super().__init__()
        
        self.data_path = data_path
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.sample_rate = sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.train_accelerations = train_accelerations
        self.val_accelerations = val_accelerations
        self.test_accelerations = test_accelerations
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.use_dataset_cache = use_dataset_cache
        self.use_seed = use_seed
        self.seed = seed
        self.val_split = val_split
        self.test_split = test_split
        self.pad_sides = pad_sides
        self.fix_acceleration_val = fix_acceleration_val
    
    def prepare_data(self):
        """Prepare the data."""
        # This is called only on 1 GPU/TPU in distributed
        pass
    
    def setup(self, stage=None):
        """Set up the datasets."""
        # Called on every GPU/TPU
        
        if stage == 'fit' or stage is None:
            self.train_dataset = CMR2025Dataset(
                data_path=self.train_path,
                transform=self.train_transform,
                challenge=self.challenge,
                sample_rate=self.sample_rate,
                volume_sample_rate=self.volume_sample_rate,
                use_dataset_cache=self.use_dataset_cache,
                crop_size=self.crop_size,
                seed=self.seed if self.use_seed else None,
                train_accelerations=self.train_accelerations,
                pad_sides=self.pad_sides
            )
            
            self.val_dataset = CMR2025Dataset(
                data_path=self.val_path,
                transform=self.val_transform,
                challenge=self.challenge,
                sample_rate=1.0,  # Use all validation slices
                volume_sample_rate=1.0,  # Use all validation volumes
                use_dataset_cache=self.use_dataset_cache,
                crop_size=self.crop_size,
                seed=self.seed if self.use_seed else None,
                train_accelerations=self.val_accelerations if self.fix_acceleration_val is None else [self.fix_acceleration_val],
                pad_sides=self.pad_sides
            )
        
        if stage == 'test' or stage is None and self.test_path:
            self.test_dataset = CMR2025Dataset(
                data_path=self.test_path,
                transform=self.test_transform,
                challenge=self.challenge,
                sample_rate=1.0,  # Use all test slices
                volume_sample_rate=1.0,  # Use all test volumes
                use_dataset_cache=self.use_dataset_cache,
                crop_size=self.crop_size,
                seed=self.seed if self.use_seed else None,
                train_accelerations=self.test_accelerations,
                pad_sides=self.pad_sides
            )
    
    def train_dataloader(self):
        """Return the training dataloader."""
        sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset) if self.distributed_sampler else None
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=sampler is None,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        """Return the validation dataloader."""
        sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False) if self.distributed_sampler else None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=False,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        """Return the test dataloader."""
        if not hasattr(self, 'test_dataset'):
            return None
        
        sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset, shuffle=False) if self.distributed_sampler else None
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=False,
            pin_memory=True,
        )
"""
Fully self-contained CMR2025 DataModule for PromptMR-plus with fixed data type handling
This module implements all required functionality without depending on external functions.

This file should be placed in data/cmr2025.py in the PromptMR-plus repository.
"""
import os
import glob
import random
import numpy as np
import h5py
from typing import Optional, List, Tuple, Union, Callable
from multiprocessing import cpu_count

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

# Only import absolute necessities from mri_utils, with fallbacks
try:
    from mri_utils import rss
except ImportError:
    # Implement our own rss function
    def rss(data, dim=0):
        """
        Root sum of squares along a specified dimension
        """
        return torch.sqrt((data ** 2).sum(dim))

try:
    from mri_utils import fft2c
except ImportError:
    # Implement our own fft2c function
    def fft2c(data):
        """
        Apply centered 2D FFT
        """
        assert data.size(-1) == 2, "Last dimension should be 2 (real and imaginary parts)"
        data = torch.view_as_complex(data)
        data = torch.fft.ifftshift(data, dim=(-2, -1))
        data = torch.fft.fft2(data, dim=(-2, -1), norm='ortho')
        data = torch.fft.fftshift(data, dim=(-2, -1))
        return torch.view_as_real(data)

try:
    from mri_utils import ifft2c
except ImportError:
    # Implement our own ifft2c function
    def ifft2c(data):
        """
        Apply centered 2D IFFT
        """
        assert data.size(-1) == 2, "Last dimension should be 2 (real and imaginary parts)"
        data = torch.view_as_complex(data)
        data = torch.fft.ifftshift(data, dim=(-2, -1))
        data = torch.fft.ifft2(data, dim=(-2, -1), norm='ortho')
        data = torch.fft.fftshift(data, dim=(-2, -1))
        return torch.view_as_real(data)

# Implement our own complex_center_crop
def complex_center_crop(data, shape):
    """
    Apply a center crop to complex data.
    
    Args:
        data: Input data with the last dimension containing real and imaginary parts
        shape: Desired output shape
        
    Returns:
        Center cropped data
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Crop shape should be smaller than input shape")
    
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    
    return data[..., w_from:w_to, h_from:h_to, :]

# Implement our own complex_random_crop
def complex_random_crop(data, shape):
    """
    Apply a random crop to complex data.
    
    Args:
        data: Input data with the last dimension containing real and imaginary parts
        shape: Desired output shape
        
    Returns:
        Randomly cropped data
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Crop shape should be smaller than input shape")
    
    w_from = np.random.randint(0, data.shape[-3] - shape[0] + 1)
    h_from = np.random.randint(0, data.shape[-2] - shape[1] + 1)
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    
    return data[..., w_from:w_to, h_from:h_to, :]

# Implement our own random mask functionality
def create_mask_for_accelerations(shape, acceleration, calib_size=16, seed=None):
    """
    Create a random mask for a given acceleration factor
    
    Args:
        shape: Shape of the mask
        acceleration: Acceleration factor
        calib_size: Size of the calibration region
        seed: Random seed
    
    Returns:
        Binary mask
    """
    if seed is not None:
        np.random.seed(seed)
        
    mask = np.zeros(shape)
    
    # Calculate the center indices
    center_h, center_w = shape[0] // 2, shape[1] // 2
    
    # Create calibration region
    calib_h_start = center_h - calib_size // 2
    calib_h_end = calib_h_start + calib_size
    calib_w_start = center_w - calib_size // 2
    calib_w_end = calib_w_start + calib_size
    
    # Set calibration region to 1
    mask[calib_h_start:calib_h_end, calib_w_start:calib_w_end] = 1
    
    # Calculate number of lines to sample
    num_lines = shape[0] // acceleration
    
    # Create probability distribution (higher probability in the center)
    pdf = np.zeros(shape[0])
    pdf_range = np.arange(shape[0]) - shape[0] // 2
    pdf = 1 / (1 + abs(pdf_range) ** 2)
    pdf[calib_h_start:calib_h_end] = 0  # Zero out the calibration region
    
    # Normalize the PDF
    if pdf.sum() > 0:
        pdf = pdf / pdf.sum()
    
    # Select random lines based on PDF
    selected_lines = np.random.choice(
        shape[0],
        size=num_lines - calib_size,
        replace=False,
        p=pdf
    )
    
    # Apply the mask
    mask[selected_lines, :] = 1
    
    return mask

# Create get_mask_func factory
def get_mask_func(acceleration, calib_size=16, seed=None):
    """
    Factory function that returns a mask function
    
    Args:
        acceleration: Acceleration factor
        calib_size: Size of the calibration region
        seed: Random seed
    
    Returns:
        Mask function that takes a shape and returns a mask
    """
    def _mask_func(shape):
        return create_mask_for_accelerations(shape, acceleration, calib_size, seed)
    
    return _mask_func

# Helper functions for data loading
def retrieve_metadata_from_file(fname):
    """
    Read metadata from h5 file.
    
    Args:
        fname: Path to h5 file
        
    Returns:
        Dictionary with metadata
    """
    with h5py.File(fname, 'r') as hf:
        metadata = {}
        for key in hf.attrs:
            metadata[key] = hf.attrs[key]
        
        # Add shape if available
        if 'kspace' in hf:
            metadata['shape'] = hf['kspace'].shape
            
    return metadata

def normalize_instance(data, mean=None, std=None, eps=0.0):
    """
    Normalize data to zero mean and unit standard deviation.
    
    Args:
        data: Input data
        mean: Mean to use for normalization (if None, calculated from data)
        std: Standard deviation to use for normalization (if None, calculated from data)
        eps: Small constant for numerical stability
        
    Returns:
        Normalized data, mean, standard deviation
    """
    if mean is None:
        mean = data.mean()
    if std is None:
        std = data.std()
    return (data - mean) / (std + eps), mean, std


class CMR2025Dataset(Dataset):
    """
    Dataset class for CMR2025 dataset.
    """
    def __init__(
        self,
        files: List[str],
        acceleration: Optional[Union[int, List[int]]] = None,
        acquisition: Optional[List[str]] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        mode: str = 'train',
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        use_seed: bool = True,
        pad_sides: bool = False,
        load_directly: bool = False,
        fix_acceleration: Optional[int] = None,
        **kwargs
    ):
        """
        Args:
            files: List of h5 files to load
            acceleration: Desired acceleration rate. Can be a single number or a list
            acquisition: Type of acquisition
            crop_size: Desired output size
            mode: train or val
            sample_rate: Rate at which to sample the slices
            volume_sample_rate: Rate at which to sample the volumes
            use_seed: Whether to use a fixed random seed for slice sampling
            pad_sides: Whether to pad the sides of k-space
            load_directly: Whether to load the data directly
            fix_acceleration: Fixed acceleration rate (for validation)
        """
        self.files = files
        self.acceleration = acceleration
        self.acquisition = acquisition
        self.crop_size = crop_size
        self.mode = mode
        self.sample_rate = sample_rate if sample_rate is not None else 1.0
        self.volume_sample_rate = volume_sample_rate if volume_sample_rate is not None else 1.0
        self.use_seed = use_seed
        self.pad_sides = pad_sides
        self.load_directly = load_directly
        self.fix_acceleration = fix_acceleration

        # Calculate the length of the dataset
        self.examples = []
        
        # Load file paths
        for fname in sorted(self.files):
            try:
                metadata = retrieve_metadata_from_file(fname)
                num_slices = metadata.get('shape', [1, 1, 1])[0]
                
                if self.volume_sample_rate < 1.0:
                    if self.use_seed:
                        random.seed(tuple(map(ord, fname)))
                    if random.random() > self.volume_sample_rate:
                        continue

                for slice_id in range(num_slices):
                    if self.sample_rate < 1.0:
                        if self.use_seed:
                            random.seed(tuple(map(ord, fname)) + (slice_id,))
                        if random.random() > self.sample_rate:
                            continue
                    
                    self.examples.append((fname, slice_id))
            except Exception as e:
                print(f"Error loading metadata from {fname}: {e}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Get a training example"""
        fname, slice_id = self.examples[idx]
        
        with h5py.File(fname, 'r') as hf:
            # Load kspace data
            kspace = hf['kspace'][()]
            
            # Handle different kspace formats
            if len(kspace.shape) >= 4:
                # Already in [coil, slice, height, width] format
                kspace = kspace[:, slice_id:slice_id+1, :, :]
            elif len(kspace.shape) == 3:
                # In [slice, height, width] format or [coil, height, width]
                if kspace.shape[0] <= 32:  # Likely [coil, height, width]
                    kspace = kspace.reshape(kspace.shape[0], 1, kspace.shape[1], kspace.shape[2])
                else:  # Likely [slice, height, width]
                    kspace = kspace[slice_id:slice_id+1, :, :]
                    kspace = kspace.reshape(1, *kspace.shape)  # Add coil dimension
            else:
                raise ValueError(f"Unexpected kspace shape: {kspace.shape}")
            
            # Check if k-space is already complex or real
            is_complex = np.iscomplexobj(kspace)
            
            if 'reconstruction_rss' in hf:
                # If RSS reconstruction is provided, use it as the target
                target = hf['reconstruction_rss'][()]
                if len(target.shape) == 3:  # [slice, height, width]
                    target = target[slice_id:slice_id+1]
                elif len(target.shape) == 2:  # [height, width]
                    target = target.reshape(1, *target.shape)  # Add slice dimension
            else:
                # Otherwise, compute RSS from kspace
                kspace_tensor = torch.from_numpy(kspace)
                
                # Handle complex data type
                if is_complex:
                    # Split complex data into real and imaginary parts
                    kspace_real = torch.from_numpy(kspace.real).float()
                    kspace_imag = torch.from_numpy(kspace.imag).float()
                    
                    # Create a complex tensor with explicit real and imaginary parts
                    kspace_complex = torch.stack([kspace_real, kspace_imag], dim=-1)
                    
                    # Compute RSS
                    img = ifft2c(kspace_complex)
                    target = rss(img, dim=0).numpy()
                else:
                    # If not already complex formatted
                    # Reshape to [coil, height, width, complex=2] for ifft2c
                    if kspace_tensor.dim() == 3:  # [coil, height, width]
                        # Add complex dimension
                        kspace_tensor = torch.stack([kspace_tensor, torch.zeros_like(kspace_tensor)], dim=-1)
                    
                    # Compute RSS
                    img = ifft2c(kspace_tensor)
                    target = rss(img, dim=0).numpy()
            
            # Get metadata
            metadata = {
                'acquisition': str(hf.attrs.get('acquisition', 'unknown')),
                'max': float(hf.attrs.get('max', 0.0)),
                'norm': float(hf.attrs.get('norm', 1.0)),
                'patient_id': str(hf.attrs.get('patient_id', '')),
                'modality': str(hf.attrs.get('modality', '')),
                'center': str(hf.attrs.get('center', '')),
                'scanner': str(hf.attrs.get('scanner', ''))
            }

        # Convert to torch tensor and ensure proper data types
        if is_complex:
            kspace_real = torch.from_numpy(kspace.real).float()
            kspace_imag = torch.from_numpy(kspace.imag).float()
            kspace = torch.stack([kspace_real, kspace_imag], dim=-1)
        else:
            kspace = torch.from_numpy(kspace).float()
        
        # Create mask for undersampling if needed
        mask = None
        acceleration = 1
        
        if self.acceleration:
            if isinstance(self.acceleration, list):
                if self.mode == "train":
                    acceleration = random.choice(self.acceleration)
                else:
                    acceleration = self.fix_acceleration if self.fix_acceleration else self.acceleration[0]
            else:
                acceleration = self.acceleration
            
            # Create mask function based on acceleration rate
            mask_func = get_mask_func(acceleration, calib_size=16)
            mask = mask_func((kspace.shape[-2], kspace.shape[-1]))
            
            # Apply mask
            mask_tensor = torch.from_numpy(mask).float()
            if is_complex:
                # Apply to both real and imaginary parts
                kspace[..., 0] = kspace[..., 0] * mask_tensor[None, None, :, :]
                kspace[..., 1] = kspace[..., 1] * mask_tensor[None, None, :, :]
            else:
                # Ensure kspace has proper dimensions for mask application
                if kspace.dim() == 3:  # [coil, height, width]
                    kspace = kspace * mask_tensor[None, :, :]
                elif kspace.dim() == 4 and kspace.shape[1] == 1:  # [coil, slice=1, height, width]
                    kspace = kspace * mask_tensor[None, None, :, :]

        # Ensure kspace has complex dimension if it doesn't already
        if not is_complex and kspace.dim() <= 4:  # No complex dimension yet
            # Convert to complex format [coil, height, width, complex=2]
            if kspace.dim() == 3:  # [coil, height, width]
                kspace = torch.stack([kspace, torch.zeros_like(kspace)], dim=-1)
            elif kspace.dim() == 4 and kspace.shape[1] == 1:  # [coil, slice=1, height, width]
                kspace = torch.stack([kspace, torch.zeros_like(kspace)], dim=-1)
        
        # Convert target to tensor
        target = torch.from_numpy(target).float()
        
        # Normalize kspace
        kspace, mean, std = normalize_instance(kspace, eps=1e-11)
        
        # Normalize target
        target = normalize_instance(target, mean, std, eps=1e-11)[0]
        
        # Apply crop if needed
        if self.crop_size is not None:
            kspace = complex_center_crop(kspace, self.crop_size)
            target = complex_center_crop(target, self.crop_size)
        
        # Create sample dictionary
        sample = {
            'kspace': kspace,
            'target': target,
            'mask': mask_tensor if mask is not None else torch.ones((kspace.shape[-3], kspace.shape[-2]), dtype=torch.float),
            'mean': mean,
            'std': std,
            'fname': fname,
            'slice_id': slice_id,
            'acceleration': acceleration,
            'metadata': metadata
        }
        
        return sample


class CMR2025DataModule(pl.LightningDataModule):
    """
    DataModule for the CMR2025 dataset
    """
    def __init__(
        self,
        data_path: str,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        train_accelerations: Optional[List[int]] = None,
        val_accelerations: Optional[List[int]] = None,
        test_accelerations: Optional[List[int]] = None,
        train_acquisition: Optional[List[str]] = None,
        val_acquisition: Optional[List[str]] = None,
        test_acquisition: Optional[List[str]] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        batch_size: int = 16,
        num_workers: Optional[int] = None,
        distributed_sampler: bool = False,
        use_seed: bool = True,
        pad_sides: bool = False,
        dataset_cache: bool = True,
        fix_acceleration_val: Optional[int] = None,
        load_directly: bool = False,
        **kwargs
    ):
        """
        Args:
            data_path: Path to the dataset
            train_path: Optional specific path to training data
            val_path: Optional specific path to validation data
            test_path: Optional specific path to test data
            sample_rate: Fraction of slices to use
            volume_sample_rate: Fraction of volumes to use
            train_accelerations: List of acceleration rates for training
            val_accelerations: List of acceleration rates for validation
            test_accelerations: List of acceleration rates for testing
            train_acquisition: List of acquisition types for training
            val_acquisition: List of acquisition types for validation
            test_acquisition: List of acquisition types for testing
            crop_size: Size to crop the data to
            batch_size: Batch size
            num_workers: Number of workers for data loading
            distributed_sampler: Whether to use distributed sampling
            use_seed: Whether to use a fixed random seed
            pad_sides: Whether to pad the sides of k-space
            dataset_cache: Whether to cache the dataset
            fix_acceleration_val: Fixed acceleration rate for validation
            load_directly: Whether to load the data directly
        """
        super().__init__()
        self.data_path = data_path
        self.train_path = train_path if train_path else os.path.join(data_path, 'train')
        self.val_path = val_path if val_path else os.path.join(data_path, 'val')
        self.test_path = test_path if test_path else val_path if val_path else os.path.join(data_path, 'val')
        
        self.sample_rate = sample_rate
        self.volume_sample_rate = volume_sample_rate
        
        self.train_accelerations = train_accelerations
        self.val_accelerations = val_accelerations
        self.test_accelerations = test_accelerations
        
        self.train_acquisition = train_acquisition
        self.val_acquisition = val_acquisition
        self.test_acquisition = test_acquisition
        
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else cpu_count() // 2
        self.distributed_sampler = distributed_sampler
        self.use_seed = use_seed
        self.pad_sides = pad_sides
        self.dataset_cache = dataset_cache
        self.fix_acceleration_val = fix_acceleration_val
        self.load_directly = load_directly
        
        # Set dataset instances to None initially
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self):
        """Prepare the data if needed"""
        # Check if data directories exist
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.val_path, exist_ok=True)
        if self.test_path:
            os.makedirs(self.test_path, exist_ok=True)
    
    def setup(self, stage: Optional[str] = None):
        """Set up the datasets for the given stage"""
        if stage == 'fit' or stage is None:
            # Find training files
            train_files = sorted(glob.glob(os.path.join(self.train_path, '*.h5')))
            print(f"Found {len(train_files)} training files in {self.train_path}")
            
            if len(train_files) == 0:
                print(f"Warning: No training files found in {self.train_path}")
                print(f"Current directory: {os.getcwd()}")
                try:
                    print(f"Directory contents: {os.listdir(self.train_path)}")
                except:
                    print(f"Could not list contents of {self.train_path}")
            
            # Set up training dataset
            self.train_dataset = CMR2025Dataset(
                files=train_files,
                acceleration=self.train_accelerations,
                acquisition=self.train_acquisition,
                crop_size=self.crop_size,
                mode='train',
                sample_rate=self.sample_rate,
                volume_sample_rate=self.volume_sample_rate,
                use_seed=self.use_seed,
                pad_sides=self.pad_sides,
                load_directly=self.load_directly
            )
            
            # Find validation files
            val_files = sorted(glob.glob(os.path.join(self.val_path, '*.h5')))
            print(f"Found {len(val_files)} validation files in {self.val_path}")
            
            if len(val_files) == 0:
                print(f"Warning: No validation files found in {self.val_path}")
                print(f"Current directory: {os.getcwd()}")
                try:
                    print(f"Directory contents: {os.listdir(self.val_path)}")
                except:
                    print(f"Could not list contents of {self.val_path}")
            
            # Set up validation dataset
            self.val_dataset = CMR2025Dataset(
                files=val_files,
                acceleration=self.val_accelerations,
                acquisition=self.val_acquisition,
                crop_size=self.crop_size,
                mode='val',
                sample_rate=1.0,  # Use all slices for validation
                volume_sample_rate=1.0,  # Use all volumes for validation
                use_seed=self.use_seed,
                pad_sides=self.pad_sides,
                load_directly=self.load_directly,
                fix_acceleration=self.fix_acceleration_val
            )
            
        if stage == 'test' or stage is None:
            # Find test files
            test_path = self.test_path if self.test_path else self.val_path
            test_files = sorted(glob.glob(os.path.join(test_path, '*.h5')))
            print(f"Found {len(test_files)} test files in {test_path}")
            
            if len(test_files) == 0 and test_path != self.val_path:
                print(f"Warning: No test files found in {test_path}")
                print(f"Current directory: {os.getcwd()}")
                try:
                    print(f"Directory contents: {os.listdir(test_path)}")
                except:
                    print(f"Could not list contents of {test_path}")
            
            # Set up test dataset
            self.test_dataset = CMR2025Dataset(
                files=test_files,
                acceleration=self.test_accelerations,
                acquisition=self.test_acquisition,
                crop_size=self.crop_size,
                mode='test',
                sample_rate=1.0,  # Use all slices for testing
                volume_sample_rate=1.0,  # Use all volumes for testing
                use_seed=self.use_seed,
                pad_sides=self.pad_sides,
                load_directly=self.load_directly,
                fix_acceleration=self.fix_acceleration_val
            )
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Return the training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Return the validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Return the test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
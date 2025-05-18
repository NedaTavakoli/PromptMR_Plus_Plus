"""
Dataset checker and preprocessor for CMR2025 dataset
This script checks the format of the h5 files and ensures they're compatible with the CMR2025 DataModule
"""
import os
import argparse
import glob
import h5py
import numpy as np
from tqdm import tqdm

def check_and_fix_dataset(data_path, fix=False, verbose=False):
    """
    Check the format of h5 files and fix them if needed
    
    Args:
        data_path: Path to h5 files
        fix: Whether to fix issues
        verbose: Whether to print detailed information
    """
    print(f"Checking dataset in {data_path}")
    
    # Find all h5 files
    h5_files = sorted(glob.glob(os.path.join(data_path, '**/*.h5'), recursive=True))
    print(f"Found {len(h5_files)} h5 files")
    
    # Statistics
    stats = {
        'correct_format': 0,
        'missing_kspace': 0,
        'missing_rss': 0,
        'missing_shape': 0,
        'wrong_kspace_format': 0,
        'fixed': 0,
        'failed': 0
    }
    
    # Check each file
    for file_path in tqdm(h5_files):
        try:
            with h5py.File(file_path, 'r') as f:
                # Check if kspace exists
                if 'kspace' not in f:
                    stats['missing_kspace'] += 1
                    if verbose:
                        print(f"Missing kspace: {file_path}")
                    continue
                
                # Get kspace shape
                kspace = f['kspace']
                kspace_shape = kspace.shape
                
                # Check kspace format
                if len(kspace_shape) != 4 and len(kspace_shape) != 3:
                    stats['wrong_kspace_format'] += 1
                    if verbose:
                        print(f"Wrong kspace format {kspace_shape}: {file_path}")
                    
                    # Fix if requested
                    if fix:
                        try:
                            # Make a copy of the file
                            fixed_file = file_path + '.fixed'
                            with h5py.File(fixed_file, 'w') as out_f:
                                # Copy kspace data
                                kspace_data = kspace[()]
                                
                                # Reshape if needed
                                if len(kspace_shape) == 2:  # [height, width]
                                    # Add coil and slice dimensions
                                    kspace_data = kspace_data.reshape(1, 1, *kspace_shape)
                                elif len(kspace_shape) == 3 and kspace_shape[0] > 10:  # Likely [height, width, coil]
                                    # Transpose and add slice dimension
                                    kspace_data = np.transpose(kspace_data, (2, 0, 1)).reshape(kspace_shape[2], 1, kspace_shape[0], kspace_shape[1])
                                
                                # Create kspace dataset
                                out_f.create_dataset('kspace', data=kspace_data)
                                
                                # Check if RSS exists
                                if 'reconstruction_rss' in f:
                                    # Copy RSS
                                    out_f.create_dataset('reconstruction_rss', data=f['reconstruction_rss'][()])
                                
                                # Copy attributes
                                for attr in f.attrs:
                                    out_f.attrs[attr] = f.attrs[attr]
                                
                                # Add shape attribute
                                out_f.attrs['shape'] = kspace_data.shape
                            
                            # Replace original with fixed
                            os.rename(fixed_file, file_path)
                            stats['fixed'] += 1
                        except Exception as e:
                            stats['failed'] += 1
                            if verbose:
                                print(f"Failed to fix {file_path}: {e}")
                    continue
                
                # Check if RSS exists
                if 'reconstruction_rss' not in f:
                    stats['missing_rss'] += 1
                    if verbose:
                        print(f"Missing RSS: {file_path}")
                
                # Check if shape attribute exists
                if 'shape' not in f.attrs:
                    stats['missing_shape'] += 1
                    if verbose:
                        print(f"Missing shape attribute: {file_path}")
                    
                    # Fix if requested
                    if fix:
                        try:
                            with h5py.File(file_path, 'a') as out_f:
                                out_f.attrs['shape'] = kspace_shape
                            stats['fixed'] += 1
                        except Exception as e:
                            stats['failed'] += 1
                            if verbose:
                                print(f"Failed to fix shape: {e}")
                
                # File is in correct format
                stats['correct_format'] += 1
        
        except Exception as e:
            stats['failed'] += 1
            if verbose:
                print(f"Error checking {file_path}: {e}")
    
    # Print statistics
    print("--- Dataset Statistics ---")
    print(f"Total files: {len(h5_files)}")
    print(f"Correct format: {stats['correct_format']}")
    print(f"Missing kspace: {stats['missing_kspace']}")
    print(f"Missing RSS: {stats['missing_rss']}")
    print(f"Missing shape attribute: {stats['missing_shape']}")
    print(f"Wrong kspace format: {stats['wrong_kspace_format']}")
    
    if fix:
        print(f"Fixed: {stats['fixed']}")
        print(f"Failed to fix: {stats['failed']}")
    
    return stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check and fix CMR2025 dataset')
    parser.add_argument('--data_path', type=str, required=True, help='Path to h5 files')
    parser.add_argument('--fix', action='store_true', help='Fix issues')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information')
    
    args = parser.parse_args()
    
    check_and_fix_dataset(args.data_path, args.fix, args.verbose)

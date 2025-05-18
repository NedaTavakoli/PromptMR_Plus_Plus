"""
Dataset converter for CMR2025 to PromptMR-plus compatible format
"""
import os
import argparse
import glob
import json
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

def convert_cmr2025_dataset(input_path, output_path, split_json=None):
    """
    Convert CMR2025 dataset to PromptMR-plus compatible format.
    
    Args:
        input_path: Path to the original h5 dataset
        output_path: Path to save the converted dataset
        split_json: Path to the train/val split JSON file
    """
    os.makedirs(output_path, exist_ok=True)
    
    # If split JSON is provided, use it to create train/val splits
    if split_json and os.path.exists(split_json):
        with open(split_json, 'r') as f:
            split_data = json.load(f)
        
        train_files = split_data.get('train', [])
        val_files = split_data.get('val', [])
        
        # Convert to basenames for easier matching
        train_basenames = [os.path.basename(f) for f in train_files]
        val_basenames = [os.path.basename(f) for f in val_files]
        
        # Create train and val directories
        train_dir = os.path.join(output_path, 'train')
        val_dir = os.path.join(output_path, 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
    else:
        # If no split file is provided, just convert all files
        train_basenames = []
        val_basenames = []
    
    # Find all h5 files in the input directory
    h5_files = sorted(glob.glob(os.path.join(input_path, '*.h5')))
    
    if not h5_files:
        print(f"No h5 files found in {input_path}")
        return
    
    print(f"Found {len(h5_files)} h5 files")
    
    # Process each file
    for h5_file in tqdm(h5_files):
        try:
            basename = os.path.basename(h5_file)
            
            # Determine output path based on split
            if basename in train_basenames:
                output_file = os.path.join(train_dir, basename)
            elif basename in val_basenames:
                output_file = os.path.join(val_dir, basename)
            else:
                # If file is not in the split or no split is provided
                if split_json:
                    # Skip files not in the split
                    continue
                else:
                    # Just put in output directory
                    output_file = os.path.join(output_path, basename)
            
            # Open the original h5 file
            with h5py.File(h5_file, 'r') as src:
                # Create output h5 file
                with h5py.File(output_file, 'w') as dst:
                    # Check if 'kspace' dataset exists
                    if 'kspace' in src:
                        # Copy kspace dataset
                        kspace = src['kspace'][()]
                        dst.create_dataset('kspace', data=kspace)
                    else:
                        print(f"No kspace dataset found in {h5_file}")
                        continue
                    
                    # Check if 'reconstruction_rss' exists
                    if 'reconstruction_rss' in src:
                        # Copy RSS reconstruction
                        rss = src['reconstruction_rss'][()]
                        dst.create_dataset('reconstruction_rss', data=rss)
                    
                    # Copy attributes
                    if 'max' in src.attrs:
                        dst.attrs['max'] = src.attrs['max']
                    else:
                        # If 'max' attribute doesn't exist, calculate it from RSS if available
                        if 'reconstruction_rss' in src:
                            dst.attrs['max'] = np.max(src['reconstruction_rss'][()])
                        else:
                            dst.attrs['max'] = 1.0
                    
                    if 'norm' in src.attrs:
                        dst.attrs['norm'] = src.attrs['norm']
                    else:
                        # If 'norm' attribute doesn't exist, calculate it from RSS if available
                        if 'reconstruction_rss' in src:
                            dst.attrs['norm'] = np.linalg.norm(src['reconstruction_rss'][()])
                        else:
                            dst.attrs['norm'] = 1.0
                    
                    # Copy other useful attributes
                    for attr in ['acquisition', 'patient_id', 'modality', 'center', 'scanner']:
                        if attr in src.attrs:
                            dst.attrs[attr] = src.attrs[attr]
                    
                    # Set shape attribute
                    dst.attrs['shape'] = kspace.shape
                    
                    # Set padding attributes (required by some models)
                    dst.attrs['padding_left'] = 0
                    dst.attrs['padding_right'] = kspace.shape[-1]
                    dst.attrs['encoding_size'] = (kspace.shape[-2], kspace.shape[-1], 1)
                    dst.attrs['recon_size'] = (kspace.shape[-2], kspace.shape[-1], 1)
        
        except Exception as e:
            print(f"Error processing {h5_file}: {e}")
    
    print("Conversion completed!")
    
    # Report statistics
    if split_json:
        train_files_converted = len(glob.glob(os.path.join(train_dir, '*.h5')))
        val_files_converted = len(glob.glob(os.path.join(val_dir, '*.h5')))
        print(f"Converted {train_files_converted} training files and {val_files_converted} validation files")
    else:
        files_converted = len(glob.glob(os.path.join(output_path, '*.h5')))
        print(f"Converted {files_converted} files")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert CMR2025 dataset to PromptMR-plus compatible format')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the original h5 dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the converted dataset')
    parser.add_argument('--split_json', type=str, default=None, help='Path to the train/val split JSON file')
    
    args = parser.parse_args()
    
    convert_cmr2025_dataset(args.input_path, args.output_path, args.split_json)

'''
This script is used to prepare h5 training dataset from original matlab dataset for CMRxRecon2025 dataset.
Simplified version that only converts mat files to h5 without handling masks.
'''
import os
import glob
import json
import argparse
import sys
from os.path import join, dirname, basename, exists, expanduser
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

# Import from existing data module
from data.transforms import to_tensor
from mri_utils import ifft2c, rss_complex


def load_kdata(file_path):
    """
    Load k-space data from .mat file with support for compound type handling
    """
    try:
        with h5py.File(file_path, 'r') as hf:
            if 'kspace' in hf:
                kspace_dataset = hf['kspace']
                
                # Handle compound data type
                if kspace_dataset.dtype.names and 'real' in kspace_dataset.dtype.names and 'imag' in kspace_dataset.dtype.names:
                    # Extract real and imaginary parts
                    kspace_real = kspace_dataset['real'][()]
                    kspace_imag = kspace_dataset['imag'][()]
                    
                    # Create complex array
                    kspace_complex = kspace_real + 1j * kspace_imag
                    return kspace_complex
                else:
                    # Handle case where k-space might be directly stored as complex values
                    return kspace_dataset[()]
            else:
                print(f"No 'kspace' key found in file {file_path}")
                return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def remove_bad_files(file_name):
    '''
    Some files may contain only background or very bad quality.
    Update this list based on your data inspection for CMRxRecon2025.
    '''
    # Update this list based on your own data quality checks
    rm_files = []
    if file_name in rm_files:
        return True
    else:
        return False


def remove_bad_slices(kdata_, file_name):
    '''
    Remove specific slices which are bad quality or only background
    '''
    # Update this function based on your own data inspection for CMRxRecon2025
    return kdata_


def create_data_split(file_list, train_ratio=0.8):
    """
    Create a random split of data into training and validation sets
    """
    np.random.seed(42)  # For reproducibility
    
    # Get patient IDs from file list
    patient_files = {}
    for ff in file_list:
        # Extract patient ID - look for patterns like P001 in the filename
        file_parts = basename(ff).split('_')
        for part in file_parts:
            if part.startswith('P') and len(part) >= 4 and part[1:].isdigit():
                patient_id = part
                break
        else:
            # If no patient ID found, use the whole filename as the key
            patient_id = basename(ff).split('.')[0]
        
        if patient_id not in patient_files:
            patient_files[patient_id] = []
        
        patient_files[patient_id].append(ff)
    
    # Get list of all patients
    patients = list(patient_files.keys())
    np.random.shuffle(patients)
    
    # Split patients into train and validation sets
    num_train = int(len(patients) * train_ratio)
    train_patients = patients[:num_train]
    val_patients = patients[num_train:]
    
    # Get file lists
    train_files = []
    for p in train_patients:
        train_files.extend(patient_files[p])
    
    val_files = []
    for p in val_patients:
        val_files.extend(patient_files[p])
    
    return {
        "train": train_files,
        "val": val_files
    }


def find_multicoil_paths(base_dirs):
    """
    Search for MultiCoil directories in various potential locations
    """
    found_paths = []
    
    for base_dir in base_dirs:
        # Expand any user paths like ~
        expanded_base = expanduser(base_dir)
        if not exists(expanded_base):
            print(f"Warning: {base_dir} does not exist, skipping")
            continue
            
        # Try specific patterns
        patterns = [
            join(expanded_base, "**/MultiCoil"),
            join(expanded_base, "**/ChallengeData/MultiCoil"),
            join(expanded_base, "**/MICCAIChallenge*/*/MultiCoil"),
            join(expanded_base, "**/Raw_data/**/MultiCoil"),
            join(expanded_base, "**/DATA/**/MultiCoil")
        ]
        
        for pattern in patterns:
            try:
                paths = glob.glob(pattern, recursive=True)
                found_paths.extend(paths)
            except Exception as e:
                print(f"Error searching pattern {pattern}: {e}")
    
    # Make unique
    found_paths = list(set(found_paths))
    return found_paths


if __name__ == '__main__':
    # Add argparse
    parser = argparse.ArgumentParser(description='Prepare H5 dataset for CMRxRecon2025 dataset (simplified)')
    parser.add_argument('--output_h5_folder', type=str,
                        default='h5_dataset',
                        help='path to save H5 dataset')
    parser.add_argument('--input_matlab_folder', type=str,
                        default=None,
                        help='path to the original matlab data (MultiCoil folder)')
    parser.add_argument('--split_json', type=str, 
                        default='configs/data_split/cmr25-cardiac.json', 
                        help='path to save the split json file')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='ratio of data to use for training (default: 0.8)')
    parser.add_argument('--auto_find', action='store_true',
                        help='automatically search for MultiCoil directory')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('--modality', type=str, default=None,
                        help='Process only specific modality')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of files to process per modality (for testing)')
    args = parser.parse_args()
    
    # Set debug flag
    DEBUG = args.debug
    
    save_folder = args.output_h5_folder
    split_json_path = args.split_json
    train_ratio = args.train_ratio
    modality_filter = args.modality
    file_limit = args.limit
    
    # Auto-find or use specified path
    mat_folder = args.input_matlab_folder
    
    if mat_folder is None or args.auto_find:
        print('Searching for MultiCoil directories...')
        # Define potential base directories to search from
        base_dirs = [
            '/mnt/kim_share/Neda',
            '/mnt/kim_share/Neda/DATA',
            '/mnt/kim_share/Neda/DATA/ChallengeData',
            '/mnt/kim_share/Neda/CMRxRecon2025',
            '/mnt/kim/share/Neda',
            '/mnt/kim-share/Neda',
            '/mnt/kim',
            '/mnt/data',
            '/mnt'
        ]
        
        found_paths = find_multicoil_paths(base_dirs)
        
        if not found_paths:
            print("Error: Could not find MultiCoil directory automatically.")
            print("Please specify the path with --input_matlab_folder")
            sys.exit(1)
        
        print(f"Found {len(found_paths)} potential MultiCoil directories:")
        for i, path in enumerate(found_paths):
            print(f"  {i+1}. {path}")
        
        if mat_folder is None:
            # Use the first found path
            mat_folder = found_paths[0]
            print(f"Using: {mat_folder}")
        else:
            # Check if specified path is among found paths
            if mat_folder not in found_paths:
                print(f"Warning: Specified path {mat_folder} not found in automatic search")
                print(f"Will try to use it anyway")
    
    # Check if directory exists
    if not exists(mat_folder):
        print(f"Error: MATLAB data folder {mat_folder} does not exist")
        sys.exit(1)
    
    print('MATLAB data folder:', mat_folder)
    print('H5 save folder:', save_folder)
    if args.debug:
        print('Debug mode enabled')
    if modality_filter:
        print(f'Filtering to modality: {modality_filter}')
    if file_limit:
        print(f'Limiting to {file_limit} files per modality')

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    print('## Step 1: Identify available modalities')
    
    # List directories in the MultiCoil folder to find available modalities
    modalities = []
    
    try:
        all_items = os.listdir(mat_folder)
        modalities = [item for item in all_items if os.path.isdir(os.path.join(mat_folder, item)) and not item.startswith('.')]
    except Exception as e:
        print(f"Error reading directory {mat_folder}: {e}")
        sys.exit(1)
    
    if not modalities:
        print(f"Error: No modality directories found in {mat_folder}")
        sys.exit(1)
    
    # Filter modalities if requested
    if modality_filter:
        if modality_filter in modalities:
            print(f"Filtering to only process modality: {modality_filter}")
            modalities = [modality_filter]
        else:
            print(f"Requested modality {modality_filter} not found in available modalities: {modalities}")
            sys.exit(1)
    
    print(f"Found {len(modalities)} modalities: {modalities}")
    
    print('## Step 2: Process each modality')
    
    all_processed_files = []
    
    # Count total files for progress bar
    total_files = 0
    for modality in modalities:
        try:
            modality_path = os.path.join(mat_folder, modality)
            fullsample_path = os.path.join(modality_path, 'TrainingSet', 'FullSample')
            if not os.path.exists(fullsample_path):
                continue
                
            # Count files in this modality
            count = 0
            for root, _, files in os.walk(fullsample_path):
                count += sum(1 for f in files if f.endswith('.mat'))
            
            if file_limit is not None:
                count = min(count, file_limit)
                
            total_files += count
        except Exception as e:
            print(f"Error counting files in {modality}: {e}")
    
    print(f"Total files to process: {total_files}")
    progress_bar = tqdm(total=total_files, desc="Processing files")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for modality in modalities:
        print(f"\nProcessing modality: {modality}")
        
        # Path to modality folder
        modality_path = os.path.join(mat_folder, modality)
        
        # Get path to FullSample directory
        training_path = os.path.join(modality_path, 'TrainingSet')
        
        if not exists(training_path):
            print(f"TrainingSet directory not found for {modality}, skipping")
            continue
        
        fullsample_dir = os.path.join(training_path, 'FullSample')
        
        if not exists(fullsample_dir):
            print(f"FullSample directory not found for {modality}, skipping")
            continue
            
        # Find all centers
        try:
            centers = [d for d in os.listdir(fullsample_dir) if os.path.isdir(os.path.join(fullsample_dir, d))]
        except Exception as e:
            print(f"Error reading directory {fullsample_dir}: {e}")
            continue
        
        files_processed_in_modality = 0
        
        for center in centers:
            center_path = os.path.join(fullsample_dir, center)
            
            if not exists(center_path):
                continue
            
            # Find all scanners
            try:
                scanners = [d for d in os.listdir(center_path) if os.path.isdir(os.path.join(center_path, d))]
            except Exception as e:
                print(f"Error reading directory {center_path}: {e}")
                continue
            
            for scanner in scanners:
                scanner_path = os.path.join(center_path, scanner)
                
                if not exists(scanner_path):
                    continue
                
                # Find all patients
                try:
                    patients = [d for d in os.listdir(scanner_path) if os.path.isdir(os.path.join(scanner_path, d))]
                except Exception as e:
                    print(f"Error reading directory {scanner_path}: {e}")
                    continue
                
                for patient in patients:
                    patient_path = os.path.join(scanner_path, patient)
                    
                    if not exists(patient_path):
                        continue
                    
                    # Find all .mat files
                    try:
                        kspace_files = glob.glob(os.path.join(patient_path, '*.mat'))
                    except Exception as e:
                        print(f"Error finding .mat files in {patient_path}: {e}")
                        continue
                    
                    # Apply file limit if specified
                    if file_limit is not None and files_processed_in_modality + len(kspace_files) > file_limit:
                        kspace_files = kspace_files[:file_limit - files_processed_in_modality]
                    
                    for kspace_file in kspace_files:
                        # Get base name without extension
                        file_type = os.path.basename(kspace_file).split('.')[0]
                        save_name = f"{modality}_{center}_{scanner}_{patient}_{file_type}"
                        
                        # Skip bad files if necessary
                        if remove_bad_files(save_name):
                            print(f"Skipping known bad file: {save_name}")
                            progress_bar.update(1)
                            continue
                        
                        try:
                            # Load k-space data
                            kdata = load_kdata(kspace_file)
                            if kdata is None:
                                print(f"Failed to load k-space data from {kspace_file}")
                                progress_bar.update(1)
                                continue
                            
                            # Swap phase_encoding and readout
                            kdata = kdata.swapaxes(-1, -2)
                            
                            # Remove bad slices if necessary
                            kdata = remove_bad_slices(kdata, save_name)
                            
                            # Generate RSS reconstruction
                            kdata_th = to_tensor(kdata)
                            img_coil = ifft2c(kdata_th).to(device)
                            img_rss = rss_complex(img_coil, dim=-3).cpu().numpy()
                            
                            # Save H5
                            h5_path = join(save_folder, save_name + '.h5')
                            
                            with h5py.File(h5_path, 'w') as file:
                                file.create_dataset('kspace', data=kdata)
                                file.create_dataset('reconstruction_rss', data=img_rss)
                            
                                file.attrs['max'] = img_rss.max()
                                file.attrs['norm'] = np.linalg.norm(img_rss)
                                file.attrs['acquisition'] = file_type
                                file.attrs['shape'] = kdata.shape
                                file.attrs['padding_left'] = 0
                                file.attrs['padding_right'] = kdata.shape[-1]
                                file.attrs['encoding_size'] = (kdata.shape[-2], kdata.shape[-1], 1)
                                file.attrs['recon_size'] = (kdata.shape[-2], kdata.shape[-1], 1)
                                file.attrs['patient_id'] = f"{patient}_{file_type}"
                                file.attrs['modality'] = modality
                                file.attrs['center'] = center
                                file.attrs['scanner'] = scanner
                            
                            all_processed_files.append(h5_path)
                            files_processed_in_modality += 1
                            progress_bar.update(1)
                        
                        except Exception as e:
                            print(f"Error processing {kspace_file}: {e}")
                            progress_bar.update(1)
                        
                        # Stop if we've reached the limit for this modality
                        if file_limit is not None and files_processed_in_modality >= file_limit:
                            print(f"Reached limit of {file_limit} files for modality {modality}")
                            break
                    
                    # Stop if we've reached the limit for this modality
                    if file_limit is not None and files_processed_in_modality >= file_limit:
                        break
                
                # Stop if we've reached the limit for this modality
                if file_limit is not None and files_processed_in_modality >= file_limit:
                    break
            
            # Stop if we've reached the limit for this modality
            if file_limit is not None and files_processed_in_modality >= file_limit:
                break
    
    # Close progress bar
    progress_bar.close()
    
    print(f"\nTotal processed files: {len(all_processed_files)}")
    
    if not all_processed_files:
        print("No files were processed. Check the input paths.")
        exit(1)
    
    print('## Step 3: Create data split json file')
    
    # Create directory for JSON file if it doesn't exist
    os.makedirs(os.path.dirname(split_json_path), exist_ok=True)
    
    # Generate the file list for H5 files - use the actual processed files
    split_dict = create_data_split(all_processed_files, train_ratio)
    
    # Save the split to json
    with open(split_json_path, 'w') as f:
        json.dump(split_dict, f, indent=2)
    
    print(f'Created split json at {split_json_path}')
    print(f'Train files: {len(split_dict["train"])}')
    print(f'Val files: {len(split_dict["val"])}')
    
    print('## Step 4: Split H5 dataset to train and val using symbolic links')
    
    train_folder = join(os.path.dirname(save_folder), 'train')
    val_folder = join(os.path.dirname(save_folder), 'val')
    
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)
    
    # Extract filenames from paths
    train_list = [os.path.basename(ff) for ff in split_dict['train']]
    val_list = [os.path.basename(ff) for ff in split_dict['val']]
    
    # Create symbolic links
    train_links_created = 0
    val_links_created = 0
    
    for ff in all_processed_files:
        save_name = os.path.basename(ff)
        if save_name in train_list:
            # Create symbolic link if it doesn't exist
            link_path = join(train_folder, save_name)
            if not os.path.exists(link_path):
                try:
                    os.symlink(os.path.abspath(ff), link_path)
                    train_links_created += 1
                except Exception as e:
                    print(f"Error creating symbolic link for {ff}: {e}")
        elif save_name in val_list:
            # Create symbolic link if it doesn't exist
            link_path = join(val_folder, save_name)
            if not os.path.exists(link_path):
                try:
                    os.symlink(os.path.abspath(ff), link_path)
                    val_links_created += 1
                except Exception as e:
                    print(f"Error creating symbolic link for {ff}: {e}")
    
    print('Done!')
    print('Number of files in H5 folder:', len(all_processed_files))
    print('Number of symbolic link files created in train folder:', train_links_created)
    print('Number of symbolic link files created in val folder:', val_links_created)
    print('Total symbolic link files in train folder:', len(glob.glob(join(train_folder, '*.h5'))))
    print('Total symbolic link files in val folder:', len(glob.glob(join(val_folder, '*.h5'))))
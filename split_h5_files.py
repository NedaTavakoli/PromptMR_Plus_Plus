import os
import glob
import json
import numpy as np
import shutil
from pathlib import Path
from os.path import join, basename, exists, abspath


def create_data_split(file_list, train_ratio=0.8):
    """
    Create a random split of data into training and validation sets based on patient IDs
    """
    np.random.seed(42)  # For reproducibility
    
    # Get patient IDs from file list
    patient_files = {}
    for ff in file_list:
        # Extract patient ID - look for patterns like P001 in the filename
        file_parts = basename(ff).split('_')
        # Try to find a patient ID in the filename
        patient_id = None
        for part in file_parts:
            if part.startswith('P') and len(part) >= 4 and part[1:].isdigit():
                patient_id = part
                break
        
        # If we couldn't find a patient ID pattern, use a different approach
        if patient_id is None:
            # Try to extract a patient identifier from the 4th position (index 3)
            # Common format: modality_center_scanner_patient_type
            if len(file_parts) >= 4:
                patient_id = file_parts[3]
            else:
                # Fallback: use the whole filename without extension
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


def main():
    # Configuration parameters - modified to use h5_dataset_simplified
    h5_dataset_folder = "h5_dataset_simplified"  # Path to your H5 files
    train_ratio = 0.8  # Percentage of data for training
    split_json_path = "configs/data_split/cmr25-cardiac.json"  # Path to save the data split
    
    # Choose method: 'copy' or 'symlink'
    method = 'copy'  # Using copy instead of symlink
    
    # Check if h5_dataset_folder exists
    if not exists(h5_dataset_folder):
        print(f"Error: H5 dataset folder {h5_dataset_folder} does not exist")
        return
    
    # Find all H5 files
    h5_files = sorted(glob.glob(join(h5_dataset_folder, "*.h5")))
    
    if not h5_files:
        print(f"Error: No H5 files found in {h5_dataset_folder}")
        return
    
    print(f"Found {len(h5_files)} H5 files in {h5_dataset_folder}")
    
    # Create train and val directories inside the h5_dataset_folder
    train_folder = join(h5_dataset_folder, "train")
    val_folder = join(h5_dataset_folder, "val")
    
    # Create directories if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    
    # Create data split
    split_dict = create_data_split(h5_files, train_ratio)
    
    # Make sure the directory for the JSON file exists
    os.makedirs(os.path.dirname(split_json_path), exist_ok=True)
    
    # Save the split to JSON
    with open(split_json_path, 'w') as f:
        json.dump(split_dict, f, indent=2)
    
    print(f"Created split JSON at {split_json_path}")
    print(f"Train files: {len(split_dict['train'])}")
    print(f"Val files: {len(split_dict['val'])}")
    
    # Create files in train and val folders
    train_files_created = 0
    val_files_created = 0
    
    print(f"Using method: {method}")
    
    for h5_file in split_dict["train"]:
        # Create file in train folder if it doesn't exist
        filename = basename(h5_file)
        dest_path = join(train_folder, filename)
        if not exists(dest_path):
            try:
                if method == 'copy':
                    shutil.copy2(h5_file, dest_path)
                else:
                    os.symlink(abspath(h5_file), dest_path)
                train_files_created += 1
            except Exception as e:
                print(f"Error creating {method} for {h5_file}: {e}")
    
    for h5_file in split_dict["val"]:
        # Create file in val folder if it doesn't exist
        filename = basename(h5_file)
        dest_path = join(val_folder, filename)
        if not exists(dest_path):
            try:
                if method == 'copy':
                    shutil.copy2(h5_file, dest_path)
                else:
                    os.symlink(abspath(h5_file), dest_path)
                val_files_created += 1
            except Exception as e:
                print(f"Error creating {method} for {h5_file}: {e}")
    
    print(f"Created {train_files_created} files in train folder")
    print(f"Created {val_files_created} files in val folder")
    
    # Verify the files were created
    train_files_count = len(glob.glob(join(train_folder, "*.h5")))
    val_files_count = len(glob.glob(join(val_folder, "*.h5")))
    
    print(f"Total files in train folder: {train_files_count}")
    print(f"Total files in val folder: {val_files_count}")


if __name__ == "__main__":
    main()
# Script for data loading (PyTorch Dataset and DataLoader)

# TODO:
# - Handle different DRR views properly if needed (e.g., load both AP/Lat and stack channels).
# - Implement proper train/validation splitting (e.g., patient-level split if possible).
# - Optimize file loading using MONAI's PersistentDataset or CacheDataset for speed.
# - Fine-tune intensity ranges and augmentation parameters.

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import SimpleITK as sitk
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

# MONAI Imports
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd, # Keep for DRR [0,1] scaling
    ScaleIntensityRanged, # For CT HU windowing/scaling
    Spacingd,         # For resampling CT volumes
    Resized,
    RandAffined,
    RandGaussianNoised, # Added augmentation
    RandFlipd,        # Added augmentation
    ToTensord,
    # Orientationd # Consider adding Orientationd to standardize orientation
)
# from monai.data import PersistentDataset, DataLoader # Consider for optimization

# --- Define Target Sizes & Spacing (Should match model expectations) ---
TARGET_DRR_SIZE = (256, 256) # Example: Height, Width
TARGET_CT_SIZE = (64, 128, 128) # Example: Depth, Height, Width (Reduced H/W for memory)
TARGET_CT_SPACING = (1.0, 1.0, 1.0) # Example: Isotropic 1mm spacing

# Define CT Intensity Window (Example: Soft tissue/bone range)
CT_WINDOW_MIN = -1000.0
CT_WINDOW_MAX = 1000.0

class DRRReconstructionDataset(Dataset):
    """
    Dataset for loading DRR images and corresponding CT volumes using MONAI transforms.
    Assumes DRRs are named based on their source CT file.
    """
    def __init__(self, data_dict_list, transform=None):
        """
        Args:
            data_dict_list (list): List of dictionaries, each like {'ct': ct_path, 'drr': drr_path_dict}.
            transform (callable, optional): MONAI transforms to be applied on each sample dict.
        """
        self.data_dict_list = data_dict_list
        self.transform = transform

    def __len__(self):
        return len(self.data_dict_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item_dict = self.data_dict_list[idx]
        # Pick the first available DRR view for now
        drr_key = list(item_dict['drr'].keys())[0]
        data_files = {'ct': item_dict['ct'], 'drr': item_dict['drr'][drr_key]}

        if self.transform:
            try:
                data_transformed = self.transform(data_files)
                return data_transformed['drr'], data_transformed['ct']
            except Exception as e:
                print(f"Error applying transforms to sample index {idx} ({data_files}): {e}")
                print("Returning placeholder tensors due to transform error.")
                return torch.randn(1, *TARGET_DRR_SIZE), torch.randn(1, *TARGET_CT_SIZE)
        else:
            # Fallback should ideally not be used when transforms are defined
            print(f"Warning: No transform provided for sample index {idx}.")
            return torch.randn(1, *TARGET_DRR_SIZE), torch.randn(1, *TARGET_CT_SIZE)


# Function to create the list of data dictionaries
def create_data_list(drr_dir, ct_dir, drr_suffix="_drr_axis0.png", ct_extensions=('.nii', '.nii.gz', '.mha', '.mhd')):
    """ Scans directories and creates a list of dictionaries containing paired file paths. """
    samples = []
    drr_suffixes = [drr_suffix] if isinstance(drr_suffix, str) else drr_suffix

    print("Scanning for paired DRR/CT files...")
    ct_files = []
    for ext in ct_extensions:
        ct_files.extend(glob.glob(os.path.join(ct_dir, f"*{ext}")))
    print(f"Found {len(ct_files)} potential CT files in {ct_dir}")

    for ct_path in tqdm(ct_files, desc="Pairing files"):
        ct_filename = os.path.basename(ct_path)
        base_name = ct_filename
        for ext in ct_extensions:
             if base_name.lower().endswith(ext.lower()):
                  base_name = base_name[:-len(ext)]
                  break
        
        drr_paths_for_ct = {}
        found_drr = False
        for suffix in drr_suffixes:
            drr_filename = f"{base_name}{suffix}"
            drr_path = os.path.join(drr_dir, drr_filename)
            if os.path.exists(drr_path):
                key = suffix
                drr_paths_for_ct[key] = drr_path
                found_drr = True

        if found_drr:
            samples.append({'ct': ct_path, 'drr': drr_paths_for_ct})

    print(f"Found {len(samples)} CT scans with at least one corresponding DRR.")
    if not samples:
        print("Warning: No paired samples found. Check directories, naming convention, and suffixes.")
    return samples


# Example function to get dataloaders
def get_dataloaders(drr_dir, ct_dir, drr_suffix="_drr_axis0.png", batch_size=4, num_workers=0, val_split=0.2, random_seed=42):
    """
    Creates training and validation dataloaders with more specific MONAI transforms.
    """
    # --- Define MONAI Transforms ---
    keys = ['drr', 'ct'] 
    
    # Define transforms for training data
    train_transform = Compose([
        # Load images: DRR with PIL (needs channel first later), CT with ITK/Nibabel
        LoadImaged(keys=keys, image_only=True, allow_missing_keys=True, reader="ITKReader"), # Use ITKReader for CT by default
        EnsureChannelFirstd(keys=keys), # Ensure channel dimension is first
        # Resample CT to target spacing (e.g., 1x1x1 mm)
        Spacingd(
            keys=['ct'], 
            pixdim=TARGET_CT_SPACING, 
            mode='bilinear', # Use 'nearest' for label maps
            align_corners=True,
        ),
        # Normalize CT intensity using windowing, then scale DRR and CT to [0, 1]
        ScaleIntensityRanged(
            keys=['ct'], 
            a_min=CT_WINDOW_MIN, 
            a_max=CT_WINDOW_MAX, 
            b_min=0.0, 
            b_max=1.0, 
            clip=True,
        ),
        ScaleIntensityd(keys=['drr'], minv=0.0, maxv=1.0), # Scale DRR separately
        # Resize to target spatial dimensions
        Resized(keys=['drr'], spatial_size=TARGET_DRR_SIZE, mode='bilinear', align_corners=False), 
        Resized(keys=['ct'], spatial_size=TARGET_CT_SIZE, mode='trilinear', align_corners=False), 
        # Augmentations
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0), # Flip along depth axis
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1), # Flip along height axis
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2), # Flip along width axis
        RandAffined(
            keys=keys,
            prob=0.5, 
            rotate_range=(np.pi / 18, np.pi / 18, np.pi / 18), # Reduced rotation range
            scale_range=(0.1, 0.1, 0.1), 
            mode=('bilinear', 'trilinear'), 
            padding_mode='zeros',
        ),
        RandGaussianNoised(keys=keys, prob=0.1, mean=0.0, std=0.1),
        ToTensord(keys=keys) # Convert to PyTorch tensors
    ])
    
    # Define transforms for validation data (no augmentation)
    val_transform = Compose([
        LoadImaged(keys=keys, ensure_channel_first=True, image_only=True, allow_missing_keys=True, reader="ITKReader"),
        # EnsureChannelFirstd(keys=keys, channel_dim="no_channel"), # Handled by LoadImaged
        Spacingd(keys=['ct'], pixdim=TARGET_CT_SPACING, mode='bilinear', align_corners=True),
        ScaleIntensityRanged(keys=['ct'], a_min=CT_WINDOW_MIN, a_max=CT_WINDOW_MAX, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityd(keys=['drr'], minv=0.0, maxv=1.0),
        Resized(keys=['drr'], spatial_size=TARGET_DRR_SIZE, mode='bilinear', align_corners=False),
        Resized(keys=['ct'], spatial_size=TARGET_CT_SIZE, mode='trilinear', align_corners=False),
        ToTensord(keys=keys)
    ])

    # --- Create Data List ---
    all_samples_list = create_data_list(drr_dir, ct_dir, drr_suffix=drr_suffix)

    if not all_samples_list:
         print("Error: Dataset is empty. Cannot create dataloaders.")
         return DataLoader([]), DataLoader([]) 

    # --- Split Data ---
    dataset_size = len(all_samples_list)
    val_size = int(np.floor(val_split * dataset_size))
    train_size = dataset_size - val_size
    train_size = max(0, train_size); val_size = max(0, val_size)
    print(f"Splitting dataset: Train={train_size}, Validation={val_size}")

    if train_size == 0 or val_size == 0:
        print("Warning: Train or validation set size is zero after split. Check val_split or dataset size.")
        # Handle edge case where one split might be empty
        if train_size == 0 and val_size > 0:
            train_files = []
            val_files = all_samples_list
        elif val_size == 0 and train_size > 0:
            train_files = all_samples_list
            val_files = []
        else: # Both zero
            train_files, val_files = [], []
    else:
         generator = torch.Generator().manual_seed(random_seed)
         train_indices, val_indices = random_split(range(dataset_size), [train_size, val_size], generator=generator)
         train_files = [all_samples_list[i] for i in train_indices]
         val_files = [all_samples_list[i] for i in val_indices]

    # --- Create Datasets ---
    # Consider PersistentDataset for caching if I/O is a bottleneck
    # cache_dir = "./persistent_cache" # Define cache directory
    # train_dataset = PersistentDataset(data=train_files, transform=train_transform, cache_dir=os.path.join(cache_dir, "train"))
    # val_dataset = PersistentDataset(data=val_files, transform=val_transform, cache_dir=os.path.join(cache_dir, "val"))
    
    train_dataset = DRRReconstructionDataset(data_dict_list=train_files, transform=train_transform) if train_files else []
    val_dataset = DRRReconstructionDataset(data_dict_list=val_files, transform=val_transform) if val_files else []

    # --- Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    print(f"Created train_loader with {len(train_loader)} batches.")
    print(f"Created val_loader with {len(val_loader)} batches.")

    return train_loader, val_loader

if __name__ == '__main__':
    # Example usage:
    print("Testing DataLoader creation with enhanced MONAI transforms...")
    base_dir = "./temp_data_loader_test"
    dummy_drr_path = os.path.join(base_dir, "drrs")
    dummy_ct_path = os.path.join(base_dir, "cts")
    os.makedirs(dummy_drr_path, exist_ok=True)
    os.makedirs(dummy_ct_path, exist_ok=True)

    print(f"Creating dummy files in {base_dir}...")
    dummy_ct_filename = "dummy_ct_scan.nii.gz"
    dummy_ct_filepath = os.path.join(dummy_ct_path, dummy_ct_filename)
    # Create dummy CT with some variation and size different from target
    dummy_ct_data = (np.random.rand(70, 140, 150) * 2000 - 1000).astype(np.float32) # D, H, W with HU-like range
    dummy_ct_img = sitk.GetImageFromArray(dummy_ct_data)
    dummy_ct_img.SetSpacing([0.8, 0.8, 1.5]) # Example non-isotropic spacing
    sitk.WriteImage(dummy_ct_img, dummy_ct_filepath)
    print(f"Created dummy CT: {dummy_ct_filepath}")

    dummy_drr_filename = "dummy_ct_scan_drr_axis0.png" # Matches default suffix
    dummy_drr_filepath = os.path.join(dummy_drr_path, dummy_drr_filename)
    # Create dummy DRR with size different from target
    dummy_drr_data = np.random.randint(0, 256, (300, 300), dtype=np.uint8) # H, W
    dummy_drr_img = Image.fromarray(dummy_drr_data)
    dummy_drr_img.save(dummy_drr_filepath)
    print(f"Created dummy DRR: {dummy_drr_filepath}")

    # Test dataloader creation
    train_loader, val_loader = get_dataloaders(dummy_drr_path, dummy_ct_path, batch_size=1, val_split=0.5)

    # Test iterating through one batch (if loaders are not empty)
    if len(train_loader) > 0:
        print("\nTesting batch loading from train_loader...")
        try:
            drr_batch, ct_batch = next(iter(train_loader)) 
            print(f"  DRR batch shape: {drr_batch.shape}") # Should be [B, 1, H, W] -> [1, 1, 256, 256]
            print(f"  CT batch shape: {ct_batch.shape}")   # Should be [B, 1, D, H, W] -> [1, 1, 64, 128, 128]
            print(f"  DRR dtype: {drr_batch.dtype}, CT dtype: {ct_batch.dtype}")
            print(f"  DRR min: {drr_batch.min():.2f}, DRR max: {drr_batch.max():.2f}") # Should be ~0.0 to ~1.0
            print(f"  CT min: {ct_batch.min():.2f}, CT max: {ct_batch.max():.2f}")   # Should be ~0.0 to ~1.0
        except Exception as e:
            print(f"  Error loading batch: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nTrain loader is empty, skipping batch loading test.")

    print("\nDataLoader creation test finished.")

    # Clean up dummy files/dirs
    import shutil
    try:
        if os.path.exists(base_dir):
             shutil.rmtree(base_dir)
             print(f"Cleaned up dummy directory: {base_dir}")
    except Exception as e:
        print(f"Error cleaning up dummy directory: {e}")

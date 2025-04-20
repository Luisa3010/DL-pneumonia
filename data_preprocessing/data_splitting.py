import os
import shutil
import random
import argparse
from pathlib import Path

def split_data(source_dir, output_dir, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15, seed=42):
    """
    Split data from source directory into train, test, and validation sets.
    
    Args:
        source_dir (str): Path to the directory containing all data files
        output_dir (str): Path to the directory where train/test/val subdirectories will be created
        train_ratio (float): Proportion of data for training set
        test_ratio (float): Proportion of data for test set
        val_ratio (float): Proportion of data for validation set
        seed (int): Random seed for reproducibility
    """
    # Validate ratios
    if abs(train_ratio + test_ratio + val_ratio - 1.0) > 1e-10:
        raise ValueError("Train, test, and validation ratios must sum to 1")
    
    # Create destination directories if they don't exist
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    val_dir = os.path.join(output_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all image files from source directory
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(Path(source_dir).glob(ext)))
    
    # Shuffle files with a fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(image_files)
    
    # Calculate split indices
    n_files = len(image_files)
    train_end = int(n_files * train_ratio)
    test_end = train_end + int(n_files * test_ratio)
    
    # Split files
    train_files = image_files[:train_end]
    test_files = image_files[train_end:test_end]
    val_files = image_files[test_end:]
    
    # Copy files to respective directories
    for files, dest_dir in [
        (train_files, train_dir),
        (test_files, test_dir),
        (val_files, val_dir)
    ]:
        for file_path in files:
            dest_path = os.path.join(dest_dir, file_path.name)
            shutil.copy2(file_path, dest_path)
    
    # Print summary
    print(f"Data split complete:")
    print(f"  - Training set: {len(train_files)} files ({train_ratio*100:.1f}%)")
    print(f"  - Test set: {len(test_files)} files ({test_ratio*100:.1f}%)")
    print(f"  - Validation set: {len(val_files)} files ({val_ratio*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into train, test, and validation sets')
    parser.add_argument('--source', '-s', required=True, help='Path to source directory containing all data files')
    parser.add_argument('--output', '-o', required=True, help='Path to output directory where train/test/val subdirectories will be created')
    parser.add_argument('--train', '-tr', type=float, default=0.7, help='Proportion for training set (default: 0.7)')
    parser.add_argument('--test', '-te', type=float, default=0.15, help='Proportion for test set (default: 0.15)')
    parser.add_argument('--val', '-v', type=float, default=0.15, help='Proportion for validation set (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    split_data(
        args.source,
        args.output,
        train_ratio=args.train, 
        test_ratio=args.test, 
        val_ratio=args.val,
        seed=args.seed
    )
import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
from medmnist import PneumoniaMNIST
from torchvision import transforms
from data_preprocessing.generate_data_set import create_combined_dataset
from data_preprocessing.data_cleaning import PreprocessTransform

def export_dataset(output_dir='exported_dataset'):
    """
    Export the dataset as PNG images with metadata CSV files.
    
    Args:
        output_dir (str): Directory where the exported dataset will be saved
    """
    # Create output directories
    splits = ['train', 'test', 'val']
    classes = ['NORMAL', 'PNEUMONIA']
    sources = ['mnist', 'raw', 'augmented']
    
    for split in splits:
        for class_name in classes:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
    
    # Create transform pipeline
    transform = transforms.Compose([
        PreprocessTransform(),
    ])
    
    # Process each split
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        # Create dataset with appropriate data sources for each split
        if split == 'train':
            dataset = create_combined_dataset(
                mnist_data=PneumoniaMNIST(split=split, download=True),
                raw_data_dir='data/raw',
                augmented_data_dir='data/augmented',
                transform=transform,
                split=split
            )
        elif split == 'val':
            dataset = create_combined_dataset(
                mnist_data=PneumoniaMNIST(split=split, download=True),
                raw_data_dir='data/raw',
                transform=transform,
                split=split
            )
        else:  # test split
            dataset = create_combined_dataset(
                mnist_data=PneumoniaMNIST(split=split, download=True),
                transform=transform,
                split=split
            )
        
        if dataset is None:
            print(f"No data found for {split} split")
            continue
        
        # Initialize metadata list
        metadata = []
        
        # Process each image
        for idx in tqdm(range(len(dataset)), desc=f"Exporting {split} images"):
            # Get image and label
            img_tensor, label = dataset[idx]
            
            # Convert tensor to numpy array and scale to 0-255
            img_np = img_tensor.squeeze().numpy()  # Remove channel dimension
            img_np = (img_np * 255).astype(np.uint8)
            
            # Determine class and source
            class_name = 'NORMAL' if label.item() == 0 else 'PNEUMONIA'
            
            # Determine source based on index ranges and split
            if split == 'train':
                if idx < len(PneumoniaMNIST(split=split)):
                    source = 'mnist'
                elif idx < len(PneumoniaMNIST(split=split)) + len(os.listdir(os.path.join('data/raw', class_name, split))):
                    source = 'raw'
                else:
                    source = 'augmented'
            elif split == 'val':
                if idx < len(PneumoniaMNIST(split=split)):
                    source = 'mnist'
                else:
                    source = 'raw'
            else:  # test split
                source = 'mnist'  # Only MNIST data in test set
            
            # Create filename
            filename = f"{source}_{idx:04d}.png"
            filepath = os.path.join(output_dir, split, class_name, filename)
            
            # Save image
            Image.fromarray(img_np).save(filepath)
            
            # Add to metadata
            metadata.append({
                'filename': filename,
                'class': int(label.item()),
                'source': source,
                'original_index': idx
            })
        
        # Save metadata CSV
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(os.path.join(output_dir, f"{split}_metadata.csv"), index=False)
        
        print(f"Exported {len(metadata)} images for {split} split")
        print(f"Class distribution:")
        print(metadata_df['class'].value_counts())
        print(f"Source distribution:")
        print(metadata_df['source'].value_counts())

if __name__ == "__main__":
    
    # Add the project root to Python path when running directly
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    export_dataset() 
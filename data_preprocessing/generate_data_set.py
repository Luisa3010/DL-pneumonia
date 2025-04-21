from torch.utils.data import Dataset, ConcatDataset
import torch
import os
import numpy as np
from PIL import Image
from data_preprocessing.data_cleaning import preprocess_image


class MNISTWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # Apply transform if needed
        if self.transform:
            img = self.transform(img)
        
        # Ensure label is a 1D tensor with one element
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.float32)
        elif label.dim() == 0:  # If it's a scalar tensor
            label = label.unsqueeze(0)  # Convert to 1D tensor
            label = torch.tensor(label, dtype=torch.float32)
        
        return img, label        

class ImageDataset(Dataset):
    def __init__(self, data, labels, train=True, transform=None):
        """
        Args:
            data: Input images
            labels: Labels
            train: Whether this is training or validation set
            transform: Optional transforms to apply to data
        """
        self.data = data
        self.labels = labels
        self.train = train # TODO: Check if this is needed and remove it
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the input and target at the given index
        x = self.data[idx]
        y = self.labels[idx]
        
        # Apply transforms if any
        if self.transform:
            x = self.transform(x)
            
        return x, y


class FolderImageDataset(Dataset):
    def __init__(self, folder_path, class_name, transform=None):
        """
        Dataset for loading images from a folder structure
        
        Args:
            folder_path: Path to the folder containing images
            class_name: Class name for these images ('NORMAL' or 'PNEUMONIA')
            transform: Optional transforms to apply to data
        """
        self.folder_path = folder_path
        self.transform = transform
        self.class_name = class_name
        self.label = 0 if class_name == 'NORMAL' else 1
        
        # Get all image files
        self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                           if os.path.isfile(os.path.join(folder_path, f)) and 
                           f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Use the preprocess_image function to preprocess the image
        image = preprocess_image(img_path)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        
        if self.transform:
            image = self.transform(image)
            
        # Create a 1D tensor with one element to match MNIST format
        label = torch.tensor([self.label], dtype=torch.float32)


        return image, label
    


def create_combined_dataset(mnist_data, raw_data_dir, augmented_data_dir=None, transform=None, split='train'):
    """
    Create a combined dataset for a specific split (train, val, or test)
    
    Args:
        mnist_data: MNIST data for the specified split (PneumoniaMNIST dataset)
        raw_data_dir: Directory containing raw data
        augmented_data_dir: Directory containing augmented data (can be None)
        transform: Optional transforms to apply to data
        split: Which split to create ('train', 'val', or 'test')
        
    Returns:
        A ConcatDataset for the specified split
    """
    is_train = (split == 'train')
    datasets_to_combine = []
    
    # 1. Add MNIST data
    if mnist_data:
        # Check the format of MNIST data to ensure consistency
        sample_img, sample_label = mnist_data[0]
           
        # Create a wrapper to ensure MNIST labels match our convention (0=NORMAL, 1=PNEUMONIA)
        mnist_data = MNISTWrapper(mnist_data, transform)
        datasets_to_combine.append(mnist_data)
    
    # 2. Add raw data
    for class_name in ['NORMAL', 'PNEUMONIA']:
        raw_folder = os.path.join(raw_data_dir, class_name, split)
        if os.path.exists(raw_folder):
            raw_dataset = FolderImageDataset(
                raw_folder,
                class_name,
                transform=transform
            )
            datasets_to_combine.append(raw_dataset)
    
    # 3. Add augmented data (only if augmented_data_dir is provided)
    if augmented_data_dir is not None:
        for class_name in ['NORMAL', 'PNEUMONIA']:
            aug_folder = os.path.join(augmented_data_dir, class_name, split)
            if os.path.exists(aug_folder):
                aug_dataset = FolderImageDataset(
                    aug_folder,
                    class_name,
                    transform=transform
                )
                datasets_to_combine.append(aug_dataset)
    
    # Combine all datasets for this split
    if datasets_to_combine:
               
        return ConcatDataset(datasets_to_combine)
    else:
        print(f"Warning: No datasets found to combine for split '{split}'")
        return None
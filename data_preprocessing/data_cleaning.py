import os
import numpy as np
from PIL import Image
import glob
import torch
from torchvision import transforms

def preprocess_image(img_path):
    """
    Load and preprocess a single image.
    
    Args:
        img_path (str): Path to the image file
        
    Returns:
        np.ndarray: Preprocessed image array
    """
    # Load image
    img = Image.open(img_path)
    
    # Convert to grayscale if it's not already
    if img.mode != 'L':
        img = img.convert('L')
    
    # Make image quadratic by cropping equally from sides
    width, height = img.size
    if width != height:
        if width > height:
            # Landscape image
            diff = width - height
            left = diff // 2
            right = width - (diff - left)
            img = img.crop((left, 0, right, height))
        else:
            # Portrait image
            diff = height - width
            top = diff // 2
            bottom = height - (diff - top)
            img = img.crop((0, top, width, bottom))
    
    # Resize to 28x28
    img = img.resize((28, 28), Image.LANCZOS)
    
    # Convert to numpy array and standardize (zero mean, unit variance)
    img_array = np.array(img)
    img_array = (img_array - np.mean(img_array)) / np.std(img_array)
    
    return img_array

# Keep the original function for backward compatibility
def preprocess_images(data_path):
    """
    Load and preprocess image data from the specified path.
    
    Args:
        data_path (str): Path to the directory containing image files
        
    Returns:
        np.ndarray: Preprocessed image data
    """
    # List to store processed images
    processed_images = []
    
    # Find all image files in the directory
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(data_path, ext)))
    
    for img_path in image_files:
        img_array = preprocess_image(img_path)
        processed_images.append(img_array)
    
    # Convert list to numpy array
    return np.array(processed_images)

class PreprocessTransform:
    """Custom transform that applies the preprocessing pipeline"""
    def __call__(self, img):
        # Convert PIL image to numpy array
        img_array = np.array(img)
                
        # Make image quadratic by cropping equally from sides
        if img_array.shape == (1, 28, 28):
            img_array = img_array.squeeze(0)  # Remove the channel dimension, resulting in (28, 28)

        h, w = img_array.shape
        if w != h:
            if w > h:
            # Landscape image
                diff = w - h
                left = diff // 2
                right = w - (diff - left)
                img_array = img_array[:, left:right]
        else:
            # Portrait image
            diff = h - w
            top = diff // 2
            bottom = h - (diff - top)
            img_array = img_array[top:bottom, :]

        
        # Resize to 28x28 
        img = Image.fromarray(img_array)
        img = img.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img)
        
        # Standardize (zero mean, unit variance)
        img_array = (img_array - np.mean(img_array)) / np.std(img_array)
        
        # Convert to torch tensor
        return torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)  # Add channel dimension


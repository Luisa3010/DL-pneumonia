
import os
import numpy as np
from PIL import Image
import glob

def load_and_preprocess_images(data_path):
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
        
        # # Convert to numpy array and standardize (zero mean, unit variance)
        img_array = np.array(img)
        img_array = (img_array - np.mean(img_array)) / np.std(img_array)
        
        processed_images.append(img_array)
    
    # Convert list to numpy array
    return np.array(processed_images)


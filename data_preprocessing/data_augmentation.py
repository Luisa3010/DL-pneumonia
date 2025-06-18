import albumentations as A
from PIL import Image
import numpy as np
import os
import glob
import argparse
from tqdm import tqdm


def augment_images(image_folder, output_folder, n=None):
    """
    Augment the images with random rotations.
    
    Args:
        image_folder (str): Path to the folder containing images to augment
        output_folder (str): Path to the folder where augmented images will be saved
        n (int, optional): Number of augmentations to create per image. If None, augment each image once.
        
    Returns:
        str: Path to the folder containing augmented images
    """
    
    # Define the transform
    transform = A.Compose([
        A.Rotate(limit=(-30, 30), p=1.0, crop_border=True)  
    ])
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all image files in the directory
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    
    if n is None:
        # Apply augmentation to each image in the folder once
        for img_path in tqdm(image_files, desc="Augmenting images"):
            # Load image
            img = np.array(Image.open(img_path))
            
            # Apply augmentation
            augmented = transform(image=img)
            
            # Save augmented image
            filename = os.path.basename(img_path)
            base, ext = os.path.splitext(filename)
            output_path = os.path.join(output_folder, f"{base}_augmented{ext}")
            Image.fromarray(augmented['image']).save(output_path)
    else:
        # Create n augmentations for each image
        for img_path in tqdm(image_files, desc="Augmenting images"):
            # Load image
            img = np.array(Image.open(img_path))
            filename = os.path.basename(img_path)
            base, ext = os.path.splitext(filename)
            
            for i in range(n):
                # Apply augmentation
                augmented = transform(image=img)
                
                # Save augmented image
                output_path = os.path.join(output_folder, f"{base}_augmented_{i}{ext}")
                Image.fromarray(augmented['image']).save(output_path)
    
    return output_folder


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Augment images with random rotations')
    parser.add_argument('--input', '-i', required=True, help='Path to folder containing images to augment')
    parser.add_argument('--output', '-o', required=True, help='Path to folder where augmented images will be saved')
    parser.add_argument('--count', '-n', type=int, default=None, 
                        help='Number of augmentations to create per image. If not specified, augment each image once.')
    
    args = parser.parse_args()
    
    augmented_folder = augment_images(args.input, args.output, args.count)
    print(f"Augmented images saved to: {augmented_folder}")

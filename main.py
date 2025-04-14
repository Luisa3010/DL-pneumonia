from config import DATA_PATH, PROCESSED_DATA_PATH
import os
import numpy as np
from data_preprocessing.data_cleaning import load_and_preprocess_images



# Load and preprocess data
processed_images = load_and_preprocess_images(DATA_PATH)


# Create directory if it doesn't exist
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# Save processed images as numpy arrays
for i, img in enumerate(processed_images):
    # Save as numpy file (.npy)
    np_path = os.path.join(PROCESSED_DATA_PATH, f"processed_image_{i}.npy")
    np.save(np_path, img)

print("Data preprocessed and saved as NumPy arrays successfully.")

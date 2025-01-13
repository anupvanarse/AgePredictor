import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

os.chdir(Path("/home/avanarse/projects/AgePredictor"))

def preprocess_and_save_npz(data_dir, output_file):
    """
    Preprocess images in the data directory and save them in a compressed NPZ file.
    """
    images = []
    labels = []
    target_size = (128, 128)
    total_files = sum(len(files) for _, _, files in os.walk(data_dir))

    with tqdm(total=total_files, desc="Creating NPZ dataset", unit="file") as pbar:
        for label in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label)
            if os.path.isdir(label_path):
                for img_file in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_file)
                    try:
                        # Open and convert to RGB to ensure 3 channels
                        image = Image.open(img_path).convert('RGB')
                        # Resize if needed
                        if image.size != target_size:
                            image = image.resize(target_size, Image.Resampling.LANCZOS)
                            print(f"Resized {img_path} to {target_size}")
                        
                        # Convert to numpy array with explicit (H, W, 3) shape
                        img_array = np.array(image, dtype=np.uint8)
                        assert img_array.shape == (128, 128, 3), f"Unexpected shape: {img_array.shape}"
                        
                        images.append(img_array)
                        labels.append(int(label))
                    except Exception as e:
                        print(f"Error processing file {img_path}: {e}")
                    finally:
                        pbar.update(1)

    # Convert to numpy arrays
    images = np.array(images) 
    labels = np.array(labels)

    # Verify final shape
    print(f"Final images array shape: {images.shape}")  # Should be (N, 128, 128, 3)
    print(f"Final labels array shape: {labels.shape}")  # Should be (N,)
    print(f"Images dtype: {images.dtype}")
    print(f"Labels dtype: {labels.dtype}")

    # Save compressed
    np.savez_compressed(output_file, images=images, labels=labels)
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    preprocess_and_save_npz("./dataset", "./dataset/np_data.npz")
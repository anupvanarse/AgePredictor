import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import warnings

class AgeDataset(Dataset):
    def __init__(self, npz_file, transform=None):
        self.transform = transform
        try:
            # Load data in memory-mapped mode
            self.data = np.load(npz_file, mmap_mode='r')
            self.images = self.data['images']
            self.labels = torch.from_numpy(self.data['labels']).int()
        except Exception as e:
            warnings.warn(f"Error loading NPZ file: {e}")
            raise

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert to contiguous array and then to PIL
        image = np.array(self.images[idx], copy=True)
        image = Image.fromarray(image)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_train_test_loaders(npz_file, batch_size, test_split=0.2, num_workers=4, 
                          train_transform=None, test_transform=None):
    # Verify file exists
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"NPZ file not found: {npz_file}")

    # Create single dataset instance
    base_dataset = AgeDataset(npz_file, transform=None)
    
    # Use generator for memory efficiency
    indices = np.arange(len(base_dataset))
    labels = base_dataset.labels.numpy()
    
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_split,
        stratify=labels,
        random_state=42
    )

    # Create train/test datasets with respective transforms
    train_dataset = torch.utils.data.Subset(
        AgeDataset(npz_file, transform=train_transform), 
        train_indices
    )
    test_dataset = torch.utils.data.Subset(
        AgeDataset(npz_file, transform=test_transform), 
        test_indices
    )

    # Use persistent workers to avoid worker initialization overhead
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False
    )

    return train_loader, test_loader

if __name__ == '__main__':
    npz_file = "./dataset/np_data.npz"
    batch_size = 128

    # Define data augmentations for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define simple resizing and normalization for testing
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader, test_loader = get_train_test_loaders(
        npz_file, 
        batch_size, 
        test_split=0.2, 
        num_workers=4,
        train_transform=train_transform,
        test_transform=test_transform
        )

    # Verify data loading
    for images, labels in train_loader:
        print(f"Train Batch - Images: {images.shape}, Dtype: {images.dtype}, Labels: {labels.shape}")
        break  # Just check first batch

    for images, labels in test_loader:
        print(f"Test Batch - Images: {images.shape},  Dtype: {images.dtype}, Labels: {labels.shape}")
        break  # Just check first batch
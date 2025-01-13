import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import warnings

# Custom Dataset class for loading age prediction data
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
        # Return the total number of samples in the dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # Retrieve a single sample from the dataset
        # Convert the image array to a PIL image for transformations
        image = np.array(self.images[idx], copy=True)
        image = Image.fromarray(image)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Function to create DataLoaders for train, validation, and test sets
def get_data_loaders(npz_file, batch_size, test_split=0.2, val_split=0.1, num_workers=4, 
                     train_transform=None, test_transform=None):
    """
    Splits the dataset into train, test, and validation sets and returns their respective DataLoaders.
    """
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"NPZ file not found: {npz_file}")

    # Load the dataset without any transformations
    base_dataset = AgeDataset(npz_file, transform=None)
    
    # Create indices for splitting the dataset
    indices = np.arange(len(base_dataset))
    labels = base_dataset.labels.numpy()

    # Split into train+val and test sets
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_split, stratify=labels, random_state=42
    )
    
    # Further split train+val into train and validation sets
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_split / (1 - test_split), stratify=labels[train_val_indices], random_state=42
    )
    
    # Apply the respective transforms to each subset
    train_dataset = torch.utils.data.Subset(
        AgeDataset(npz_file, transform=train_transform), 
        train_indices
    )
    val_dataset = torch.utils.data.Subset(
        AgeDataset(npz_file, transform=test_transform), 
        val_indices
    )
    test_dataset = torch.utils.data.Subset(
        AgeDataset(npz_file, transform=test_transform), 
        test_indices
    )

    # Create DataLoaders for each subset
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
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False
    )

    return train_loader, test_loader, val_loader

# Additional code to verify the data loading process

# if __name__ == '__main__':
#     # Define the path to the dataset and batch size
#     npz_file = "./dataset/np_data.npz"
#     batch_size = 128

#     # Define transformations for data augmentation and preprocessing
#     train_transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(p=0.4),
#         transforms.RandomRotation(degrees=10),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     test_transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # Generate DataLoaders for train, validation, and test sets
#     train_loader, val_loader, test_loader = get_data_loaders(
#         npz_file, 
#         batch_size, 
#         test_split=0.2, 
#         val_split=0.1, 
#         num_workers=4,
#         train_transform=train_transform,
#         test_transform=test_transform,
#         val_transform=test_transform
#     )

#     # Verify data loading by checking the shapes of a single batch
#     for images, labels in train_loader:
#         print(f"Train Batch - Images: {images.shape}, Labels: {labels.shape}")
#         break

#     for images, labels in val_loader:
#         print(f"Validation Batch - Images: {images.shape}, Labels: {labels.shape}")
#         break

#     for images, labels in test_loader:
#         print(f"Test Batch - Images: {images.shape}, Labels: {labels.shape}")
#         break
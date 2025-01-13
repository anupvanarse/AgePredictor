import os
from pathlib import Path
os.chdir(Path("/home/avanarse/projects/AgePredictor"))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from utils.loader import get_train_test_loaders
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torchsummary import summary 


def main():
    # Define dataset path
    npz_file = "./dataset/np_data.npz"  # Replace with the actual path to the .npz file
    batch_size = 128
    epochs = 10

    # Define transformations with augmentations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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

    # Model setup
    model = models.efficientnet_v2_s(pretrained=True)  # Use EfficientNet-V2 Small
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),  # Prevent overfitting
        nn.Linear(256, 1)  # Final output for regression
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Print model summary
    print("Model Summary:")
    summary(model, input_size=(3, 128, 128))
    
    criterion = nn.HuberLoss(delta=1.0)  # Using Huber Loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)  # Using AdamW optimizer

    # Warmup Scheduler
    def warmup_lr_lambda(epoch):
        warmup_epochs = 3  # Adjust warmup period
        if epoch < warmup_epochs:
            return epoch / warmup_epochs  # Linear warmup
        return 1  # Default scale after warmup

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - 5) 

    def train_one_epoch(model, dataloader, criterion, optimizer, device):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc="Training", unit="batch")
        for images, ages in progress_bar:
            images, ages = images.to(device), ages.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        return epoch_loss / len(dataloader)

    def evaluate(model, dataloader, criterion, device):
        model.eval()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")
        with torch.no_grad():
            for images, ages in progress_bar:
                images, ages = images.to(device), ages.to(device, dtype=torch.float32)
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, ages)
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
        return epoch_loss / len(dataloader)

    # Training Loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, test_loader, criterion, device)

        # Update learning rate schedulers
        if epoch < 5:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "./models/efficientnet_v2_s_age_prediction.pth")

    print("Training complete and model saved.")

if __name__ == "__main__":
    main()
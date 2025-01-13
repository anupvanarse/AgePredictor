import os
from pathlib import Path

# Set working directory
os.chdir(Path("/home/avanarse/projects/AgePredictor"))

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from utils.loader import get_data_loaders
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torchsummary import summary 
import wandb


# Define training for one epoch
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    for images, ages in progress_bar:
        images, ages = images.to(device), ages.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # Mixed precision
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, ages)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    return epoch_loss / len(dataloader)

# Define evaluation
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

def main():
    # Define dataset path
    npz_file = "./dataset/np_data.npz"  # Replace with the actual path to the .npz file
    epochs = 10

    config={
        "batch_size": 128,
        "epochs": 10,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "model": "MobileNetV3-Large",
        "num_workers": 4
        }

    wandb.init(project="age-prediction", config=config)

    # Define transformations with augmentations for training and testing data
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.4),  # Randomly flip images horizontally
        transforms.RandomRotation(degrees=10),  # Apply random rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Apply color jitter
        transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),  # Randomly crop and resize
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to fixed size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    # Load training and testing datasets
    train_loader, test_loader, _ = get_data_loaders(
        npz_file, 
        batch_size=128, 
        test_split=0.2, 
        val_split=0.1, 
        num_workers=4,
        train_transform=train_transform,
        test_transform=test_transform,
    )

    # Initialize MobileNetV3 Large model
    model = models.mobilenet_v3_large(pretrained=True)  # Load pretrained weights
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 256),  # First fully connected layer
        nn.ReLU(),  # Activation function
        nn.Dropout(0.3),  # Dropout to prevent overfitting
        nn.Linear(256, 1)  # Final output for regression
    )
    
    # Set device to GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Print model summary
    print("Model Summary:")
    summary(model, input_size=(3, 128, 128))

    # Initialize loss function and optimizer
    criterion = nn.HuberLoss(delta=1.0)  # Robust loss function for regression
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)  # Optimizer with weight decay

    # Define learning rate schedulers
    def warmup_lr_lambda(epoch):
        warmup_epochs = 3  # Number of warmup epochs
        if epoch < warmup_epochs:
            return epoch / warmup_epochs  # Linear warmup
        return 1  # Default scale after warmup

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - 5)  # Cosine decay after warmup

    # Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler()

    # Training Loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss = evaluate(model, test_loader, criterion, device)

        # Log metrics to WandB
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        # Save the best model weights, implemented like a callback
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "./models/mobilenet_v3_large_best_weights.pth")
            print(f"Best model weights saved at epoch {epoch + 1}")

        # Update learning rate schedulers
        if epoch < 3:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save final model weights
    torch.save(model, "./models/mobilenet_v3_large_final_weights.pth")
    print("Training complete and final model weights saved.")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
from torchvision import transforms
from utils.loader import get_train_test_loaders

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images, targets = images.to(device), targets.to(device, dtype=torch.float32)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss

def quantize_model(model, dataloader, device):
    # Perform dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    return quantized_model

def train_qat(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", unit="batch"):
            images, targets = images.to(device), targets.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        val_loss = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Test Loss: {val_loss:.4f}")

    return model

# Initialize data loaders
npz_file = "./dataset/np_data.npz"  # Path to your dataset file
batch_size = 128

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
    npz_file=npz_file,
    batch_size=batch_size,
    test_split=0.2,
    num_workers=4,
    train_transform=train_transform,
    test_transform=test_transform
)

# Load the pre-trained model and data loaders
device = "cpu"
from torchvision import models
# model = models.efficientnet_v2_s(pretrained=False) 
# model.classifier = nn.Sequential(
#     nn.Linear(model.classifier[1].in_features, 256),
#     nn.ReLU(),
#     nn.Dropout(0.3),  # Prevent overfitting
#     nn.Linear(256, 1)  # Final output for regression
# )
# model.load_state_dict(torch.load("./models/efficientnet_v2_s_age_prediction.pth"))
model = torch.load("./models/mobilenet_v3_small_age_prediction.pth")
model.to(device)

criterion = nn.HuberLoss(delta=1.0)  # Using Huber Loss for regression

# Evaluate original model
print("Evaluating original model...")
original_loss = evaluate_model(model, test_loader, criterion, device)
print(f"Original Test Loss: {original_loss:.4f}")

# Quantize the model
print("Quantizing model...")
quantized_model = quantize_model(copy.deepcopy(model), test_loader, device)
quantized_model.to(device)

# Evaluate quantized model
print("Evaluating quantized model...")
quantized_loss = evaluate_model(quantized_model, test_loader, criterion, device)
print(f"Quantized Test Loss: {quantized_loss:.4f}")

# Check performance drop before training
if (quantized_loss - original_loss) / original_loss > 0.25:
    print("Performance dropped by more than 25%. Proceeding with Quantization Aware Training...")
    qat_model = copy.deepcopy(model)
    qat_model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
    qat_model.train()
    quantization.prepare_qat(qat_model, inplace=True)

    optimizer = torch.optim.AdamW(qat_model.parameters(), lr=1e-4, weight_decay=0.01)
    num_epochs = 2

    qat_model = train_qat(qat_model, train_loader, test_loader, criterion, optimizer, num_epochs, device)
    quantization.convert(qat_model, inplace=True)

    print("Evaluating QAT-trained model...")
    qat_loss = evaluate_model(qat_model, test_loader, criterion, device)
    print(f"QAT Test Loss: {qat_loss:.4f}")
else:
    print("Performance drop is within acceptable limits. Skipping QAT.")
    quantized_model = quantization.convert(quantized_model, inplace=False)
    print("Finalized quantized model without QAT.")

# Compare performances
print("Performance Comparison:")
print(f"Original Test Loss: {original_loss:.4f}")
print(f"Quantized Test Loss: {quantized_loss:.4f}")

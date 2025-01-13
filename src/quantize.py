import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
from torchvision import transforms
from utils.loader import get_data_loaders

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images, targets = images.to(device), targets.to(device, dtype=torch.float32)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            total_mae += torch.abs(outputs - targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    return avg_loss, avg_mae

def quantize_model(model):
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    return quantized_model

def train_qat(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_mae = 0.0
        for images, targets in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", unit="batch"):
            images, targets = images.to(device), targets.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.size(0)
            total_mae += torch.abs(outputs - targets).sum().item()

        avg_loss = total_loss / len(train_loader.dataset)
        avg_mae = total_mae / len(train_loader.dataset)
        val_loss, val_mae = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Train MAE: {avg_mae:.4f}, Test Loss: {val_loss:.4f}, Test MAE: {val_mae:.4f}")

    return model

def main():
    # Initialize data loaders
    npz_file = "./dataset/np_data.npz"  # Path to your dataset file
    batch_size = 128

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to fixed size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    train_loader, test_loader, _ = get_data_loaders(
        npz_file, 
        batch_size=128, 
        test_split=0.2, 
        val_split=0.1, 
        num_workers=4,
        train_transform=transform,
        test_transform=transform,
    )

    # Load the pre-trained model
    device = "cpu"
    model = torch.load("./models/mobilenet_v3_large_final_model.pth")
    model.to(device)

    criterion = nn.HuberLoss(delta=1.0)  # Using Huber Loss for regression

    # Evaluate original model
    print("Evaluating original model...")
    original_loss, original_mae = evaluate_model(model, test_loader, criterion, device)
    print(f"Original Test Loss: {original_loss:.4f}, Original Test MAE: {original_mae:.4f}")

    # Quantize the model
    print("Quantizing model...")
    quantized_model = quantize_model(copy.deepcopy(model))
    quantized_model.to(device)

    # Evaluate quantized model
    print("Evaluating quantized model...")
    quantized_loss, quantized_mae = evaluate_model(quantized_model, test_loader, criterion, device)
    print(f"Quantized Test Loss: {quantized_loss:.4f}, Quantized Test MAE: {quantized_mae:.4f}")

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
        qat_model = quantization.convert(qat_model, inplace=True)

        print("Evaluating QAT-trained model...")
        qat_loss, qat_mae = evaluate_model(qat_model, test_loader, criterion, device)
        print(f"QAT Test Loss: {qat_loss:.4f}, QAT Test MAE: {qat_mae:.4f}")
        torch.save(qat_model, "./models/quantized_mobilenet_v3_large_model.pth")
        print("QAT-trained quantized model saved.")
    else:
        print("Performance drop is within acceptable limits. Skipping QAT.")
        quantized_model = quantization.convert(quantized_model, inplace=False)
        torch.save(quantized_model, "./models/quantized_mobilenet_v3_large_model.pth")
        print("Quantized model saved.")

    # Compare performances
    print("Performance Comparison:")
    print(f"Original Test Loss: {original_loss:.4f}, Original Test MAE: {original_mae:.4f}")
    print(f"Quantized Test Loss: {quantized_loss:.4f}, Quantized Test MAE: {quantized_mae:.4f}")

if __name__ == "__main__":
    main()
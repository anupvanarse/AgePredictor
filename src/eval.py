import torch
from tqdm import tqdm
from torchvision import transforms
from utils.loader import get_data_loaders

def evaluate_model_mae(model, dataloader, device):
    """
    Evaluate the model on the given dataloader and compute MAE.
    
    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader for the validation set.
        device (str): Device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        float: Mean Absolute Error (MAE) across the dataset.
    """
    model.eval()
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images, targets = images.to(device), targets.to(device, dtype=torch.float32)

            outputs = model(images).squeeze(1)
            total_mae += torch.abs(outputs - targets).sum().item()
            total_samples += targets.size(0)

    avg_mae = total_mae / total_samples
    return avg_mae

def main():
    # Define paths and parameters
    npz_file = "./dataset/np_data.npz"  # Path to your dataset file
    model_path = "./models/mobilenet_v3_large_final_model.pth"  # Path to the trained model
    device = torch.device("cpu")

    # Define transformations for the validation set
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to fixed size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    # Load the validation set
    _, _, val_loader = get_data_loaders(
        npz_file, 
        batch_size=128, 
        test_split=0.2, 
        val_split=0.1, 
        num_workers=4,
        train_transform=None,  # Training transform not needed
        test_transform=transform,  # Use test transform for validation
    )

    # Load the model
    print(f"Loading model from {model_path}...")
    model = torch.load(model_path)
    model.to(device)

    # Evaluate the model
    print("Evaluating the model on the validation set...")
    val_mae = evaluate_model_mae(model, val_loader, device)

    # Print performance metrics
    print("Validation Set Performance:")
    print(f"Validation MAE: {val_mae:.4f}")

if __name__ == "__main__":
    main()

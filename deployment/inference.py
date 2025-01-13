import os
import argparse
import torch
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

def load_model(model_path, device):
    """
    Load the trained PyTorch model.

    Args:
        model_path (str): Path to the trained model file.
        device (torch.device): Device to load the model on ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: Loaded model.
    """
    print(f"Loading model from {model_path}...")
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, transform):
    """
    Preprocess a single image for inference.

    Args:
        image_path (str): Path to the image file.
        transform (torchvision.transforms.Compose): Transformations to apply.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

def infer_folder(model, folder_path, device, transform):
    """
    Perform inference on all images in a folder.

    Args:
        model (torch.nn.Module): Trained model for inference.
        folder_path (str): Path to the folder containing image files.
        device (torch.device): Device for inference ('cpu' or 'cuda').
        transform (torchvision.transforms.Compose): Transformations to apply.

    Returns:
        list: Results for each image as (filename, prediction).
    """
    results = []
    file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    print(f"Found {len(file_list)} files in {folder_path}. Starting inference...")
    for file_name in tqdm(file_list, desc="Processing Files", unit="file"):
        file_path = os.path.join(folder_path, file_name)
        try:
            image_tensor = preprocess_image(file_path, transform).to(device)
            with torch.no_grad():
                output = model(image_tensor).squeeze(1).item() 
            results.append((file_name, output))
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    
    return results

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Inference script for batch processing images in a folder.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing input images.")
    parser.add_argument("--output_file", type=str,required=False, default="inference_results.csv", help="Path to save the inference results.")
    args = parser.parse_args()

    # Set device
    device = torch.device("cpu")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to fixed size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    # Load the model
    model = load_model(args.model_path, device)

    # Perform inference
    results = infer_folder(model, args.input_folder, device, transform)

    # Save results to CSV
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, "w") as f:
        f.write("filename,prediction\n")
        for file_name, prediction in results:
            f.write(f"{file_name},{prediction}\n")
    print("Inference completed.")

if __name__ == "__main__":
    main()

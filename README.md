# Age Prediction from Facial Images

This repository contains a pipeline for predicting the age of individuals from facial images using deep learning. The project focuses on developing a scalable, efficient, and mobile-friendly solution, ultimately enabling deployment on edge devices.

---

## Project Overview

The project encompasses the following key components:

1. **Data Preparation**: Preprocessing and cleaning facial image datasets.
2. **Transfer Learning**: Fine-tuning a pre-trained deep learning model for age prediction.
3. **Evaluation**: Assessing model performance using Mean Absolute Error (MAE).
4. **Model Quantization**: Optimizing the model for edge deployment.
5. **C++ Integration**: Packaging the trained model for seamless deployment in C++ environments.

### Objective

The task of age prediction from facial images is approached as a machine learning regression problem, where the goal is to predict a continuous variable (age) based on input features extracted from the images.

---

## Folder Structure

```
project-root/
|-- dataset/               # Directory to store the dataset (recommended to create this manually)
|-- models/                # Directory for storing trained and quantized models
|-- deployment/            # Directory for deployment-related files
|   |-- Dockerfile         # Dockerfile for containerization
|   |-- inference.py       # Batch inference script
|-- src/                   # Source code for the project
|   |-- data_explorer.ipynb# Jupyter notebook for data exploration
|   |-- eval.py            # Evaluation script
|   |-- quantize.py        # Model quantization script
|   |-- train.py           # Training script
|   |-- utils/             # Helper modules
|       |-- loader.py      # Data loading utilities
|       |-- npz_generator.py # Data preprocessing script
|-- requirements.txt       # Python dependencies
|-- README.md              # Documentation
```

---

## Getting Started

### Prerequisites

1. **Python 3.10+**
2. **PyTorch**: Install using the following command for compatibility with CUDA 12.4:

    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    ```

   Refer to [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) for detailed instructions.

3. **CUDA (optional)**: For GPU acceleration.
4. **Dependencies**: Install using `pip install -r requirements.txt`.

### Dataset Preparation

The dataset contains facial images categorized by age. Each folder name corresponds to the age group. Preprocessing steps include resizing, normalization, and saving in `.npz` format for efficient loading.

Run the following to preprocess the data:

```bash
python src/utils/npz_generator.py
```

### Transfer Learning

Fine-tune the pre-trained model for age prediction using:

```bash
python src/train.py
```

Key features of the training script:
- Utilizes MobileNetV3_large as a base model with additional fully connected layers for regression.
- Data augmentations for robustness.
- Mixed precision training for efficiency.

### Evaluation

Evaluate the model on a validation set:

```bash
python src/eval.py
```

### Batch Inference

Perform inference on a folder of images:

```bash
python deployment/inference.py --model_path ./models/model.pth --input_folder ./images --output_file results.csv
```

### Quantization

Optimize the model for edge deployment by quantizing it:

```bash
python src/quantize.py
```

---

## Deployment

### Exporting Model to TorchScript

Convert the trained model to TorchScript for integration into a C++ pipeline:

```python
import torch
model = YourModel()
model.load_state_dict(torch.load("model.pth"))
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")
```

### Using the Model in C++

Refer to `deployment/README.md` for detailed instructions on integrating the model with a C++ application. It includes an example `CMakeLists.txt` and sample C++ code for inference.

---

## Key Features

1. **Preprocessing**: Image resizing, normalization, and `.npz` format conversion.
2. **Transfer Learning**: Fine-tuning a MobileNet model for regression.
3. **Quantization**: Dynamic quantization and Quantization-Aware Training (QAT) to enhance efficiency.
4. **Evaluation**: Robust metrics (e.g., MAE) for assessing performance.
5. **Edge Deployment**: Seamless integration into C++ applications.

---

## Dependencies

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Docker Support

To run the project in a containerized environment, use the provided `Dockerfile`:

```bash
docker build -t age-prediction .
docker run -it age-prediction
```

---

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [TorchVision](https://pytorch.org/vision/stable/index.html)

---
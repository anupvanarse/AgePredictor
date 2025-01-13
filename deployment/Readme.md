# Age Prediction Model Deployment in C++

This repository provides a guide for deploying a PyTorch-based age prediction model into a C++ environment. The deployment process ensures the model is optimized for edge and mobile applications.

---

## Prerequisites

1. **PyTorch C++ API (LibTorch)**: Download from [PyTorch.org](https://pytorch.org/cppdocs/).
2. **C++ Compiler**: A compiler like GCC (Linux) or MSVC (Windows).
3. **Exported Model**: The trained PyTorch model in TorchScript format.
4. **OpenCV**: For preprocessing images in the C++ pipeline (optional).

---

## Deployment Steps

### 1. Export the Model to TorchScript

In Python, convert the trained PyTorch model into TorchScript format using the following code:

```python
import torch

# Load your trained model
model = YourTrainedModel()
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# Export to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("age_prediction_model.pt")
```

This will create a `age_prediction_model.pt` file for deployment.

---

### 2. Setup C++ Environment

- Install LibTorch.
- Configure your C++ build system to link against LibTorch.

Example `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(AgePrediction)

find_package(Torch REQUIRED)

add_executable(age_prediction main.cpp)
target_link_libraries(age_prediction "${TORCH_LIBRARIES}")
set_property(TARGET age_prediction PROPERTY CXX_STANDARD 14)
```

---

### 3. Load the Model in C++

Load and use the model for inference in your C++ code:

```cpp
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>

int main() {
    // Load the scripted model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("age_prediction_model.pt");
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }

    std::cout << "Model loaded successfully!\n";

    // Example input: A single image tensor
    torch::Tensor input = torch::rand({1, 3, 224, 224}); // Replace with actual preprocessed input
    auto output = model.forward({input}).toTensor();

    std::cout << "Predicted Age: " << output.argmax(1).item<int>() << "\n";
    return 0;
}
```

---

### 4. Image Preprocessing

Ensure the input image matches the preprocessing used during training. Resize, normalize, and convert the image into a tensor. Example using OpenCV:

```cpp
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

torch::Tensor preprocess_image(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path);
    cv::resize(image, image, cv::Size(224, 224));
    image.convertTo(image, CV_32F, 1.0 / 255);
    torch::Tensor tensor_image = torch::from_blob(
        image.data, {1, image.rows, image.cols, 3}, torch::kFloat);
    tensor_image = tensor_image.permute({0, 3, 1, 2}); // Convert to CxHxW
    return tensor_image.clone();
}
```

---

### 5. Compile and Run

1. Build the project using CMake:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

2. Run the compiled executable:
   ```bash
   ./age_prediction path/to/image.jpg
   ```

---

## Notes

- Ensure the input preprocessing in C++ matches the preprocessing used during training.
- Consider model quantization to reduce size and improve inference speed on edge devices.

By following these steps, you can seamlessly deploy the age prediction model in C++ for real-world applications.


# Base image with PyTorch
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set working directory
WORKDIR /workspace

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone the GitHub repository
RUN git clone https://github.com/anupvanarse/AgePredictor.git /workspace/AgePredictor

# Set up Python environment
WORKDIR /workspace/AgePredictor
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wandb scikit-learn pandas torchsummary tqdm
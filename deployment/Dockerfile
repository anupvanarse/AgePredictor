# Base image with PyTorch
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Set working directory
WORKDIR /workspace

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create AgePredictor folder
RUN mkdir -p /workspace/AgePredictor

# Clone the GitHub repository
RUN git clone https://${GITHUB_TOKEN}@github.com/anupvanarse/AgePredictor.git /workspace/AgePredictor

# Set AgePredictor as the working directory
WORKDIR /workspace/AgePredictor

# Set up Python environment
WORKDIR /workspace
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wandb scikit-learn pandas torchsummary tqdm

# Set default entrypoint to the AgePredictor folder
ENTRYPOINT ["/bin/bash"]

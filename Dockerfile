# Multi-stage Dockerfile for GPT-OSS Vision Training
# Optimized for RunPod and GPU training

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    datasets==2.14.0 \
    accelerate==0.24.0 \
    pillow \
    numpy \
    wandb \
    tensorboard \
    jupyter \
    ipywidgets

# Pre-download CLIP model to save time
RUN python -c "from transformers import CLIPVisionModel, CLIPImageProcessor; \
    CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32'); \
    CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32')"

# Copy your code
COPY . /workspace/gpt-oss-vision

# Set up environment
ENV PYTHONPATH=/workspace/gpt-oss-vision:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

# Create volumes for persistent storage
VOLUME ["/workspace/data", "/workspace/checkpoints"]

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Default command - can be overridden
CMD ["bash"]
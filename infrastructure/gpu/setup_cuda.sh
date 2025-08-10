#!/bin/bash

# YTEmpire GPU/CUDA Environment Setup Script
# Installs NVIDIA drivers, CUDA toolkit, and verifies GPU setup

set -e

echo "========================================="
echo "YTEmpire GPU/CUDA Environment Setup"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# System information
echo -e "\n${YELLOW}System Information:${NC}"
uname -a
lsb_release -a

# Step 1: Update system packages
echo -e "\n${YELLOW}Step 1: Updating system packages...${NC}"
sudo apt-get update
sudo apt-get upgrade -y

# Step 2: Install required dependencies
echo -e "\n${YELLOW}Step 2: Installing dependencies...${NC}"
sudo apt-get install -y \
    build-essential \
    dkms \
    software-properties-common \
    curl \
    wget \
    git \
    python3-pip \
    python3-dev

# Step 3: Check for NVIDIA GPU
echo -e "\n${YELLOW}Step 3: Checking for NVIDIA GPU...${NC}"
if ! lspci | grep -i nvidia > /dev/null; then
    echo -e "${RED}No NVIDIA GPU detected! Exiting...${NC}"
    exit 1
fi
echo -e "${GREEN}NVIDIA GPU detected:${NC}"
lspci | grep -i nvidia

# Step 4: Remove old NVIDIA drivers (if any)
echo -e "\n${YELLOW}Step 4: Removing old NVIDIA drivers...${NC}"
sudo apt-get remove --purge '^nvidia-.*' -y || true
sudo apt-get remove --purge '^libnvidia-.*' -y || true
sudo apt-get remove --purge '^cuda-.*' -y || true

# Step 5: Add NVIDIA package repositories
echo -e "\n${YELLOW}Step 5: Adding NVIDIA repositories...${NC}"

# Add NVIDIA driver repository
sudo add-apt-repository ppa:graphics-drivers/ppa -y

# Add CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Step 6: Install NVIDIA Driver (535 for RTX 5090 compatibility)
echo -e "\n${YELLOW}Step 6: Installing NVIDIA Driver 535...${NC}"
sudo apt-get install -y nvidia-driver-535

# Step 7: Install CUDA Toolkit 12.2
echo -e "\n${YELLOW}Step 7: Installing CUDA Toolkit 12.2...${NC}"
sudo apt-get install -y cuda-12-2

# Step 8: Install cuDNN
echo -e "\n${YELLOW}Step 8: Installing cuDNN...${NC}"
# Download cuDNN (requires NVIDIA developer account)
# For automated setup, we'll use apt
sudo apt-get install -y libcudnn8 libcudnn8-dev

# Step 9: Set up environment variables
echo -e "\n${YELLOW}Step 9: Setting up environment variables...${NC}"
cat >> ~/.bashrc << 'EOF'

# CUDA Environment Variables
export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-12.2
EOF

# Apply environment variables
export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-12.2

# Step 10: Install NVIDIA Docker support
echo -e "\n${YELLOW}Step 10: Installing NVIDIA Docker support...${NC}"
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Step 11: Install PyTorch with CUDA support
echo -e "\n${YELLOW}Step 11: Installing PyTorch with CUDA support...${NC}"
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 12: Install TensorFlow with CUDA support
echo -e "\n${YELLOW}Step 12: Installing TensorFlow with CUDA support...${NC}"
pip3 install tensorflow[and-cuda]

# Step 13: Verify installation
echo -e "\n${YELLOW}Step 13: Verifying installation...${NC}"

# Check NVIDIA driver
echo -e "\n${GREEN}NVIDIA Driver Version:${NC}"
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# Check CUDA version
echo -e "\n${GREEN}CUDA Version:${NC}"
nvcc --version

# Check GPU details
echo -e "\n${GREEN}GPU Details:${NC}"
nvidia-smi

# Test CUDA with simple program
echo -e "\n${GREEN}Testing CUDA compilation...${NC}"
cat > /tmp/test_cuda.cu << 'EOF'
#include <stdio.h>

__global__ void helloCUDA() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    printf("Hello from CPU!\n");
    helloCUDA<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
EOF

nvcc /tmp/test_cuda.cu -o /tmp/test_cuda
/tmp/test_cuda

# Test PyTorch CUDA
echo -e "\n${GREEN}Testing PyTorch CUDA...${NC}"
python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Test TensorFlow GPU
echo -e "\n${GREEN}Testing TensorFlow GPU...${NC}"
python3 -c "import tensorflow as tf; print(f'TensorFlow GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}'); print(f'GPU devices: {tf.config.list_physical_devices(\"GPU\")}')"

# Create verification report
echo -e "\n${YELLOW}Creating verification report...${NC}"
cat > gpu_setup_report.txt << EOF
YTEmpire GPU/CUDA Setup Report
Generated: $(date)
========================================

System Information:
$(uname -a)

GPU Information:
$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv)

CUDA Version:
$(nvcc --version | grep release)

PyTorch CUDA:
$(python3 -c "import torch; print(f'Available: {torch.cuda.is_available()}, Version: {torch.version.cuda}')" 2>/dev/null || echo "Not available")

TensorFlow GPU:
$(python3 -c "import tensorflow as tf; print(f'Available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')" 2>/dev/null || echo "Not available")

Docker GPU Support:
$(docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1 && echo "Working" || echo "Not working")

Environment Variables:
CUDA_HOME=$CUDA_HOME
PATH=$PATH
LD_LIBRARY_PATH=$LD_LIBRARY_PATH
EOF

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}GPU/CUDA Setup Complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Verification report saved to: gpu_setup_report.txt"
echo ""
echo -e "${YELLOW}Please reboot the system to ensure all drivers are loaded:${NC}"
echo "  sudo reboot"
echo ""
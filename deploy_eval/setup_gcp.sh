#!/bin/bash
# GCP GPU Instance Setup Script for LLaMA3-8B-GPTQ Model
# ======================================================

set -e

echo "=========================================="
echo "Setting up GPU environment for vLLM"
echo "=========================================="

# Update system
echo "[1/6] Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y build-essential python3-pip python3-venv git wget

# Install NVIDIA drivers and CUDA
echo "[2/6] Installing NVIDIA drivers..."
sudo apt-get install -y linux-headers-$(uname -r)

# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update -y

# Install CUDA toolkit (includes drivers)
echo "[3/6] Installing CUDA toolkit..."
sudo apt-get install -y cuda-toolkit-12-4 nvidia-driver-550

# Set up environment
echo "[4/6] Configuring environment..."
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Create Python virtual environment
echo "[5/6] Setting up Python environment..."
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate

# Install Python packages
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install vllm>=0.6.0 transformers accelerate auto-gptq optimum
pip install datasets evaluate rouge-score nltk scikit-learn
pip install fastapi uvicorn pydantic httpx pandas numpy tqdm

echo "[6/6] Setup complete!"
echo ""
echo "=========================================="
echo "IMPORTANT: Reboot required for GPU drivers"
echo "Run: sudo reboot"
echo ""
echo "After reboot, activate environment with:"
echo "  source ~/vllm-env/bin/activate"
echo ""
echo "Then run:"
echo "  python deploy_vllm.py --interactive"
echo "=========================================="

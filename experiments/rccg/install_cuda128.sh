#!/bin/bash
# Install CUDA 12.8 toolkit
if [ -f /usr/local/cuda-12.8/bin/nvcc ]; then
    echo "CUDA 12.8 already installed"
    exit 0
fi
apt-get update -qq
apt-get install -y cuda-toolkit-12-8
echo "CUDA 12.8 install complete"

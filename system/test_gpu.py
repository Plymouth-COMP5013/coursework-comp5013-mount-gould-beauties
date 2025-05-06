# Author: Reef Lakin
# Last Modified: 30.04.2025
# Description: A script to test the GPU availability on your device.
import torch

if torch.backends.mps.is_available():
    print("Using MPS (Metal) backend.")
elif torch.cuda.is_available():
    print("Using CUDA backend.")
else:
    print("Using CPU.")

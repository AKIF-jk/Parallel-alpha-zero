#!/bin/bash

# Alpha Zero General C++ - Google Colab Setup Script
# Run this script in a Colab cell: !bash colab_setup.sh

set -e

echo "=== Alpha Zero General C++ - GPU Setup ==="

# Check GPU availability
echo "Checking GPU..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Install pybind11 if needed
echo "Installing pybind11..."
apt-get update && apt-get install -y pybind11-dev || pip install pybind11

# Compile C++ extension
echo "Compiling C++ MCTS extension..."
cd /content/alpha-zero-general-cpp
python3 setup.py build_ext --inplace

# Test GPU
echo "Testing GPU inference..."
python3 -c "
import torch
from othello.pytorch.NNetGPU import NNetGPUWrapper
from othello.OthelloGame import OthelloGame

game = OthelloGame(6)
nnet = NNetGPUWrapper(game)
print('GPU NNet initialized successfully!')

# Quick test
import numpy as np
board = np.zeros((6, 6))
board[2,2] = 1
board[3,3] = -1
pi, v = nnet.predict(board)
print(f'Prediction test: pi shape={pi.shape}, v={v}')
"

echo "=== Setup Complete ==="
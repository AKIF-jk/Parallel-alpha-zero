# Google Colab GPU Setup for Alpha Zero

## Quick Start

1. **Upload project to Colab**
   - Upload this folder to your Google Drive or clone from GitHub

2. **Open the notebook** 
   - Open `alpha_zero_colab.ipynb` in Google Colab

3. **Run cells in order**
   - The notebook handles all GPU setup automatically

## Manual Setup (Alternative)

If you prefer running the script directly:

```python
# In Colab cell:
!git clone <your-repo>
%cd alpha-zero-general-cpp

# Install dependencies  
!pip install torch torchvision pybind11 coloredlogs tqdm

# Compile C++ extension
!python setup.py build_ext --inplace

# Run training
!python main_colab.py
```

## Files Added

| File | Description |
|------|-------------|
| `othello/pytorch/NNetGPU.py` | GPU-accelerated neural network wrapper |
| `main_colab.py` | Colab-compatible main script |
| `alpha_zero_colab.ipynb` | Ready-to-run Colab notebook |
| `colab_setup.sh` | Shell script for setup |

## GPU Features

- **CUDA acceleration**: Uses T4 GPU on Colab
- **DataParallel**: Multi-GPU support
- **Batched inference**: `predict_batch()` method for efficiency
- **cuDNN optimization**: Enabled for faster convolutions

## Checkpoints

Checkpoints are saved to `/content/drive/MyDrive/alpha_zero_checkpoints/` (Google Drive).
# Learning to See in the Dark — Implementation from Scratch

> Deep Learning for Perception | Final Semester Project  
> Implementation of Chen et al., CVPR 2018 — "Learning to See in the Dark"

---

## Project Overview

This project implements the **See-in-the-Dark (SID)** pipeline from scratch using PyTorch.  
A U-Net fully-convolutional network takes raw short-exposure Sony images and produces clean full-resolution sRGB output — replacing the entire traditional camera ISP pipeline.

**Our Results (3 epochs):**
| Metric | Ours (3 epochs) | Paper (4000 epochs) |
|--------|----------------|---------------------|
| PSNR   | 27.36 dB       | 28.88 dB            |
| SSIM   | 0.8087         | 0.787               |

---

## Repository Structure

```
├── DLP-Project4.ipynb          # Main Kaggle notebook (all code)
├── README.md                   # This file
├── report/
│   └── SID_Project_Report.pdf  # Final project report
├── checkpoints/
│   ├── best_model.pth          # Best model weights (lowest val loss)
│   └── latest_checkpoint.pth  # Full training state (model + optimizer)
├── results/
│   ├── loss_curve.png          # Training and validation loss across epochs
│   └── visual_results.png      # Input vs Output vs Ground Truth comparisons
└── dataset/
    ├── Sony_train_list.txt     # Training pairs annotation file
    ├── Sony_val_list.txt       # Validation pairs annotation file
    └── Sony_test_list.txt      # Test pairs annotation file
```

---

## Requirements

```
torch
torchvision
rawpy
numpy
matplotlib
tqdm
scikit-image
imageio
```

Install all dependencies:
```bash
pip install torch torchvision rawpy numpy matplotlib tqdm scikit-image imageio
```

---

## Dataset

Download the **Sony subset** of the SID dataset from Kaggle:
```
https://www.kaggle.com/datasets/moodoki/sid-sony
```

After downloading, the folder structure should be:
```
Sony/
  short/    ← short-exposure input .ARW files
  long/     ← long-exposure ground truth .ARW files
Sony_train_list.txt
Sony_val_list.txt
Sony_test_list.txt
```

---

## Running on Kaggle (Recommended)

This notebook was developed and trained on **Kaggle with GPU T4 x2**.

1. Go to [kaggle.com](https://kaggle.com) and create a new notebook
2. Add the dataset: search `moodoki/sid-sony` under **Add Input → Datasets**
3. Set **Accelerator → GPU T4 x2** under Settings
4. Upload `DLP-Project4.ipynb`
5. Click **Run All**

---

## Running Training Locally

### Step 1 — Set dataset paths
Edit the config in the last training cell:
```python
config = {
    'input_dir'     : '/path/to/Sony/short',
    'gt_dir'        : '/path/to/Sony/long',
    'train_txt'     : '/path/to/Sony_train_list.txt',
    'val_txt'       : '/path/to/Sony_val_list.txt',
    'test_txt'      : '/path/to/Sony_test_list.txt',
    'checkpoint_dir': './checkpoints',
    'num_epochs'    : 3,
    'batch_size'    : 1,
    'lr'            : 1e-4,
}
```

### Step 2 — Run training
```bash
jupyter notebook DLP-Project4.ipynb
# Run all cells from top to bottom
# Or run cells in this order:
# Cell 1  → Create checkpoint directory
# Cell 3  → Install rawpy + imports
# Cell 5  → pack_raw function
# Cell 7  → SIDDataset class
# Cell 9  → UNet architecture
# Cell 11 → train_model function
# Cell 12 → Start training (config + run)
```

Training automatically saves checkpoints after every epoch to `checkpoints/`.  
If training is interrupted, re-running will resume from the last saved epoch.

---

## Running Inference on a Single Image

Add this cell to the notebook after loading the model:

```python
import rawpy
import numpy as np
import torch

def run_inference(model, short_path, ratio, device):
    """
    Run inference on a single raw .ARW file.
    
    Args:
        model:      trained UNet model
        short_path: path to short-exposure .ARW file
        ratio:      amplification ratio (e.g. 100, 250, 300)
        device:     torch device
    
    Returns:
        output image as numpy uint8 array (H, W, 3)
    """
    model.eval()
    
    # Load and pack raw image
    with rawpy.imread(short_path) as raw:
        packed = pack_raw(raw)               # (H/2, W/2, 4)
    
    inp = packed * ratio
    inp = torch.from_numpy(
        inp.transpose(2, 0, 1)
    ).unsqueeze(0).to(device)                # (1, 4, H/2, W/2)
    
    with torch.no_grad():
        out = model(inp)                     # (1, 3, H, W)
    
    result = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    return result

# Example usage:
# Load model
model = UNet().to(device)
model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))

# Run on one image
output = run_inference(
    model      = model,
    short_path = '/path/to/Sony/short/00001_00_0.1s.ARW',
    ratio      = 100,   # exposure ratio from filename
    device     = device
)

# Save result
import imageio
imageio.imwrite('my_output.png', output)
print("Saved to my_output.png")
```

---

## Resuming Training from Checkpoint

The training loop automatically resumes if a checkpoint exists:

```python
# Just re-run the training cell — it will detect and load the checkpoint
model, train_losses, val_losses = train_model(config)
# Output: "Resuming from checkpoint... Resumed from epoch X"
```

To manually load the best model for evaluation:
```python
model = UNet().to(device)
model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
model.eval()
```

---

## Model Architecture Summary

```
Input:  (B, 4,  H/2, W/2)  — 4-channel packed Bayer at half resolution
         ↓
Encoder: 4 stages of DoubleConv + MaxPool
         (4→32→64→128→256 channels)
         ↓
Bottleneck: DoubleConv (512 channels)
         ↓
Decoder: 4 stages of ConvTranspose + Skip Connection + DoubleConv
         (256→128→64→32 channels)
         ↓
Output Head: Conv1×1 (32→12 ch) → PixelShuffle(×2) → Clamp(0,1)
         ↓
Output: (B, 3,  H,   W)   — full resolution sRGB image

Total Parameters: ~7.7 million
Loss Function:    L1 (Mean Absolute Error)
Optimizer:        Adam (lr=1e-4, β1=0.9, β2=0.999)
```

---

## Key Implementation Details

- **Black level subtraction**: Sony sensors have black level = 512; we subtract and normalise by (16383 - 512)
- **Bayer packing**: RGGB pattern packed into 4 channels — avoids information loss vs masking
- **Amplification ratio**: Derived from exposure time in filename (e.g. `0.1s` and `10s` → ratio = 100), capped at 300
- **PixelShuffle output**: 12 output channels rearranged to 3 full-resolution channels — avoids checkerboard artifacts
- **No residual connections**: Input and output are in different colour spaces (raw vs sRGB), so residual connections are not applicable

---

## Reference

Chen, C., Chen, Q., Xu, J., & Koltun, V. (2018).  
**Learning to See in the Dark.**  
*Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).*  
arXiv:1805.01934

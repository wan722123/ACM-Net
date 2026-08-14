ACM-Net

Official PyTorch implementation of **Adaptive Correlation Matching Network with Confidence-Guided B-Spline Free-Form Deformation for Unsupervised Deformable Brain MRI Registration**.

ACM-Net combines hierarchical feature extraction, adaptive correlation matching, entropy-based confidence estimation, and confidence-guided B-spline free-form deformation for coarse-to-fine medical image registration.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10 or later and a CUDA-enabled PyTorch installation are recommended.

## Data

The datasets are not redistributed. Each preprocessed subject should be stored as a `.pkl` file containing an image and its segmentation label. See `data/datasets.py` for the supported formats.

## Training

```bash
python train.py \
  --dataset lpba40 \
  --train-dir datasets/LPBA40/Train \
  --val-dir datasets/LPBA40/Val \
  --output-dir runs/lpba40
```

## Inference

```bash
python infer.py \
  --dataset lpba40 \
  --test-dir datasets/LPBA40/Val \
  --checkpoint runs/lpba40/best.pth.tar
```

## Note

This repository follows the architecture described in the manuscript. Checkpoints from earlier development versions may not be compatible with the current implementation.

## License

This project is released under the MIT License.


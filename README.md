# ACM-Net

Official PyTorch implementation of **Adaptive Correlation Matching Network with
Confidence-Guided B-Spline Free-Form Deformation for Unsupervised Deformable
Brain MRI Registration**.

> **Reproducibility notice:** this folder implements the manuscript literally:
> the ACM soft-correspondence field is used directly as the BFFD observation,
> and TASG uses three parallel axial branches. Checkpoints trained with the
> earlier observation-head/residual-head implementation are not compatible and
> must not be presented as weights for this architecture. Retraining and metric
> recomputation are required before this version can be released as the code
> underlying the reported experimental tables.

## Architecture

The public model names follow the terminology used in the manuscript:

- `TriAxialStripGating`: TASG modules at the first three encoder stages.
- `HeterogeneousAxialMixingAttention`: HAMA modules at the deepest two stages.
- `BidirectionalFeatureInteraction`: cross-image channel interaction in ACM.
- `AdaptiveCorrelationMatching`: local soft matching and entropy confidence.
- `ConfidenceGuidedBFFD`: confidence-weighted B-spline least-squares solver.
- `ACMNet`: shared encoder and coarse-to-fine registration network.

The `SLNet_BSplineSolve` alias is retained only for development-stage import
compatibility. It does not make historical checkpoints architecturally
compatible with this implementation.

## Environment

Python 3.10 or later and a CUDA-capable PyTorch installation are recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

For exact reproduction, record the CUDA, cuDNN, Python, and PyTorch versions
used on the target machine with `python environment.py`.

Before publishing a historical checkpoint, verify strict key compatibility and
the model output shapes:

```bash
python verify_checkpoint.py /path/to/best.pth.tar
```

## Data Format

Raw datasets are not redistributed. Preprocess each subject into a pickle file
containing either:

```python
(image, label)
```

or:

```python
{"image": image, "label": label}
```

`image` and `label` must be NumPy arrays with shape `(D, H, W)`. Images should
already be affinely aligned, cropped/resampled, skull-stripped where required,
and intensity-normalized according to the manuscript. Labels are used only for
validation and testing.

The loaders support two evaluation layouts:

- `subjects`: one `(image, label)` file per subject; all directed pairs are used.
- `prepaired`: each file stores `(moving, fixed, moving_label, fixed_label)`.

Suggested directory layout:

```text
datasets/
  LPBA40/
    Train/*.pkl
    Val/*.pkl
```

The manuscript input sizes are `(160, 192, 160)` for LPBA40 and Mindboggle,
`(160, 192, 224)` for OASIS, and `(160, 160, 192)` for ABCT.

## Training

LPBA40:

```bash
python train.py \
  --dataset lpba40 \
  --train-dir datasets/LPBA40/Train \
  --val-dir datasets/LPBA40/Val \
  --output-dir runs/lpba40
```

OASIS with pre-paired validation files:

```bash
python train.py \
  --dataset oasis \
  --train-dir datasets/OASIS/Train \
  --val-dir datasets/OASIS/Val \
  --val-format prepaired \
  --output-dir runs/oasis
```

Defaults follow the manuscript: batch size 1, learning rate `1e-4`, random seed
24, equal NCC and smoothness-loss weights, and 30/30/200/30 epochs for
LPBA40/Mindboggle/OASIS/ABCT. The optimizer retains `amsgrad=True` from the
original training script. Deterministic cuDNN settings are enabled by default.

## Evaluation

```bash
python infer.py \
  --dataset lpba40 \
  --test-dir datasets/LPBA40/Val \
  --checkpoint runs/lpba40/best.pth.tar \
  --output-dir evaluation/lpba40
```

The evaluation script reports DSC, HD95, ASSD, the non-positive Jacobian ratio,
and inference time. Use `--save-confidence` to save multi-scale confidence maps.
Dataset-specific physical spacing defaults to 1 mm for the brain datasets and
2 mm for ABCT; override it with `--spacing DZ DY DX` if needed.

HD95 and ASSD can vary with the surface-extraction, missing-label, averaging,
and spacing conventions. Confirm these conventions against the evaluation code
used to generate the manuscript tables before reporting reproduced numbers.

## Model Defaults

- Input image channels: 1; base feature channels: 16.
- Five encoder feature widths: `16, 32, 64, 128, 256`.
- ACM window: `2 x 2 x 2`.
- Control grids, coarse to fine: `10, 12, 16, 20, 24`.
- Maximum sampled points: `4096, 8192, 16384, 32768, 32768`.
- Sampling stride: `(2, 2, 2)`.
- BFFD regularization: `1e-2`.
- CG iterations: 20; relative-residual tolerance: `1e-6`.
- Entropy confidence exponent: 1.0.
- Solver precision: FP32.

## Checkpoints and Datasets

Do not commit datasets or large checkpoints to Git. GitHub rejects ordinary Git
objects larger than 100 MB. Publish trained checkpoints as GitHub Release assets,
or archive them on Zenodo/Hugging Face and link them here. Dataset access remains
subject to the licenses and terms of the original providers.

## License

A software license must be selected with all authors/institutions before public
release. Merely placing code on GitHub does not by itself grant reuse rights.

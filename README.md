Medical Image Registration Network： ACM-net

PyTorch implementation of a medical image registration framework.

Installation

pip install -r requirements.txt

Python 3.10+ and CUDA-enabled PyTorch are recommended.

Data

Datasets are not provided.
Please prepare the preprocessed data according to the format described
in data/datasets.py.

Training

python train.py --dataset lpba40 --output-dir runs/lpba40

Inference

python infer.py --dataset lpba40 --checkpoint runs/lpba40/best.pth.tar

License

MIT License.

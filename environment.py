"""Print the software and accelerator versions used for an experiment."""

import json
import platform

import numpy
import scipy
import torch


environment = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}

print(json.dumps(environment, indent=2))

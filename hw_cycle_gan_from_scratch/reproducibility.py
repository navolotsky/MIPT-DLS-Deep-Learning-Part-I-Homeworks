import os
import random

import numpy as np
import torch


SEED = 0


def enable_reproducibility(seed=SEED, raise_if_no_deterministic=True, cudnn_deterministic=True):
    # https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms
    if raise_if_no_deterministic:
        torch.set_deterministic(True)

    torch.manual_seed(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

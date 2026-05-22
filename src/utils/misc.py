import os
import random
import sys

import numpy as np
import torch

try:
    import PIL
except ImportError:
    PIL = None

try:
    import torchvision
except ImportError:
    torchvision = None

try:
    from torchsummary import summary
except ImportError:
    summary = None


_DEFAULT_DATA_ROOTS = {
    ".",
    "./data",
    "../data",
    "data",
}


def _normalize_path(path):
    if path is None:
        return None
    return os.path.normpath(path)


def _should_replace_data_root(data_root):
    normalized = _normalize_path(data_root)
    if normalized is None:
        return True
    return normalized in _DEFAULT_DATA_ROOTS

def get_model_state(model):
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    model_state = model.state_dict()
    return model_state

def define_gpus(dataset_name):
    del dataset_name
    visible_devices = os.environ.get("SAM_CALIBRATION_CUDA_VISIBLE_DEVICES")
    if visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        
def update_dataroot(dataset_name, data_root):
    del dataset_name
    override_root = os.environ.get("SAM_CALIBRATION_DATA_ROOT")
    if override_root:
        return override_root
    return data_root

def get_local_scratch_path_train():
    return os.path.join(get_local_scratch_path(), "train")

def get_local_scratch_path():
    candidates = (
        "LOCAL_SCRATCH_DIR",
        "SLURM_TMPDIR",
        "TMPDIR",
        "DATA_DIR_PATH",
    )
    for env_name in candidates:
        path = os.environ.get(env_name)
        if path:
            return path
    raise RuntimeError(
        "No scratch directory configured. Set one of "
        "LOCAL_SCRATCH_DIR, SLURM_TMPDIR, TMPDIR, or DATA_DIR_PATH."
    )

def print_libs_version():
    print("\nComputational Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    torchvision_version = torchvision.__version__ if torchvision is not None else "not installed"
    print("\tTorchvision: {}".format(torchvision_version))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    pil_version = PIL.__version__ if PIL is not None else "not installed"
    print("\tPIL: {}".format(pil_version))

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    
    # torch.use_deterministic_algorithms=True
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def print_model_summary(model, dataset):
    if summary is None:
        raise ImportError("torchsummary is required to print the model summary.")
    print("\nModel Arch Summary:")
    if dataset == 'cifar10':
        summary(model, (3, 32, 32))
    else:
        raise ValueError("Only 'cifar10' is supported in the paper reproduction code.")

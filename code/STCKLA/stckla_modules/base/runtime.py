"""base.py definitions moved here without algorithm changes."""

from contextlib import contextmanager
import torch


def set_seed(seed: int = 0, deterministic: bool = True):
    import os
    import random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)


@contextmanager
def sdp_kernel_ctx(device: torch.device, force_sdp: bool):
    if device.type != "cuda" or (not force_sdp):
        yield
        return
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=True,
            enable_math=True,
        ):
            yield
    except TypeError:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        yield


def amp_autocast_kwargs(amp: bool, amp_dtype: str):
    if (not amp) or (not torch.cuda.is_available()):
        return dict(enabled=False)
    dt = str(amp_dtype).lower().strip()
    if dt == "bf16":
        return dict(enabled=True, dtype=torch.bfloat16)
    return dict(enabled=True, dtype=torch.float16)

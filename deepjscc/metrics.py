from __future__ import annotations

import torch
from torch import Tensor


def per_image_psnr(reference: Tensor, reconstruction: Tensor) -> Tensor:
    mse = (reference - reconstruction).square().flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))

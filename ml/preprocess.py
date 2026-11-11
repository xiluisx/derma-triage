"""Preprocesamiento de evaluación. web/lib/preprocess.ts replica esto paso por paso.

Reglas (no cambiar sin cambiar el TS):
1. EXIF transpose, RGB, float32 en [0, 1].
2. Redimensionar para que el lado corto mida 224; el lado largo = floor(224 * largo / corto).
   Bilineal con antialias (F.interpolate, align_corners=False, antialias=True).
3. Recorte central 224x224 con offset floor((dim - 224) / 2).
4. Normalización ImageNet.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from ml.constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def resized_hw(h: int, w: int, target: int = IMAGE_SIZE) -> tuple[int, int]:
    if h <= w:
        return target, (target * w) // h
    return (target * h) // w, target


def load_rgb(path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def to_model_input(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> tensor float32 [3, 224, 224] normalizado."""
    arr = np.asarray(img, dtype=np.float32) / 255.0  # H, W, 3
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    h, w = x.shape[1], x.shape[2]
    nh, nw = resized_hw(h, w)
    x = F.interpolate(x[None], size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)[0]
    top = (nh - IMAGE_SIZE) // 2
    left = (nw - IMAGE_SIZE) // 2
    x = x[:, top : top + IMAGE_SIZE, left : left + IMAGE_SIZE]
    return (x - _MEAN) / _STD

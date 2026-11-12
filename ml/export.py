"""Exporta el modelo a ONNX con salidas (probs, cam) y escribe model-meta.json.

Real:  python -m ml.export --run runs/convnext_base_f0 --out artifacts
Dummy: python -m ml.export --dummy --out artifacts   (ConvNeXt ImageNet + head aleatorio, T=1)
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from ml.constants import CLASSES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, UNCERTAINTY_THRESHOLD
from ml.models import create_model


class CamWrapper(nn.Module):
    def __init__(self, model: nn.Module, temperature: float):
        super().__init__()
        self.model = model.eval()
        self.register_buffer("inv_t", torch.tensor(1.0 / temperature))
        head = model.head if hasattr(model, "head") else None
        self.is_convnext = head is not None and hasattr(head, "norm") and hasattr(head, "fc")
        fc = head.fc if self.is_convnext else model.classifier
        self.register_buffer("W", fc.weight.detach().clone())   # [K, C]
        self.register_buffer("b", fc.bias.detach().clone())     # [K]
        if self.is_convnext:
            self.register_buffer("gamma", head.norm.weight.detach().clone())
            self.register_buffer("beta", head.norm.bias.detach().clone())
            self.eps = float(head.norm.eps)

    def forward(self, x: torch.Tensor):
        f = self.model.forward_features(x)                       # [B, C, H, W]
        if self.is_convnext:
            p = f.mean(dim=(2, 3))                               # [B, C]
            mu = p.mean(dim=1, keepdim=True)
            sigma = torch.sqrt(p.var(dim=1, unbiased=False, keepdim=True) + self.eps)
            fn = (f - mu[..., None, None]) / sigma[..., None, None] * self.gamma[None, :, None, None]
            cam = torch.einsum("bchw,kc->bkhw", fn, self.W)
            logits = cam.mean(dim=(2, 3)) + self.W @ self.beta + self.b
        else:
            cam = torch.einsum("bchw,kc->bkhw", f, self.W)
            logits = cam.mean(dim=(2, 3)) + self.b
        probs = torch.softmax(logits * self.inv_t, dim=1)
        return probs, cam


def build(run_dir: Path | None, dummy: bool) -> tuple[CamWrapper, dict]:
    if dummy:
        model = create_model("convnext_tiny.fb_in1k", pretrained=True)
        return CamWrapper(model, 1.0), {"arch": "convnext_tiny.fb_in1k", "run_id": "dummy", "temperature": 1.0}
    info = json.loads((run_dir / "run.json").read_text())
    t = json.loads((run_dir / "temperature.json").read_text())["temperature"]
    model = create_model(info["model"], pretrained=False)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    return CamWrapper(model, t), {"arch": info["model"], "run_id": info["run_id"], "temperature": t}


def export_onnx(wrapper: CamWrapper, path: Path) -> None:
    dummy_in = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(wrapper, dummy_in, str(path), input_names=["input"], output_names=["probs", "cam"],
                      opset_version=17, dynamo=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wrapper, info = build(Path(args.run) if args.run else None, args.dummy)
    export_onnx(wrapper, out / "model-fp32.onnx")
    meta = {**info, "classes": CLASSES, "input_size": IMAGE_SIZE, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
            "uncertainty_threshold": UNCERTAINTY_THRESHOLD, "cam_size": 7,
            "variants": {"fp32": "model-fp32.onnx"}}
    (out / "model-meta.json").write_text(json.dumps(meta, indent=2))
    print(f"OK -> {out}")


if __name__ == "__main__":
    main()

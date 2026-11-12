"""Wrapper == modelo original, y ONNX == wrapper."""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from ml.export import CamWrapper, export_onnx
from ml.models import create_model


@pytest.mark.parametrize("arch", ["convnext_tiny", "efficientnet_b0"])
def test_wrapper_matches_model_and_onnx(arch, tmp_path: Path):
    torch.manual_seed(0)
    model = create_model(arch, pretrained=False).eval()
    wrapper = CamWrapper(model, temperature=1.7).eval()
    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        ref = torch.softmax(model(x) / 1.7, dim=1)
        probs, cam = wrapper(x)
    assert torch.allclose(probs, ref, atol=1e-5), (probs - ref).abs().max()
    assert cam.shape == (1, 3, 7, 7)

    path = tmp_path / "m.onnx"
    export_onnx(wrapper, path)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    o_probs, o_cam = sess.run(None, {"input": x.numpy()})
    assert np.abs(o_probs - probs.numpy()).max() < 1e-4
    assert np.abs(o_cam - cam.numpy()).max() < 1e-3

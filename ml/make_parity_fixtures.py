"""Genera fixtures para web/tests/parity.test.ts.

Uso: python -m ml.make_parity_fixtures --model artifacts/model-fp32.onnx
Salida (por directorio): NN.png, NN.input.bin (float32 [3,224,224]), expected.json (probs y cam por imagen)

- web/tests/fixtures/parity/        imágenes sintéticas (se versionan).
- web/tests/fixtures/parity-local/  imágenes reales del split de test (Fitzpatrick17k, CC BY-NC-SA):
  solo local, en .gitignore. Se genera únicamente si existe data/processed/splits.csv.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from ml.data.split import load_split
from ml.preprocess import load_rgb, to_model_input

SYNTHETIC_SIZES = [(640, 480), (333, 777), (224, 224), (1000, 300), (2000, 1500)]  # (w, h)


def synthetic(w: int, h: int, seed: int) -> Image.Image:
    """Gradiente + franjas cerca de Nyquist (estresan el antialias) + bloques 16px de ruido.
    Evita ruido por pixel: PNG no lo comprime y los fixtures versionados pesarían ~10 MB."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([xx / w, yy / h, (xx + yy) / (w + h)], -1) * 180
    f = rng.uniform(0.3, 1.2, 3)
    stripes = np.stack([np.sin(xx * f[0]), np.sin(yy * f[1]), np.sin((xx + yy) * f[2])], -1) * 35
    blocks = rng.normal(0, 20, (h // 16 + 1, w // 16 + 1, 3)).repeat(16, 0).repeat(16, 1)[:h, :w]
    return Image.fromarray(np.clip(base + stripes + blocks + 20, 0, 255).astype(np.uint8))


def write_fixtures(sess: ort.InferenceSession, images: list[Image.Image], out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    expected = []
    for i, img in enumerate(images):
        name = f"{i:02d}"
        img.save(out / f"{name}.png")
        x = to_model_input(Image.open(out / f"{name}.png").convert("RGB")).numpy().astype(np.float32)
        x.tofile(out / f"{name}.input.bin")
        probs, cam = sess.run(None, {"input": x[None]})
        expected.append({"file": f"{name}.png", "width": img.width, "height": img.height,
                         "probs": probs[0].tolist(), "cam": cam[0].reshape(-1).tolist()})
    (out / "expected.json").write_text(json.dumps(expected))
    return len(expected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/model-fp32.onnx")
    ap.add_argument("--out", default="web/tests/fixtures/parity")
    ap.add_argument("--out-local", default="web/tests/fixtures/parity-local")
    ap.add_argument("--data", default="data/processed")
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    images = [synthetic(w, h, i) for i, (w, h) in enumerate(SYNTHETIC_SIZES)]
    n = write_fixtures(sess, images, Path(args.out))
    print(f"{n} fixtures -> {args.out}")

    data = Path(args.data)
    if (data / "splits.csv").exists():
        test = load_split(data).query("split == 'test'").head(5)
        real = [load_rgb(data / "images" / f"{m}.jpg") for m in test.md5hash]
        n = write_fixtures(sess, real, Path(args.out_local))
        print(f"{n} fixtures (locales, no versionar) -> {args.out_local}")


if __name__ == "__main__":
    main()

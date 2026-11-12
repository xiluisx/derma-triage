"""int8 estático (QDQ, per-channel) con imágenes de calibración de val, y fp16.

Uso: python -m ml.quantize --artifacts artifacts --data data/processed
"""
import argparse
import json
from pathlib import Path

import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from ml.data.split import load_split
from ml.preprocess import load_rgb, to_model_input


class ValReader(CalibrationDataReader):
    def __init__(self, data: Path, n: int = 200):
        df = load_split(data).query("split == 'val'").sample(n, random_state=0)
        self.items = iter([to_model_input(load_rgb(data / "images" / f"{m}.jpg")).numpy()[None] for m in df.md5hash])

    def get_next(self):
        x = next(self.items, None)
        return None if x is None else {"input": x}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--data", default="data/processed")
    args = ap.parse_args()
    art, data = Path(args.artifacts), Path(args.data)
    fp32 = art / "model-fp32.onnx"

    pre = art / "model-fp32-pre.onnx"
    quant_pre_process(str(fp32), str(pre))
    quantize_static(str(pre), str(art / "model-int8.onnx"), ValReader(data), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    pre.unlink()

    m16 = float16.convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True,
                                            op_block_list=["ReduceMean"])  # sin esto ORT rechaza el grafo (tipos mixtos)
    onnx.save(m16, str(art / "model-fp16.onnx"))

    meta_path = art / "model-meta.json"
    meta = json.loads(meta_path.read_text())
    meta["variants"] = {"fp32": "model-fp32.onnx", "fp16": "model-fp16.onnx", "int8": "model-int8.onnx"}
    meta_path.write_text(json.dumps(meta, indent=2))
    for v in ["fp32", "fp16", "int8"]:
        print(v, f"{(art / f'model-{v}.onnx').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

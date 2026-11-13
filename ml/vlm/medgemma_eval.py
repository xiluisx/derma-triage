"""Evalúa MedGemma 4B zero-shot sobre el test set (o sobre la submuestra de Claude).

Uso: python -m ml.vlm.medgemma_eval --data data/processed --out runs/vlm_medgemma [--limit 5]
     python -m ml.vlm.medgemma_eval --subset runs/vlm_claude/test_index.csv   (misma submuestra que Claude)
Requiere aceptar los términos de google/medgemma-4b-it en Hugging Face y un token (HF_TOKEN / huggingface-cli login).
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from ml.data.split import load_split
from ml.preprocess import load_rgb
from ml.vlm.prompt import PROMPT, parse_label

MODEL_ID = "google/medgemma-4b-it"


def load():
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    if bf16 and vram_gb >= 14:  # 4B params en bf16 ≈ 8.6 GB + activaciones
        kw = {"dtype": torch.bfloat16}
    else:  # p. ej. RTX 5050 de 8 GB, o T4 sin bf16
        kw = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float32)}
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, device_map="auto", **kw).eval()
    return model, AutoProcessor.from_pretrained(MODEL_ID)


def select_test(data: Path, subset: str | None, limit: int | None):
    test = load_split(data).query("split == 'test'")
    if subset:  # misma submuestra (y orden) que Claude
        order = pd.read_csv(subset).md5hash
        test = test.set_index("md5hash").loc[order].reset_index()
    if limit:
        test = test.head(limit)
    return test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs/vlm_medgemma")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--subset", default=None, help="test_index.csv de la submuestra (p. ej. runs/vlm_claude/test_index.csv)")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)

    test = select_test(data, args.subset, args.limit)
    model, proc = load()  # si falla (modelo gated sin token) no se crea el directorio de salida
    out.mkdir(parents=True, exist_ok=True)

    preds, raw = [], []
    for m in tqdm(test.md5hash, desc="medgemma"):
        messages = [{"role": "user", "content": [{"type": "image", "image": load_rgb(data / "images" / f"{m}.jpg")},
                                                 {"type": "text", "text": PROMPT}]}]
        inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt").to(model.device)
        n_in = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        text = proc.decode(gen[0][n_in:], skip_special_tokens=True)
        raw.append(text)
        preds.append(parse_label(text))

    res = test[["md5hash", "label_idx", "tone"]].copy()
    res["pred_idx"] = preds
    res["raw"] = raw
    res.to_csv(out / "preds_test.csv", index=False)
    (out / "run.json").write_text(json.dumps({"run_id": "vlm_medgemma", "config": "vlm_medgemma", "seed": None,
                                              "source": "vlm", "model": MODEL_ID,
                                              "unparsed": int((res.pred_idx < 0).sum())}, indent=2))
    print(res.pred_idx.value_counts())


if __name__ == "__main__":
    main()

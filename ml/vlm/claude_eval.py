"""Evalúa Claude zero-shot sobre el test set con la Message Batches API.

Uso:
  python -m ml.vlm.claude_eval pilot  --limit 5 --out runs_debug/claude_pilot   (síncrono, mide costo real)
  python -m ml.vlm.claude_eval submit --data data/processed --out runs/vlm_claude
  python -m ml.vlm.claude_eval collect --out runs/vlm_claude      (cuando el batch termine)
Requiere ANTHROPIC_API_KEY en el entorno del comando (nunca se imprime ni se guarda).
"""
import argparse
import base64
import json
import time
from pathlib import Path

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from ml.constants import CLASS_TO_IDX, CLASSES, MALIGNANT_IDX
from ml.data.split import load_split
from ml.vlm.prompt import PROMPT

MODEL = "claude-opus-5-5"
SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": CLASSES}},
    "required": ["label"],
    "additionalProperties": False,
}
# Precios de Opus 5.5 por millón de tokens: estándar $4/$20; Batches (-50%) $2/$10.
PRICE_BATCH = (2e-6, 10e-6)

PER_GROUP_OTHERS = 100  # imágenes de I-II y de III-IV; V-VI va completo


def stratified_subset(test: pd.DataFrame) -> pd.DataFrame:
    """Todo V-VI + PER_GROUP_OTHERS de I-II y de III-IV (estratificado por clase) + TODOS los malignos de test.

    Los malignos se agregan completos para que el recall maligno por tono (métrica central) tenga IC
    comparables en los tres grupos; sin ellos I-II/III-IV quedaban con ~14 malignos. Consecuencia: la
    submuestra está enriquecida en malignos, así que la exactitud/macro-F1 global no es representativa del
    test; el recall por grupo sí. CNN y VLMs se evalúan sobre exactamente estas mismas imágenes.
    """
    parts = [test[test.tone_group == "V-VI"]]
    for g in ("I-II", "III-IV"):
        grp = test[test.tone_group == g]
        frac = min(1.0, PER_GROUP_OTHERS / len(grp))
        parts.append(grp.groupby("label_idx").sample(frac=frac, random_state=0))  # pandas 3: sin groupby.apply
    parts.append(test[(test.label_idx == MALIGNANT_IDX) & test.tone_group.isin(["I-II", "III-IV"])])
    sub = pd.concat(parts)
    return sub[~sub.md5hash.duplicated()]


def _params(data: Path, md5: str) -> MessageCreateParamsNonStreaming:
    b64 = base64.standard_b64encode((data / "images" / f"{md5}.jpg").read_bytes()).decode()
    return MessageCreateParamsNonStreaming(
        model=MODEL,
        max_tokens=4096,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": PROMPT},
        ]}],
    )


def _subset(data: Path) -> pd.DataFrame:
    return stratified_subset(load_split(data).query("split == 'test'"))


def submit(data: Path, out: Path, limit: int | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    test = _subset(data)
    if limit:
        test = test.head(limit)
    test[["md5hash", "label_idx", "tone"]].to_csv(out / "test_index.csv", index=False)
    print(f"Submuestra: {len(test)} imágenes\n{test.groupby(['tone_group', 'label_idx']).size()}")

    requests = [Request(custom_id=m, params=_params(data, m)) for m in test.md5hash]
    batch = anthropic.Anthropic().messages.batches.create(requests=requests)
    (out / "batch.json").write_text(json.dumps({"batch_id": batch.id, "model": MODEL}))
    print(f"Batch {batch.id} creado con {len(requests)} peticiones")


def pilot(data: Path, out: Path, limit: int) -> None:
    """Llamadas síncronas (sin Batches) a N imágenes mezcladas: mide tokens reales y proyecta el costo del batch."""
    out.mkdir(parents=True, exist_ok=True)
    full = _subset(data)
    pil = full.sample(n=limit, random_state=0)  # mezcla de grupos, no solo V-VI
    client = anthropic.Anthropic()
    rows, tin, tout = [], 0, 0
    for r in pil.itertuples():
        msg = client.messages.create(**_params(data, r.md5hash))
        tin += msg.usage.input_tokens
        tout += msg.usage.output_tokens
        pred = -1
        if msg.stop_reason != "refusal":
            text = next((b.text for b in msg.content if b.type == "text"), "")
            try:
                pred = CLASS_TO_IDX[json.loads(text)["label"]]
            except (json.JSONDecodeError, KeyError):
                pass
        rows.append({"md5hash": r.md5hash, "tone_group": r.tone_group, "label": CLASSES[r.label_idx],
                     "pred": CLASSES[pred] if pred >= 0 else "none", "stop": msg.stop_reason,
                     "in_tok": msg.usage.input_tokens, "out_tok": msg.usage.output_tokens})
    df = pd.DataFrame(rows)
    df.to_csv(out / "pilot.csv", index=False)
    print(df.to_string(index=False))
    n = len(df)
    per_img = (tin / n * PRICE_BATCH[0]) + (tout / n * PRICE_BATCH[1])
    print(f"\ntokens: in={tin} ({tin / n:.0f}/img) out={tout} ({tout / n:.0f}/img)")
    print(f"costo por imagen con Batches ≈ ${per_img:.5f}; submuestra completa = {len(full)} imágenes "
          f"-> proyección ≈ ${per_img * len(full):.3f} (estándar sin Batches ≈ ${per_img * len(full) * 2:.3f})")
    print(f"costo real del piloto (precio estándar) ≈ ${(tin * 4e-6 + tout * 20e-6):.4f}")


def collect(out: Path) -> None:
    client = anthropic.Anthropic()
    batch_id = json.loads((out / "batch.json").read_text())["batch_id"]
    while (b := client.messages.batches.retrieve(batch_id)).processing_status != "ended":
        print(f"status={b.processing_status} processing={b.request_counts.processing}")
        time.sleep(60)

    preds, outcomes = {}, {"succeeded": 0, "refusal": 0, "errored": 0, "other": 0}
    tokens = {"input": 0, "output": 0}
    for r in client.messages.batches.results(batch_id):  # llegan en cualquier orden: se indexa por custom_id
        if r.result.type != "succeeded":
            outcomes["errored" if r.result.type == "errored" else "other"] += 1
            preds[r.custom_id] = -1
            continue
        msg = r.result.message
        tokens["input"] += msg.usage.input_tokens
        tokens["output"] += msg.usage.output_tokens
        if msg.stop_reason == "refusal":
            outcomes["refusal"] += 1
            preds[r.custom_id] = -1
            continue
        text = next((blk.text for blk in msg.content if blk.type == "text"), "")
        try:
            preds[r.custom_id] = CLASS_TO_IDX[json.loads(text)["label"]]
            outcomes["succeeded"] += 1
        except (json.JSONDecodeError, KeyError):
            outcomes["other"] += 1
            preds[r.custom_id] = -1

    idx = pd.read_csv(out / "test_index.csv")
    idx["pred_idx"] = idx.md5hash.map(preds).fillna(-1).astype(int)
    idx.to_csv(out / "preds_test.csv", index=False)
    (out / "run.json").write_text(json.dumps({"run_id": "vlm_claude", "config": "vlm_claude", "seed": None,
                                              "source": "vlm", "model": MODEL, "outcomes": outcomes,
                                              "tokens": tokens}, indent=2))
    n = max(1, outcomes["succeeded"] + outcomes["refusal"])
    cost = tokens["input"] * PRICE_BATCH[0] + tokens["output"] * PRICE_BATCH[1]
    print(outcomes, tokens, f"costo ≈ ${cost:.3f} · por imagen ≈ ${cost / n:.5f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "submit", "collect"])
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs/vlm_claude")
    ap.add_argument("--limit", type=int, default=None, help="piloto: N imágenes de la submuestra")
    args = ap.parse_args()
    if args.cmd == "pilot":
        pilot(Path(args.data), Path(args.out), args.limit or 5)
    elif args.cmd == "submit":
        submit(Path(args.data), Path(args.out), args.limit)
    else:
        collect(Path(args.out))


if __name__ == "__main__":
    main()

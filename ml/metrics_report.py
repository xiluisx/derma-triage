"""Genera reports/metricas.md: hoja completa de métricas (en español) a partir de predicciones guardadas.

Uso: python -m ml.metrics_report [--out reports/metricas.md]
No entrena ni usa GPU.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

from ml.constants import CLASSES, GROUP_NAMES, MALIGNANT_IDX, tone_group
from ml.metrics import ece, softmax

ES = ["Benigno", "Maligno", "No neoplásico"]
LG = ["logit_0", "logit_1", "logit_2"]
RUNS = Path("runs")
L = [0, 1, 2]


def f3(x):
    return "n/d" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pct(x):
    return "n/d" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load(run, kind):
    """kind: 'pre' (softmax(logits)), 'T:<val>' (softmax(logits/T)), 'vlm'. Devuelve df, pred, probs."""
    df = pd.read_csv(RUNS / run / "preds_test.csv")
    if kind == "vlm":
        return df, df.pred_idx.to_numpy(), None
    T = 1.0 if kind == "pre" else float(kind[2:])
    p = softmax(df[LG].to_numpy() / T)
    return df, p.argmax(1), p


def mal_stats(y, pred):
    pos, neg = y == MALIGNANT_IDX, y != MALIGNANT_IDX
    rec = (pred[pos] == MALIGNANT_IDX).mean() if pos.any() else np.nan
    far = (pred[neg] == MALIGNANT_IDX).mean() if neg.any() else np.nan
    pm = pred == MALIGNANT_IDX
    prec = (y[pm] == MALIGNANT_IDX).mean() if pm.any() else np.nan
    return rec, prec, far


def brier(p, y):
    return float(((p - np.eye(3)[y]) ** 2).sum(1).mean())


def section_model(title, desc, df, pred, probs):
    y = df.label_idx.to_numpy()
    n = len(y)
    out = [f"### {title}", "", f"{desc} n = {n:,}.", ""]
    acc = (pred == y).mean()
    rows = [("Accuracy", f3(acc)), ("Balanced accuracy", f3(balanced_accuracy_score(y, pred))),
            ("Macro-F1", f3(f1_score(y, pred, labels=L, average="macro", zero_division=0))),
            ("Weighted-F1", f3(f1_score(y, pred, labels=L, average="weighted", zero_division=0)))]
    rec, prec, far = mal_stats(y, pred)
    rows += [("Sensibilidad maligno (vs resto)", f3(rec)), ("Especificidad maligno (vs resto)", f3(1 - far)),
             ("Precisión maligno", f3(prec)), ("Tasa de falsas alarmas (no-maligno predicho maligno)", f3(far))]
    if probs is not None:
        yb = (y == MALIGNANT_IDX).astype(int)
        rows += [("AUROC macro (OvR)", f3(roc_auc_score(y, probs, multi_class="ovr", average="macro")))]
        for k in L:
            rows.append((f"AUROC {ES[k]} vs resto", f3(roc_auc_score((y == k).astype(int), probs[:, k]))))
        rows += [("Average precision (PR-AUC) maligno vs resto", f3(average_precision_score(yb, probs[:, 1]))),
                 ("ECE (15 bins)", f3(ece(probs, y))), ("Brier (multiclase)", f3(brier(probs, y)))]
    out += [table(["Métrica", "Valor"], rows), ""]
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=L, zero_division=0)
    out += ["Por clase:", "", table(["Clase", "Precisión", "Recall", "F1", "Soporte"],
                                    [(ES[k], f3(p[k]), f3(r[k]), f3(f[k]), int(s[k])) for k in L]), ""]
    cm = confusion_matrix(y, pred, labels=L)
    out += ["Matriz de confusión (filas = real, columnas = predicho):", "",
            table(["Real \\ Pred"] + ES, [(ES[i],) + tuple(int(v) for v in cm[i]) for i in L]), ""]
    g = df.tone.map(tone_group).to_numpy()
    rows = []
    for name in GROUP_NAMES + ["unknown"]:
        m = g == name
        if not m.any():
            continue
        rc, pr, fa = mal_stats(y[m], pred[m])
        rows.append((name if name != "unknown" else "tono desconocido", int(m.sum()), int((y[m] == 1).sum()),
                     f3((pred[m] == y[m]).mean()),
                     f3(f1_score(y[m], pred[m], labels=L, average="macro", zero_division=0)), f3(rc), f3(pr), f3(fa)))
    out += ["Por grupo de tono:", "", table(["Grupo", "n", "n maligno", "Accuracy", "Macro-F1", "Recall maligno",
                                            "Precisión maligno", "Falsas alarmas"], rows), ""]
    return "\n".join(out), acc


def bootstrap(y, pred, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    a, f, r = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yy, pp = y[i], pred[i]
        a.append((yy == pp).mean())
        f.append(f1_score(yy, pp, labels=L, average="macro", zero_division=0))
        r.append(mal_stats(yy, pp)[0])
    ci = lambda v: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5))
    return ci(a), ci(f), ci(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/metricas.md")
    out_path = Path(ap.parse_args().out)

    meta = pd.read_csv("data/processed/meta.csv")
    sp = pd.read_csv("data/processed/splits.csv")
    ysp = sp.label_idx.to_numpy()
    cnt = np.bincount(ysp, minlength=3)
    triv = cnt.max() / cnt.sum()
    T0 = json.loads((RUNS / "convnext_base_f0/temperature.json").read_text())["temperature"]

    md = ["# Hoja de métricas del modelo", "",
          "Generado por `ml/metrics_report.py` a partir de predicciones guardadas (sin reentrenar). "
          "Clases: " + ", ".join(f"{ES[k]} ({CLASSES[k]})" for k in L) + ". Los números están en [0, 1] salvo que se indique porcentaje.", ""]

    # --- datos
    md += ["## 1. Datos y protocolo", "",
           f"- Imágenes totales en el dataset (`meta.csv`): **{len(meta):,}**.",
           f"- Tras deduplicar casi-duplicados por hash perceptual (`splits.csv`): **{len(sp):,}** "
           f"({len(meta) - len(sp):,} eliminadas).", ""]
    md += [table(["Clase", "n", "% del total"], [(ES[k], f"{cnt[k]:,}", pct(cnt[k] / cnt.sum())) for k in L]), ""]
    tg = sp.tone_group.value_counts()
    md += [table(["Grupo de tono", "n", "% del total"],
                 [(k, f"{tg.get(k, 0):,}", pct(tg.get(k, 0) / len(sp))) for k in GROUP_NAMES + ["unknown"]]),
           "", "(`unknown` = imágenes sin tono de piel anotado.)", ""]
    ratio = cnt.max() / cnt.min()
    md += [f"Desbalance de clases: la clase mayoritaria ({ES[int(cnt.argmax())]}, {cnt.max():,}) es "
           f"**{ratio:.1f} veces** la minoritaria ({ES[int(cnt.argmin())]}, {cnt.min():,}); "
           f"maligno es {pct(cnt[1] / cnt.sum())} del total.", "",
           "**Protocolo**: 5 folds estratificados por clase × tono; deduplicación de casi-duplicados por hash perceptual "
           "para evitar fuga entre train y test; por fold ~65% entrenamiento / ~15% validación / 20% test; "
           "selección del modelo por macro-F1 en validación; temperatura (calibración) ajustada en validación. "
           "El modelo principal se evalúa con las predicciones de test de los 5 folds juntas (CV agrupada): "
           "cada imagen se predice exactamente una vez con un modelo que no la vio.", "",
           "## 2. Por qué macro-F1 y no accuracy", "",
           f"Un clasificador trivial que siempre responde \"{ES[int(cnt.argmax())]}\" obtiene accuracy = "
           f"{cnt.max():,}/{cnt.sum():,} = **{f3(triv)}** ({pct(triv)}) sin detectar ni un solo cáncer (recall maligno = 0). "
           "Con este desbalance, la accuracy premia acertar la clase mayoritaria y oculta fallos en la clase minoritaria "
           "que importa clínicamente. El macro-F1 promedia el F1 de las tres clases con el mismo peso, así que fallar "
           "en maligno o benigno pesa tanto como fallar en la clase grande; por eso se usó como criterio de selección "
           "y métrica principal. Aun así se reporta accuracy y balanced accuracy para comparar con la línea base trivial.", ""]

    # --- glosario
    md += ["## 3. Cómo leer cada métrica", "",
           "- **Accuracy**: de cada 100 imágenes, cuántas clasifica bien (engañosa con clases desbalanceadas).",
           "- **Balanced accuracy**: promedio del recall de las tres clases; un clasificador trivial saca 0.333.",
           "- **Precisión (de una clase)**: de las imágenes que el modelo marca como esa clase, cuántas lo son de verdad. "
           "Precisión maligno = de cada 100 alarmas de cáncer, cuántas son cáncer real.",
           "- **Recall / sensibilidad**: de las imágenes que realmente son de la clase, cuántas detecta. "
           "Recall maligno = de cada 100 cánceres reales, cuántos detecta (los que faltan son cánceres perdidos).",
           "- **Especificidad (maligno vs resto)**: de cada 100 lesiones no malignas, cuántas deja correctamente sin alarma.",
           "- **Tasa de falsas alarmas**: 1 - especificidad; de cada 100 lesiones no malignas, cuántas se marcan como malignas.",
           "- **F1**: media armónica de precisión y recall; solo es alta si ambas lo son.",
           "- **Macro-F1**: promedio simple del F1 de las 3 clases (cada clase pesa igual).",
           "- **Weighted-F1**: promedio del F1 ponderado por el tamaño de cada clase (se parece más a la accuracy).",
           "- **AUROC**: probabilidad de que el modelo ponga una imagen positiva con puntaje más alto que una negativa; "
           "0.5 = azar, 1 = perfecto; no depende del umbral.",
           "- **Average precision (PR-AUC)**: área bajo la curva precisión-recall; más informativa que AUROC cuando la "
           "clase positiva es rara (la línea base al azar es la prevalencia de la clase).",
           "- **ECE**: diferencia promedio entre la confianza del modelo y su acierto real; 0 = probabilidades bien calibradas.",
           "- **Brier**: error cuadrático medio de las probabilidades (menor es mejor; 0 = perfecto).",
           "- **Matriz de confusión**: filas = clase real, columnas = clase predicha; la diagonal son los aciertos.", ""]

    # --- 1 MAIN
    md += ["## 4. Resultados por modelo", ""]
    df, pred, probs = load("convnext_base_cv", "pre")
    s, _ = section_model("4.1 PRINCIPAL: ConvNeXt-T base, validación cruzada de 5 folds (agrupada)",
                         "Calculado sobre las predicciones de test de los 5 folds juntas; probabilidades = softmax(logits) "
                         "(ya calibradas por fold con su temperatura).", df, pred, probs)
    md += [s]
    y = df.label_idx.to_numpy()
    ca, cf, cr = bootstrap(y, pred)
    rec = mal_stats(y, pred)[0]
    md += ["**IC 95% por bootstrap** (1000 remuestreos de imágenes, semilla 0; modelo principal, n = "
           f"{len(y):,}):", "",
           table(["Métrica", "Valor", "IC 95%"],
                 [("Accuracy", f3((pred == y).mean()), f"[{f3(ca[0])}, {f3(ca[1])}]"),
                  ("Macro-F1", f3(f1_score(y, pred, labels=L, average="macro")), f"[{f3(cf[0])}, {f3(cf[1])}]"),
                  ("Recall maligno", f3(rec), f"[{f3(cr[0])}, {f3(cr[1])}]")]), ""]
    main_vals = dict(acc=(pred == y).mean(), ci=(ca, cf, cr))

    df, pred, probs = load("convnext_coloraug_cv", "pre")
    s, _ = section_model("4.2 ConvNeXt-T + color augmentation, validación cruzada de 5 folds (agrupada)",
                         "Predicciones de test de los 5 folds juntas; probabilidades = softmax(logits) (pre-calibradas).",
                         df, pred, probs)
    md += [s]

    df, pred, probs = load("convnext_base_f0", f"T:{T0}")
    s, _ = section_model("4.3 Modelo desplegado: ConvNeXt-T base, test del fold 0",
                         f"Probabilidades = softmax(logits / T) con T = {T0:.3f} (ajustada en validación).", df, pred, probs)
    md += [s]
    y0 = df.label_idx.to_numpy()
    p1 = softmax(df[LG].to_numpy())
    md += ["Calibración antes y después de escalar por temperatura (mismo conjunto):", "",
           table(["Versión", "ECE (15 bins)", "Brier"],
                 [("Sin calibrar (T = 1)", f3(ece(p1, y0)), f3(brier(p1, y0))),
                  (f"Calibrada (T = {T0:.3f})", f3(ece(probs, y0)), f3(brier(probs, y0)))]), "",
           "La temperatura no cambia la clase predicha (argmax), solo la confianza; por eso accuracy/F1 son idénticos en ambas versiones.", ""]

    # --- 4 subset
    sub_n = len(df := pd.read_csv(RUNS / "cnn_vlm_subset/preds_test.csv"))
    md += ["## 5. Comparación en la misma submuestra (CNN vs modelos de lenguaje-visión)", "",
           f"**Atención: submuestra de {sub_n} imágenes del test del fold 0, enriquecida en malignos** "
           "(no es representativa de la prevalencia real, así que accuracy y precisión de maligno NO son comparables con las "
           "de la sección 4; sí son comparables entre los tres modelos de esta sección, que ven exactamente las mismas imágenes). "
           "Los VLM son zero-shot (sin entrenar en este dataset) y dan una etiqueta, no probabilidades: no tienen AUROC/ECE.", ""]
    sub_idx = df.md5hash
    ysub = df.label_idx.to_numpy()
    md += [f"Composición: " + ", ".join(f"{ES[k]} {int((ysub == k).sum())}" for k in L) +
           f" (maligno = {pct((ysub == 1).mean())}).", ""]
    for title, run, kind, d in [("5.1 CNN elegida (misma arquitectura que la desplegada)", "cnn_vlm_subset", f"T:{T0}",
                                 "Probabilidades = softmax(logits / T) con T de `convnext_base_f0`."),
                                ("5.2 Claude (zero-shot)", "vlm_claude", "vlm", "Predicciones de etiqueta del VLM."),
                                ("5.3 MedGemma 4B (zero-shot)", "vlm_medgemma", "vlm", "Predicciones de etiqueta del VLM.")]:
        d_, pr_, pb_ = load(run, kind)
        d_ = d_.set_index("md5hash").loc[sub_idx.values].reset_index()
        if kind != "vlm":
            pb_ = softmax(d_[LG].to_numpy() / T0)
            pr_ = pb_.argmax(1)
        else:
            pr_ = d_.pred_idx.to_numpy()
        s, _ = section_model(title, d, d_, pr_, pb_)
        md += [s]

    # --- 5 operating point
    op = json.loads((RUNS / "operating_point.json").read_text())
    cd = op["cnn_default"]
    md += ["## 6. Punto de operación: CNN vs VLM a igual tasa de falsas alarmas", "",
           f"Calculado sobre la misma submuestra de {sub_n} imágenes (enriquecida en malignos). Los VLM tienen un solo punto de "
           "operación; el umbral de \"maligno\" de la CNN se ajusta en validación para igualar la tasa de falsas alarmas (FPR) "
           "de cada VLM y se aplica al test. Diferencia = recall CNN - recall VLM, con IC 95% bootstrap pareado "
           "(por imagen). Los grupos por tono usan solo imágenes con tono conocido.", "",
           f"CNN con argmax (sin mover umbral): FPR = {f3(cd['fpr'])}, recall maligno = {f3(cd['recall'])}.", ""]
    rows = []
    for m in op["matched"]:
        d = m["recall_diff_cnn_minus_vlm"]
        rows.append((m["vlm"], f3(m["vlm_fpr"]), f3(m["vlm_recall"]), f3(m["cnn_fpr"]), f3(m["cnn_recall"]),
                     f"{d['overall']['diff']:+.3f} [{d['overall']['ci_low']:+.3f}, {d['overall']['ci_high']:+.3f}]",
                     f3(m["cnn_oracle_recall"])))
    md += [table(["VLM", "FPR VLM", "Recall VLM", "FPR CNN (umbral en val)", "Recall CNN", "Dif. recall (IC 95%)",
                  "Recall CNN (umbral oráculo en test, control)"], rows), "", "Diferencia de recall por grupo de tono:", ""]
    rows = []
    for m in op["matched"]:
        for g in GROUP_NAMES:
            d = m["recall_diff_cnn_minus_vlm"][g]
            rows.append((m["vlm"], g, m["vlm_by_group"][g]["n_malignant"], f3(m["vlm_by_group"][g]["recall"]),
                         f3(m["cnn_by_group"][g]["recall"]),
                         f"{d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"))
    md += [table(["VLM", "Grupo", "n malignos", "Recall VLM", "Recall CNN", "Dif. (IC 95%)"], rows), "",
           "Un IC que incluye 0 significa que la diferencia no es estadísticamente distinguible de cero con esos datos "
           "(en V-VI hay pocos malignos).", ""]

    # --- 6 folds
    md += ["## 7. Variabilidad entre folds", "",
           "Calculado sobre el test de cada fold por separado (n ≈ 3,270 por fold). Media ± desviación estándar (muestral, ddof = 1) "
           "entre los 5 folds. Brecha = recall maligno máximo entre grupos de tono menos el mínimo (de `metrics_test.json`). "
           "Accuracy calculada desde las predicciones del fold (argmax; la temperatura no la cambia).", ""]
    for cfg, name in [("convnext_base", "ConvNeXt-T base"), ("convnext_coloraug", "ConvNeXt-T + color aug")]:
        rows, vals = [], {k: [] for k in ["acc", "f1", "rec", "gap"]}
        for k in range(5):
            rd = RUNS / f"{cfg}_f{k}"
            mj = json.loads((rd / "metrics_test.json").read_text())
            pf = pd.read_csv(rd / "preds_test.csv")
            acc = (pf[LG].to_numpy().argmax(1) == pf.label_idx.to_numpy()).mean()
            gap = mj["malignant_recall_gap"]["value"]
            v = (acc, mj["overall"]["macro_f1"], mj["overall"]["malignant_recall"], gap)
            for key, x in zip(vals, v):
                vals[key].append(x)
            rows.append((k, f3(v[0]), f3(v[1]), f3(v[2]), f3(v[3])))
        sd = lambda a: np.std(a, ddof=1)
        rows.append(("**media ± std**",) + tuple(f"**{np.mean(vals[k]):.3f} ± {sd(vals[k]):.3f}**" for k in vals))
        md += [f"### {name}", "", table(["Fold", "Accuracy", "Macro-F1", "Recall maligno", "Brecha recall maligno entre tonos"], rows), ""]

    # --- 7 quant
    q = json.loads((RUNS / "quant_gate.json").read_text())
    md += ["## 8. Compuerta de cuantización", "",
           "Se compara el modelo cuantizado contra fp32 (`ml/quant_gate.py`). Aprueba si la pérdida de macro-F1 es <= 0.01 Y la pérdida "
           "de recall maligno es <= 0.02 en cada grupo de tono. Caída positiva = el cuantizado empeora.", "",
           table(["Variante", "Aprobada", "Tamaño (MB)", "Macro-F1 fp32", "Macro-F1 cuantizado", "Caída macro-F1",
                  "Caída recall maligno I-II / III-IV / V-VI"],
                 [(r["variant"], "sí" if r["passed"] else "no", f"{r['size_mb']:.1f}", f3(r["macro_f1_fp32"]),
                   f3(r["macro_f1_quant"]), f"{r['macro_f1_drop']:+.3f}",
                   " / ".join(f"{r['malignant_recall_drop_by_group'][g]:+.3f}" for g in GROUP_NAMES)) for r in q]), ""]

    # --- 8 comparisons
    comp = json.loads((RUNS / "comparisons.json").read_text())
    md += ["## 9. Comparación pareada de mitigación (color aug vs base)", "",
           "Diferencia = color aug - base (b - a en `ml/compare.py`; negativo = color aug peor), sobre las predicciones CV agrupadas (n = 16,352), IC 95% bootstrap pareado (de `comparisons.json`). "
           "Un IC que excluye 0 indica diferencia sistemática.", "",
           table(["Métrica", "Diferencia (color aug - base)", "IC 95%", "Excluye 0"],
                 [(c["metric"], f"{c['diff']:+.3f}", f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]",
                   "sí" if c["ci_low"] > 0 or c["ci_high"] < 0 else "no")
                  for c in comp if c["config_a"] == "convnext_base_cv" and c["config_b"] == "convnext_coloraug_cv"]), ""]

    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("\n".join(md) + "\n")
    print(f"escrito {out_path}")
    print("trivial", triv, "main acc", main_vals["acc"], main_vals["ci"])


if __name__ == "__main__":
    main()

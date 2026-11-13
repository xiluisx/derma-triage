import type { AuditSummary, Comparison } from "@/lib/audit-schema";

const pts = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(1)}`;
const label = (m: string) =>
  m === "macro_f1" ? "Macro-F1 (acierto general)" : m === "malignant_recall_gap" ? "Brecha entre tonos" : m.replace("malignant_recall_", "Recall de cáncer, tono ");
const conclusive = (c: { ci_low: number; ci_high: number }) => c.ci_low > 0 || c.ci_high < 0;

const MITIGATION_BASE = "convnext_base_cv";

type Pair = { a: string; b: string; rows: Comparison[] };

function groupPairs(comparisons: Comparison[]): Pair[] {
  const pairs: Pair[] = [];
  for (const c of comparisons) {
    const p = pairs.find((x) => x.a === c.config_a && x.b === c.config_b);
    if (p) p.rows.push(c);
    else pairs.push({ a: c.config_a, b: c.config_b, rows: [c] });
  }
  // La pareja de mitigación (validación cruzada) va primero; el resto conserva su orden.
  return pairs.sort((x, y) => Number(y.a === MITIGATION_BASE) - Number(x.a === MITIGATION_BASE));
}

function PairBlock({ pair, summary }: { pair: Pair; summary: AuditSummary }) {
  const name = (c: string) => summary.experiments.find((e) => e.config === c)?.display_name ?? c;
  const isCv = pair.a.endsWith("_cv") && pair.b.endsWith("_cv");
  const gap = pair.rows.find((c) => c.metric === "malignant_recall_gap");
  const f1 = pair.rows.find((c) => c.metric === "macro_f1");
  const tones = pair.rows.filter((c) => c.metric.startsWith("malignant_recall_") && c.metric !== "malignant_recall_gap");
  const fairnessMoved = pair.rows.some((c) => c.metric !== "macro_f1" && conclusive(c));

  let conclusion: string;
  if (isCv) {
    conclusion =
      (fairnessMoved
        ? "Al menos una métrica de equidad cambió de forma concluyente; revisa la tabla."
        : "No. Ningún cambio en el recall de cáncer por tono ni en la brecha es concluyente (todos los intervalos incluyen 0).") +
      (f1 && conclusive(f1) && f1.diff < 0 ? " Además, el acierto general empeoró un poco." : "") +
      (gap && !conclusive(gap) ? " Esta mitigación barata no cierra la brecha." : "");
  } else {
    const parts: string[] = [];
    if (f1) parts.push(conclusive(f1) ? `El acierto general (macro-F1) es ${f1.diff < 0 ? "menor" : "mayor"} que el de la CNN (${pts(f1.diff)} pts, concluyente).` : "El acierto general no difiere de forma concluyente de la CNN.");
    const sig = tones.filter(conclusive);
    if (sig.length === 0) parts.push("Ninguna diferencia de recall por tono es concluyente.");
    else
      parts.push(
        sig.map((c) => `En el tono ${c.metric.replace("malignant_recall_", "")} detecta ${c.diff > 0 ? "más" : "menos"} cánceres que la CNN (${pts(c.diff)} pts, concluyente)`).join("; ") + "."
      );
    conclusion = parts.join(" ");
  }

  return (
    <div className="flex flex-col gap-3">
      <h3 className="font-medium">{isCv ? "¿La mitigación redujo el sesgo?" : `${name(pair.b)} vs ${name(pair.a)}`}</h3>
      <p className="text-sm">
        {isCv
          ? `${name(pair.b)} vs ${name(pair.a)}, sobre las mismas imágenes (validación cruzada de 5 folds, diferencia pareada).`
          : `Misma submuestra de 990 imágenes del test del fold 0 (con todos los cánceres de ese test), diferencia pareada. Cambio = ${name(pair.b)} − ${name(pair.a)}.`}
      </p>
      <p className="rounded-lg border p-3 text-sm">{conclusion}</p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left border-b">
              <th className="py-2 pr-4">Métrica</th>
              <th className="py-2 pr-4">Cambio</th>
              <th className="py-2 pr-4">IC 95%</th>
              <th className="py-2">¿Concluyente?</th>
            </tr>
          </thead>
          <tbody>
            {pair.rows.map((c) => (
              <tr key={c.metric} className="border-b last:border-0">
                <td className="py-2 pr-4">{label(c.metric)}</td>
                <td className="py-2 pr-4 tabular-nums">{pts(c.diff)} pts</td>
                <td className="py-2 pr-4 tabular-nums">[{pts(c.ci_low)}, {pts(c.ci_high)}]</td>
                <td className="py-2">{conclusive(c) ? "Sí" : "No"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function ComparisonCard({ summary }: { summary: AuditSummary }) {
  if (summary.comparisons.length === 0) return null;
  const pairs = groupPairs(summary.comparisons);
  return (
    <section className="flex flex-col gap-8">
      <h2 className="text-lg font-semibold">Comparaciones</h2>
      {pairs.map((p) => (
        <PairBlock key={`${p.a}|${p.b}`} pair={p} summary={summary} />
      ))}
      <p className="text-xs text-neutral-500">Concluyente = el intervalo no incluye 0. Para la brecha, un valor negativo sería una mejora.</p>
    </section>
  );
}

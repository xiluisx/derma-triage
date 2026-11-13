import type { AuditSummary } from "@/lib/audit-schema";

const pts = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(1)}`;
const label = (m: string) =>
  m === "macro_f1" ? "Macro-F1 (acierto general)" : m === "malignant_recall_gap" ? "Brecha entre tonos" : m.replace("malignant_recall_", "Recall de cáncer, tono ");
const conclusive = (c: { ci_low: number; ci_high: number }) => c.ci_low > 0 || c.ci_high < 0;

export function ComparisonCard({ summary }: { summary: AuditSummary }) {
  if (summary.comparisons.length === 0) return null;
  const name = (c: string) => summary.experiments.find((e) => e.config === c)?.display_name ?? c;
  const { config_a, config_b } = summary.comparisons[0];
  const gap = summary.comparisons.find((c) => c.metric === "malignant_recall_gap");
  const f1 = summary.comparisons.find((c) => c.metric === "macro_f1");
  const fairnessMoved = summary.comparisons.some((c) => c.metric !== "macro_f1" && conclusive(c));
  return (
    <section className="flex flex-col gap-3">
      <h2 className="text-lg font-semibold">¿La mitigación redujo el sesgo?</h2>
      <p className="text-sm">
        {name(config_b)} vs {name(config_a)}, sobre las mismas imágenes (validación cruzada de 5 folds, diferencia pareada).
      </p>
      <p className="rounded-lg border p-3 text-sm">
        {fairnessMoved
          ? "Al menos una métrica de equidad cambió de forma concluyente; revisa la tabla."
          : "No. Ningún cambio en el recall de cáncer por tono ni en la brecha es concluyente (todos los intervalos incluyen 0)."}
        {f1 && conclusive(f1) && f1.diff < 0 && " Además, el acierto general empeoró un poco."}
        {gap && !conclusive(gap) && " Esta mitigación barata no cierra la brecha."}
      </p>
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
            {summary.comparisons.map((c) => (
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
      <p className="text-xs text-neutral-500">Concluyente = el intervalo no incluye 0. Para la brecha, un valor negativo sería una mejora.</p>
    </section>
  );
}

import type { AuditSummary } from "@/lib/audit-schema";

export function QuantizationCard({ summary }: { summary: AuditSummary }) {
  if (summary.quantization.length === 0) return null;
  return (
    <section className="flex flex-col gap-3">
      <h2 className="text-lg font-semibold">¿Comprimir el modelo afecta a algún grupo?</h2>
      <p className="text-sm">
        Para que corra en tu navegador hay que reducir el modelo. Probamos dos formas y la app usa la que pasó la prueba.
      </p>
      <div className="grid gap-4 sm:grid-cols-2">
        {summary.quantization.map((q) => (
          <article key={q.variant} className="rounded-lg border p-4 text-sm flex flex-col gap-1">
            <p className="font-medium">
              {q.variant.toUpperCase()} · {q.size_mb.toFixed(0)} MB
            </p>
            <p className={q.passed ? "font-medium text-green-700 dark:text-green-400" : "font-medium text-red-700 dark:text-red-400"}>
              {q.passed ? "✓ Pasa la prueba (la app usa esta)" : "✗ No pasa la prueba (descartado)"}
            </p>
            <p>
              Macro-F1: {(q.macro_f1_fp32 * 100).toFixed(1)}% → {(q.macro_f1_quant * 100).toFixed(1)}% ({q.macro_f1_drop > 0 ? "−" : "+"}
              {Math.abs(q.macro_f1_drop * 100).toFixed(1)} pts)
            </p>
            <ul>
              {Object.entries(q.malignant_recall_drop_by_group).map(([g, d]) => (
                <li key={g}>
                  Recall de cáncer {g}: {d > 0 ? "−" : "+"}
                  {Math.abs(d * 100).toFixed(1)} pts
                </li>
              ))}
            </ul>
          </article>
        ))}
      </div>
      <p className="text-xs text-neutral-500">Prueba: pérdida ≤ 1 pt de macro-F1 y ≤ 2 pts de recall de cáncer en cada grupo.</p>
    </section>
  );
}

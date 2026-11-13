import type { AuditSummary, ExperimentSummary } from "@/lib/audit-schema";

const pct = (m: { mean: number; std: number } | null, runs: number) =>
  m ? `${(m.mean * 100).toFixed(1)}${runs > 1 && m.std > 0 ? ` ± ${(m.std * 100).toFixed(1)}` : ""}%` : "—";

function Table({ rows }: { rows: ExperimentSummary[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left border-b">
            <th className="py-2 pr-4">Modelo</th>
            <th className="py-2 pr-4">Macro-F1</th>
            <th className="py-2 pr-4">Recall de cáncer</th>
            <th className="py-2 pr-4">Brecha entre tonos</th>
            <th className="py-2 pr-4">ECE</th>
            <th className="py-2">Corridas</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((e) => (
            <tr key={e.config} className="border-b last:border-0">
              <td className="py-2 pr-4">{e.display_name}</td>
              <td className="py-2 pr-4 tabular-nums">{pct(e.macro_f1, e.n_runs)}</td>
              <td className="py-2 pr-4 tabular-nums">{pct(e.malignant_recall, e.n_runs)}</td>
              <td className="py-2 pr-4 tabular-nums">{pct(e.malignant_recall_gap, e.n_runs)}</td>
              <td className="py-2 pr-4 tabular-nums">{pct(e.ece, e.n_runs)}</td>
              <td className="py-2 tabular-nums">{e.n_runs}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ExperimentTable({ summary }: { summary: AuditSummary }) {
  const cvRows = summary.experiments.filter((e) => e.config.endsWith("_cv"));
  const explore = summary.experiments.filter((e) => !e.config.endsWith("_cv"));
  return (
    <section className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <h2 className="text-lg font-semibold">Todas las métricas</h2>
        <h3 className="font-medium">Evidencia principal: validación cruzada de 5 folds</h3>
        <p className="text-xs text-neutral-500">
          Cada fila es una sola corrida agrupada (todas las imágenes evaluadas una vez), por eso no lleva desviación estándar.
        </p>
        <Table rows={cvRows} />
      </div>
      <div className="flex flex-col gap-2">
        <h3 className="font-medium">Exploración (solo fold 0, o varias semillas)</h3>
        <p className="text-xs text-neutral-500">
          Sirvió para elegir qué probar. Con una sola corrida no hay forma de medir la variación, así que no se sacan conclusiones de estas filas.
        </p>
        <Table rows={explore} />
      </div>
      <p className="text-xs text-neutral-500">
        Brecha = recall máximo − recall mínimo entre grupos de tono (menos es más equitativo). ECE = error de calibración de las probabilidades (menos es mejor). ± = desviación estándar entre corridas.
      </p>
    </section>
  );
}

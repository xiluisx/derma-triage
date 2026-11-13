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

const VLM_CONFIGS = ["cnn_vlm_subset", "vlm_claude", "vlm_medgemma"];
const p1 = (v: number) => `${(v * 100).toFixed(1)}%`;
const sgn = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(1)}`;
const sig = (c: { ci_low: number; ci_high: number }) => c.ci_low > 0 || c.ci_high < 0;

function VlmSection({ summary, rows }: { summary: AuditSummary; rows: ExperimentSummary[] }) {
  const cnn = rows.find((e) => e.config === "cnn_vlm_subset");
  const vlms = rows.filter((e) => e.config !== "cnn_vlm_subset");
  if (!cnn || vlms.length === 0) return null;
  const cmp = (cfg: string, metric: string) => summary.comparisons.find((c) => c.config_a === cnn.config && c.config_b === cfg && c.metric === metric);
  const toneGroups = summary.groups;
  const first = toneGroups[0];
  const last = toneGroups[toneGroups.length - 1];
  const lower = [cnn, ...vlms].filter((e) => e.malignant_recall_by_group[last].mean < e.malignant_recall_by_group[first].mean);
  const gapAll = lower.length === vlms.length + 1;
  const recallUp = vlms.filter((e) => e.malignant_recall.mean > cnn.malignant_recall.mean);
  const f1Down = vlms.filter((e) => {
    const c = cmp(e.config, "macro_f1");
    return c && sig(c) && c.diff < 0;
  });
  const lastTone = last;
  const strongest = vlms.filter((e) => {
    const c = cmp(e.config, `malignant_recall_${lastTone}`);
    return c && sig(c) && c.diff > 0;
  });
  return (
    <div className="flex flex-col gap-3">
      <h3 className="font-medium">CNN vs modelos de lenguaje con visión (misma submuestra)</h3>
      <p className="text-xs text-neutral-500">
        La submuestra (990 imágenes) incluye todos los casos malignos del test del fold 0, así que el recall de cáncer por tono tiene intervalos usables. Por eso tiene más cánceres que el test completo: su macro-F1 global no es representativo y no es comparable con las filas de validación cruzada. Compara solo dentro de esta sección.
      </p>
      <Table rows={rows} />
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <caption className="text-left text-xs text-neutral-500 pb-1">Recall de cáncer por tono en la submuestra (estimación puntual; los intervalos de las diferencias están en la comparación de arriba)</caption>
          <thead>
            <tr className="text-left border-b">
              <th className="py-2 pr-4">Modelo</th>
              {toneGroups.map((g) => (
                <th key={g} className="py-2 pr-4">Tono {g}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((e) => (
              <tr key={e.config} className="border-b last:border-0">
                <td className="py-2 pr-4">{e.display_name}</td>
                {toneGroups.map((g) => (
                  <td key={g} className="py-2 pr-4 tabular-nums">{e.malignant_recall_by_group[g] ? p1(e.malignant_recall_by_group[g].mean) : "—"}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="rounded-lg border p-3 text-sm flex flex-col gap-2">
        <p>
          {gapAll
            ? "La brecha por tono aparece en todos los modelos: todos detectan menos cánceres en V–VI que en I–II. "
            : `Detectan menos cánceres en V–VI que en I–II: ${lower.map((e) => e.display_name).join(", ")}. En los demás no ocurre en esta submuestra, pero con tan pocos casos en V–VI esa diferencia es incierta. `}
          Todos tienen incertidumbre alta en V–VI por pocos casos.
        </p>
        {recallUp.length > 0 && f1Down.length > 0 && (
          <p>
            Los modelos zero-shot cambian acierto general por sensibilidad: {recallUp.map((e) => `${e.display_name} detecta más cánceres (${p1(e.malignant_recall.mean)} vs ${p1(cnn.malignant_recall.mean)})`).join("; ")}, pero su macro-F1 es menor (
            {f1Down.map((e) => `${e.display_name} ${sgn(cmp(e.config, "macro_f1")!.diff)} pts`).join("; ")}, concluyente). Detectan más malignos a costa de más falsas alarmas.
          </p>
        )}
        {strongest.length > 0 && (
          <p>
            En V–VI, {strongest.map((e) => `${e.display_name} supera a la CNN en recall (${sgn(cmp(e.config, `malignant_recall_${lastTone}`)!.diff)} pts, concluyente)`).join("; ")}; aun así, esa ganancia viene con su menor acierto general, así que no es una mejora sin costo.
          </p>
        )}
        <p>Ninguno sustituye a la CNN afinada en el desempeño general.</p>
      </div>
    </div>
  );
}

export function ExperimentTable({ summary }: { summary: AuditSummary }) {
  const cvRows = summary.experiments.filter((e) => e.config.endsWith("_cv"));
  const vlmRows = VLM_CONFIGS.map((c) => summary.experiments.find((e) => e.config === c)).filter((e): e is ExperimentSummary => !!e);
  const explore = summary.experiments.filter((e) => !e.config.endsWith("_cv") && !VLM_CONFIGS.includes(e.config));
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
      <VlmSection summary={summary} rows={vlmRows} />
      <p className="text-xs text-neutral-500">
        Brecha = recall máximo − recall mínimo entre grupos de tono (menos es más equitativo). ECE = error de calibración de las probabilidades (menos es mejor). ± = desviación estándar entre corridas.
      </p>
    </section>
  );
}

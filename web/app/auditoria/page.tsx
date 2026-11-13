"use client";
import { useEffect, useState } from "react";
import { ComparisonCard } from "@/components/audit/ComparisonCard";
import { ExperimentTable } from "@/components/audit/ExperimentTable";
import { QuantizationCard } from "@/components/audit/QuantizationCard";
import { RecallByGroupChart } from "@/components/audit/RecallByGroupChart";
import { AuditSummary } from "@/lib/audit-schema";
import { CvGroups } from "@/lib/cv-groups-schema";

async function load<T>(url: string, parse: (j: unknown) => T): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return parse(await r.json());
}

export default function AuditPage() {
  const [data, setData] = useState<{ summary: AuditSummary; cv: CvGroups } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      load("/audit/summary.json", (j) => AuditSummary.parse(j)),
      load("/audit/cv_groups.json", (j) => CvGroups.parse(j)),
    ])
      .then(([summary, cv]) => setData({ summary, cv }))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error)
    return (
      <p role="alert" className="rounded-lg border border-red-500 p-4 text-sm">
        No se pudo cargar la auditoría (los datos faltan o no cumplen el formato esperado). Detalle: {error.slice(0, 300)}
      </p>
    );
  if (!data) return <p>Cargando…</p>;
  const { summary, cv } = data;

  return (
    <div className="flex flex-col gap-10">
      <header className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold">Auditoría de sesgo por tono de piel</h1>
        <p className="text-sm text-neutral-600 dark:text-neutral-400">
          Evaluación sobre Fitzpatrick17k, agrupada por tono de piel (escala Fitzpatrick I–VI). Resultados generados el{" "}
          {new Date(summary.generated_at).toLocaleDateString("es-MX")}.
        </p>
        <p className="text-sm">
          En resumen: el modelo detecta menos cánceres en pieles oscuras (V–VI) que en claras, la diferencia es real pero con mucha incertidumbre por pocos casos, y una mitigación sencilla (aumento de color)
          no la cerró.
        </p>
        <details className="rounded-lg border p-3 text-sm">
          <summary className="cursor-pointer font-medium">Cómo leer esto</summary>
          <ul className="mt-2 list-disc pl-5 flex flex-col gap-1">
            <li>
              <strong>Recall</strong>: de cada 100 cánceres reales, cuántos detecta el modelo. Un recall bajo significa más cánceres que pasan sin alerta.
            </li>
            <li>
              <strong>IC 95%</strong>: rango donde probablemente está el valor real. Un intervalo ancho (pocos casos) significa más incertidumbre; si dos intervalos se traslapan mucho, la diferencia puede ser ruido.
            </li>
            <li>Esto es un estudio académico, no una prueba clínica.</li>
          </ul>
        </details>
      </header>
      <RecallByGroupChart summary={summary} cv={cv} />
      <ComparisonCard summary={summary} />
      <ExperimentTable summary={summary} />
      <QuantizationCard summary={summary} />
    </div>
  );
}

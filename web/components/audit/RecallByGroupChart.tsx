import type { AuditSummary } from "@/lib/audit-schema";
import type { CvGroups } from "@/lib/cv-groups-schema";

const LO = 0.4;
const HI = 0.9;
const TICKS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
const pos = (v: number) => `${((Math.min(Math.max(v, LO), HI) - LO) / (HI - LO)) * 100}%`;
const p1 = (v: number) => `${(v * 100).toFixed(1)}%`;
const SERIES = [
  { key: "convnext_base_cv", color: "var(--series-1)", round: true },
  { key: "convnext_coloraug_cv", color: "var(--series-2)", round: false },
] as const;

export function RecallByGroupChart({ summary, cv }: { summary: AuditSummary; cv: CvGroups }) {
  const name = (c: string) => summary.experiments.find((e) => e.config === c)?.display_name ?? c;
  const series = SERIES.filter((s) => cv.runs[s.key]);
  const base = cv.runs.convnext_base_cv;
  const gapSig = base && (base.gap_ci_low > 0 || base.gap_ci_high < 0);

  return (
    <figure className="viz flex flex-col gap-4">
      <figcaption>
        <h2 className="text-lg font-semibold">Recall de cáncer por tono de piel, con IC 95%</h2>
        <p className="text-sm text-neutral-600 dark:text-neutral-400">
          Validación cruzada de 5 folds, {cv.n.toLocaleString("es-MX")} imágenes evaluadas una vez cada una. Esta es la evidencia principal.
        </p>
      </figcaption>

      <ul className="flex flex-wrap gap-x-5 gap-y-1 text-sm" aria-label="Leyenda">
        {series.map((s) => (
          <li key={s.key} className="flex items-center gap-2">
            <svg width="14" height="14" aria-hidden="true">
              {s.round ? <circle cx="7" cy="7" r="5" fill={s.color} /> : <rect x="2" y="2" width="10" height="10" rx="1" fill={s.color} transform="rotate(45 7 7)" />}
            </svg>
            {name(s.key)}
          </li>
        ))}
      </ul>

      <div className="flex flex-col gap-5">
        {summary.groups.map((g) => {
          const nm = base?.by_group[g];
          return (
            <div key={g}>
              <p className="text-sm font-medium">
                Tono {g}
                {nm && (
                  <span className="font-normal text-neutral-600 dark:text-neutral-400">
                    {" "}· {nm.n_malignant} cánceres reales (de {nm.n.toLocaleString("es-MX")} fotos)
                  </span>
                )}
              </p>
              {series.map((s) => {
                const d = cv.runs[s.key].by_group[g];
                if (!d) return null;
                return (
                  <div key={s.key} className="mt-2">
                    <div
                      className="relative h-6"
                      role="img"
                      aria-label={`${name(s.key)}, tono ${g}: recall ${p1(d.recall)}, IC 95% de ${p1(d.ci_low)} a ${p1(d.ci_high)}`}
                    >
                      {TICKS.map((t) => (
                        <span key={t} className="absolute inset-y-0 w-px" style={{ left: pos(t), background: "var(--grid)" }} />
                      ))}
                      <span
                        className="absolute top-1/2 h-0.5 -translate-y-1/2"
                        style={{ left: pos(d.ci_low), width: `calc(${pos(d.ci_high)} - ${pos(d.ci_low)})`, background: s.color }}
                      />
                      {[d.ci_low, d.ci_high].map((v, i) => (
                        <span key={i} className="absolute top-1 h-4 w-0.5" style={{ left: pos(v), background: s.color }} />
                      ))}
                      <span
                        className={`absolute top-1/2 size-3.5 -translate-x-1/2 -translate-y-1/2 ${s.round ? "rounded-full" : "rotate-45 rounded-[2px]"}`}
                        style={{ left: pos(d.recall), background: s.color, boxShadow: "0 0 0 2px var(--background)" }}
                      />
                    </div>
                    <p className="text-xs tabular-nums text-neutral-600 dark:text-neutral-400">
                      {p1(d.recall)} <span className="text-neutral-500">[{(d.ci_low * 100).toFixed(1)}–{(d.ci_high * 100).toFixed(1)}]</span>
                    </p>
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>

      <div className="relative h-5 text-xs text-neutral-500" aria-hidden="true">
        {TICKS.map((t) => (
          <span key={t} className="absolute -translate-x-1/2 tabular-nums" style={{ left: pos(t) }}>
            {Math.round(t * 100)}%
          </span>
        ))}
      </div>
      <p className="text-xs text-neutral-500">El eje empieza en 40% (marcas de intervalo, no barras). Más a la derecha = detecta más cánceres.</p>

      {base && (
        <p className="rounded-lg border p-3 text-sm">
          <strong>Brecha {name("convnext_base_cv")}:</strong> {(base.gap * 100).toFixed(1)} puntos entre el mejor y el peor grupo (IC 95% {(base.gap_ci_low * 100).toFixed(1)} a{" "}
          {(base.gap_ci_high * 100).toFixed(1)}).{" "}
          {gapSig
            ? "El intervalo no incluye 0: la brecha es estadísticamente distinguible de cero. Aun así, los intervalos de los grupos se traslapan, y V–VI tiene pocos casos, así que la magnitud es incierta."
            : "El intervalo incluye 0: no hay evidencia clara de brecha."}
        </p>
      )}

      <details className="text-sm">
        <summary className="cursor-pointer">Ver como tabla</summary>
        <div className="overflow-x-auto">
          <table className="mt-2 w-full text-left tabular-nums">
            <caption className="sr-only">Recall de cáncer por grupo de tono con IC 95%</caption>
            <thead>
              <tr className="border-b">
                <th className="py-1 pr-3">Modelo</th>
                <th className="py-1 pr-3">Tono</th>
                <th className="py-1 pr-3">Cánceres</th>
                <th className="py-1 pr-3">Recall</th>
                <th className="py-1">IC 95%</th>
              </tr>
            </thead>
            <tbody>
              {series.flatMap((s) =>
                summary.groups.map((g) => {
                  const d = cv.runs[s.key].by_group[g];
                  return (
                    <tr key={`${s.key}${g}`} className="border-b last:border-0">
                      <td className="py-1 pr-3">{name(s.key)}</td>
                      <td className="py-1 pr-3">{g}</td>
                      <td className="py-1 pr-3">{d.n_malignant}</td>
                      <td className="py-1 pr-3">{p1(d.recall)}</td>
                      <td className="py-1">{p1(d.ci_low)} – {p1(d.ci_high)}</td>
                    </tr>
                  );
                }),
              )}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}

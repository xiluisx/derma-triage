import { CLASSES, CLASS_LABELS_ES } from "@/lib/classes";

export function ResultCard({ probs, threshold, ms }: { probs: number[]; threshold: number; ms: number }) {
  const top = probs.indexOf(Math.max(...probs));
  const uncertain = probs[top] < threshold;
  return (
    <section className="flex flex-col gap-3 w-full">
      {uncertain ? (
        <p className="rounded-md bg-amber-100 text-amber-900 p-3 text-sm dark:bg-amber-900/30 dark:text-amber-200">
          <strong>El modelo no está seguro.</strong> Ninguna clase supera el {Math.round(threshold * 100)}% de
          probabilidad calibrada. Puede ser una foto poco clara o una condición que el modelo no conoce.
        </p>
      ) : (
        <p className="text-lg">
          Clase más probable: <strong>{CLASS_LABELS_ES[CLASSES[top]]}</strong>
        </p>
      )}
      <ul className="flex flex-col gap-2">
        {CLASSES.map((c, i) => (
          <li key={c} className="flex items-center gap-3">
            <span className="w-32 text-sm">{CLASS_LABELS_ES[c]}</span>
            <div className="flex-1 h-3 rounded bg-neutral-200 dark:bg-neutral-800" aria-hidden>
              <div className="h-3 rounded bg-neutral-800 dark:bg-neutral-200" style={{ width: `${probs[i] * 100}%` }} />
            </div>
            <span className="w-12 text-right tabular-nums text-sm">{(probs[i] * 100).toFixed(0)}%</span>
          </li>
        ))}
      </ul>
      <p className="text-xs text-neutral-500">Inferencia local en tu navegador · {ms.toFixed(0)} ms</p>
    </section>
  );
}

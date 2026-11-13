"use client";
import { useState } from "react";
import { CLASSES } from "@/lib/classes";

interface Expl {
  resumen: string;
  que_significa: string;
  siguientes_pasos: string[];
  advertencia: string;
}

export function Explanation({ probs, uncertain }: { probs: number[]; uncertain: boolean }) {
  const [expl, setExpl] = useState<Expl | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function explain() {
    setLoading(true);
    setError(null);
    try {
      const predicted = CLASSES[probs.indexOf(Math.max(...probs))];
      const res = await fetch("/api/explain", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ predicted, probs, uncertain }),
      });
      const data: Expl & { error?: string } = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Error");
      setExpl(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  if (!expl)
    return (
      <div className="flex flex-col gap-2">
        <button onClick={explain} disabled={loading} className="rounded-md border px-4 py-2 self-start">
          {loading ? "Generando explicación…" : "Explicar resultado"}
        </button>
        <p className="text-xs text-neutral-500">Solo se envían las probabilidades, nunca la foto.</p>
        {error && <p role="alert" className="text-sm text-red-600">{error}</p>}
      </div>
    );

  return (
    <article className="flex flex-col gap-2 text-sm">
      <p>{expl.resumen}</p>
      <p>{expl.que_significa}</p>
      <ul className="list-disc pl-5">{expl.siguientes_pasos.map((s) => <li key={s}>{s}</li>)}</ul>
      <p className="text-xs text-neutral-500">{expl.advertencia}</p>
    </article>
  );
}

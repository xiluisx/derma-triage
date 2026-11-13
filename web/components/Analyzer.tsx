"use client";
import { useEffect, useState } from "react";
import { CamOverlay } from "./CamOverlay";
import { Explanation } from "./Explanation";
import { ResultCard } from "./ResultCard";
import { useClassifier } from "@/lib/inference/useClassifier";
import type { Prediction } from "@/lib/inference/types";

export function Analyzer() {
  const clf = useClassifier();
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [pred, setPred] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => () => { if (imageUrl) URL.revokeObjectURL(imageUrl); }, [imageUrl]);

  async function onFile(file: File | undefined) {
    if (!file) return;
    setError(null);
    setPred(null);
    setImageUrl(URL.createObjectURL(file));
    setBusy(true);
    try {
      setPred(await clf.classify(file));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  if (clf.status === "error") return <p role="alert">No se pudo cargar el modelo: {clf.error}</p>;

  const top = pred ? pred.probs.indexOf(Math.max(...pred.probs)) : 0;
  const threshold = clf.meta?.uncertainty_threshold ?? 0.6;

  return (
    <div className="flex flex-col gap-6">
      {clf.status === "loading" && (
        <p className="text-sm">Descargando el modelo (una sola vez)… {Math.round(clf.progress * 100)}%</p>
      )}
      <div className="flex flex-wrap gap-3">
        <label className="cursor-pointer rounded-md bg-neutral-900 text-white px-4 py-2 dark:bg-white dark:text-black">
          Subir foto
          <input type="file" accept="image/*" className="sr-only" disabled={clf.status !== "ready" || busy}
                 onChange={(e) => onFile(e.target.files?.[0])} />
        </label>
        <label className="cursor-pointer rounded-md border px-4 py-2">
          Tomar foto
          <input type="file" accept="image/*" capture="environment" className="sr-only"
                 disabled={clf.status !== "ready" || busy} onChange={(e) => onFile(e.target.files?.[0])} />
        </label>
      </div>
      {busy && <p className="text-sm">Analizando en tu dispositivo…</p>}
      {error && <p role="alert" className="text-red-600">{error}</p>}
      {pred && imageUrl && (
        <div className="grid gap-6 md:grid-cols-2">
          <CamOverlay imageUrl={imageUrl} prediction={pred} cls={top} />
          <div className="flex flex-col gap-6">
            <ResultCard probs={pred.probs} threshold={threshold} ms={pred.ms} />
            <Explanation probs={pred.probs} uncertain={pred.probs[top] < threshold} />
          </div>
        </div>
      )}
      {clf.backend && <p className="text-xs text-neutral-500">Motor: {clf.backend}</p>}
    </div>
  );
}

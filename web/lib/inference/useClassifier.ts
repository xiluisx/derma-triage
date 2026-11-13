"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import type { ModelMeta, Prediction, Variant, WorkerOut } from "./types";

type Status = "loading" | "ready" | "error";
interface State {
  status: Status;
  progress: number; // 0..1
  backend?: string;
  error?: string;
  meta?: ModelMeta;
}

export function useClassifier() {
  const workerRef = useRef<Worker | null>(null);
  const pending = useRef(new Map<number, { resolve: (p: Prediction) => void; reject: (e: Error) => void }>());
  const nextId = useRef(0);
  const [state, setState] = useState<State>({ status: "loading", progress: 0 });

  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_MODEL_BASE_URL!;
    const variant = (process.env.NEXT_PUBLIC_MODEL_VARIANT ?? "fp16") as Variant;
    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;
    let cancelled = false;

    worker.onmessage = (e: MessageEvent<WorkerOut>) => {
      const m = e.data;
      if (m.type === "progress") setState((s) => ({ ...s, progress: m.total ? m.loaded / m.total : 0 }));
      else if (m.type === "ready") {
        console.info(`[modelo] listo, motor: ${m.backend}`);
        setState((s) => ({ ...s, status: "ready", progress: 1, backend: m.backend }));
      } else if (m.type === "result") {
        const { type: _t, id, ...pred } = m;
        pending.current.get(id)?.resolve(pred);
        pending.current.delete(id);
      } else if (m.type === "error") {
        if (m.id !== undefined) {
          pending.current.get(m.id)?.reject(new Error(m.message));
          pending.current.delete(m.id);
        } else setState((s) => ({ ...s, status: "error", error: m.message }));
      }
    };
    worker.onerror = (e) => setState((s) => ({ ...s, status: "error", error: e.message || "Error en el worker" }));

    (async () => {
      try {
        const res = await fetch(`${base}/model-meta.json`);
        if (!res.ok) throw new Error(`No se pudo leer model-meta.json (HTTP ${res.status})`);
        const meta: ModelMeta = await res.json();
        const file = meta.variants[variant];
        if (!file) throw new Error(`La variante ${variant} no existe en model-meta.json`);
        if (cancelled) return;
        setState((s) => ({ ...s, meta }));
        const fb = variant !== "fp32" ? meta.variants.fp32 : undefined;
        worker.postMessage({
          type: "init",
          url: `${base}/${file}`,
          variant,
          fallbackUrl: fb ? `${base}/${fb}` : undefined,
        });
      } catch (err) {
        if (!cancelled) setState({ status: "error", progress: 0, error: String(err) });
      }
    })();

    return () => {
      cancelled = true;
      worker.terminate();
    };
  }, []);

  const classify = useCallback(async (file: File): Promise<Prediction> => {
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image", colorSpaceConversion: "none" });
    const id = nextId.current++;
    return new Promise((resolve, reject) => {
      pending.current.set(id, { resolve, reject });
      workerRef.current!.postMessage({ type: "run", id, bitmap }, [bitmap]);
    });
  }, []);

  return { ...state, classify };
}

/// <reference lib="webworker" />
import * as ort from "onnxruntime-web/webgpu";
import { IMAGE_SIZE, preprocess } from "../preprocess";
import type { WorkerIn, WorkerOut } from "./types";

declare const self: DedicatedWorkerGlobalScope;

let session: ort.InferenceSession | null = null;
const post = (m: WorkerOut, transfer: Transferable[] = []) => self.postMessage(m, transfer);

const MODEL_CACHE = "derma-models";

// Cache Storage: el modelo se guarda explícitamente y la siguiente visita no toca la red.
// La caché HTTP no basta (Chrome puede desalojar entradas de ~56 MB y DevTools la desactiva).
// La URL incluye la versión (v1/), así que una entrada guardada nunca queda obsoleta.
async function download(url: string): Promise<Uint8Array> {
  const cache = "caches" in self ? await caches.open(MODEL_CACHE).catch(() => null) : null;
  const hit = await cache?.match(url);
  if (hit) {
    const bytes = new Uint8Array(await hit.arrayBuffer());
    post({ type: "progress", loaded: bytes.length, total: bytes.length, cached: true });
    return bytes;
  }
  const bytes = await fetchWithProgress(url);
  if (cache) {
    // Borra modelos de versiones anteriores (otra carpeta vN/) y guarda el actual; si falla la cuota, sigue sin caché.
    const versionDir = url.slice(0, url.lastIndexOf("/") + 1);
    for (const req of await cache.keys()) if (!req.url.startsWith(versionDir)) await cache.delete(req);
    await cache.put(url, new Response(bytes.slice().buffer, { headers: { "content-type": "application/octet-stream" } })).catch(
      (e) => console.warn("[modelo] no se pudo guardar en caché", e),
    );
  }
  return bytes;
}

async function fetchWithProgress(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`No se pudo descargar el modelo (HTTP ${res.status})`);
  const total = Number(res.headers.get("content-length") ?? 0);
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    post({ type: "progress", loaded, total });
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    bytes.set(c, offset);
    offset += c.length;
  }
  return bytes;
}

const create = (bytes: Uint8Array, ep: "webgpu" | "wasm") =>
  ort.InferenceSession.create(bytes, { executionProviders: [ep], graphOptimizationLevel: "all" });

async function init(url: string, fallbackUrl?: string) {
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const wasmLabel = `WASM × ${ort.env.wasm.numThreads} hilos`;

  const bytes = await download(url);
  const errors: string[] = [];

  // 1) WebGPU (si el navegador lo expone), 2) WASM, con el modelo principal.
  if ("gpu" in navigator) {
    try {
      session = await create(bytes, "webgpu");
      post({ type: "ready", backend: "WebGPU" });
      return;
    } catch (e) {
      errors.push(`webgpu: ${e}`);
    }
  }
  try {
    session = await create(bytes, "wasm");
    post({ type: "ready", backend: wasmLabel });
    return;
  } catch (e) {
    errors.push(`wasm: ${e}`);
  }

  // 3) Respaldo: fp32 en WASM (p. ej. faltan kernels fp16).
  if (fallbackUrl && fallbackUrl !== url) {
    console.warn("[modelo] fallo con la variante principal; descargando fp32", errors);
    const fb = await download(fallbackUrl);
    session = await create(fb, "wasm");
    post({ type: "ready", backend: `${wasmLabel} (respaldo fp32)` });
    return;
  }
  throw new Error(errors.join(" | "));
}

async function run(id: number, bitmap: ImageBitmap) {
  if (!session) throw new Error("El modelo aún no está listo");
  const t0 = performance.now();
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(bitmap, 0, 0);
  const { data, width, height } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close();

  const { input, crop } = preprocess(data, width, height);
  const out = await session.run({ input: new ort.Tensor("float32", input, [1, 3, IMAGE_SIZE, IMAGE_SIZE]) });
  const probs = Array.from(out.probs.data as ArrayLike<number>);
  const cam = Float32Array.from(out.cam.data as ArrayLike<number>);
  post({ type: "result", id, probs, cam, crop, ms: performance.now() - t0 }, [cam.buffer]);
}

self.onmessage = async (e: MessageEvent<WorkerIn>) => {
  const msg = e.data;
  try {
    if (msg.type === "init") await init(msg.url, msg.fallbackUrl);
    else await run(msg.id, msg.bitmap);
  } catch (err) {
    post({ type: "error", id: msg.type === "run" ? msg.id : undefined, message: String(err) });
  }
};

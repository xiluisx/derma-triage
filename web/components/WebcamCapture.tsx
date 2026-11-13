"use client";
import { useEffect, useRef, useState } from "react";

interface Props {
  open: boolean;
  onClose: () => void;
  onCapture: (file: File) => void;
}

/**
 * Captura con la webcam (escritorio). En móvil se usa el <input capture> nativo, que da mejor calidad.
 * Los cuadros nunca salen del navegador: se dibujan en un canvas y se pasan al mismo flujo que "Subir foto".
 */
export function WebcamCapture({ open, onClose, onCapture }: Props) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string | undefined>();
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  // Abrir/cerrar el <dialog> nativo (da foco atrapado y Escape gratis).
  useEffect(() => {
    const d = dialogRef.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  // Encender la cámara mientras el modal está abierto; apagarla siempre al cerrar.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setError(null);
    setReady(false);

    (async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setError("Este navegador no permite usar la cámara. Usa «Subir foto».");
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: deviceId
            ? { deviceId: { exact: deviceId }, width: { ideal: 1920 }, height: { ideal: 1080 } }
            : { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1080 } },
          audio: false,
        });
        if (cancelled) return stream.getTracks().forEach((t) => t.stop());
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        setReady(true);
        // Las etiquetas de los dispositivos solo están disponibles después de conceder el permiso.
        const all = await navigator.mediaDevices.enumerateDevices();
        if (!cancelled) setDevices(all.filter((d) => d.kind === "videoinput"));
      } catch (e) {
        const name = (e as DOMException)?.name;
        setError(
          name === "NotAllowedError"
            ? "No diste permiso para usar la cámara. Puedes habilitarlo en la barra de direcciones o usar «Subir foto»."
            : name === "NotFoundError" || name === "OverconstrainedError"
              ? "No se encontró una cámara disponible."
              : name === "NotReadableError"
                ? "La cámara está siendo usada por otra aplicación."
                : `No se pudo abrir la cámara (${name ?? e}).`,
        );
      }
    })();

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, [open, deviceId]);

  function capture() {
    const v = videoRef.current;
    if (!v || !v.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = v.videoWidth;
    canvas.height = v.videoHeight;
    canvas.getContext("2d")!.drawImage(v, 0, 0);
    // PNG sin pérdida: el modelo recibe exactamente lo que se vio en pantalla.
    canvas.toBlob((blob) => {
      if (!blob) return;
      onCapture(new File([blob], `webcam-${Date.now()}.png`, { type: "image/png" }));
      onClose();
    }, "image/png");
  }

  return (
    <dialog
      ref={dialogRef}
      onClose={onClose}
      onClick={(e) => e.target === dialogRef.current && onClose()}
      className="m-auto w-[min(100vw-2rem,48rem)] rounded-lg bg-white p-0 text-neutral-900 backdrop:bg-black/60 dark:bg-neutral-900 dark:text-neutral-100"
      aria-labelledby="webcam-title"
    >
      <div className="flex flex-col gap-4 p-4">
        <div className="flex items-center justify-between gap-4">
          <h2 id="webcam-title" className="font-semibold">Tomar foto con la cámara</h2>
          <button type="button" onClick={onClose} className="rounded-md px-2 py-1 text-sm hover:bg-neutral-100 dark:hover:bg-neutral-800" aria-label="Cerrar">
            ✕
          </button>
        </div>

        {error ? (
          <p role="alert" className="text-sm text-red-600 dark:text-red-400">{error}</p>
        ) : (
          <div className="relative overflow-hidden rounded-md bg-black">
            <video ref={videoRef} playsInline muted className="block h-auto w-full" />
            {/* Recuadro guía: el modelo recorta el cuadrado central del lado corto. */}
            {ready && (
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                <div className="aspect-square h-full rounded-sm border-2 border-white/90 shadow-[0_0_0_9999px_rgba(0,0,0,0.35)]" />
              </div>
            )}
            {!ready && <p className="absolute inset-0 flex items-center justify-center text-sm text-white/80">Encendiendo la cámara…</p>}
          </div>
        )}

        <p className="text-xs text-neutral-500">
          Centra la lesión dentro del recuadro, con buena luz y sin flash. La imagen se analiza en tu dispositivo y no se envía a ningún servidor.
        </p>

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={capture}
            disabled={!ready}
            className="rounded-md bg-neutral-900 px-4 py-2 text-white disabled:opacity-40 dark:bg-white dark:text-black"
          >
            Capturar
          </button>
          {devices.length > 1 && (
            <label className="flex items-center gap-2 text-sm">
              Cámara
              <select
                value={deviceId ?? ""}
                onChange={(e) => setDeviceId(e.target.value || undefined)}
                className="rounded-md border bg-transparent px-2 py-1"
              >
                <option value="">Predeterminada</option>
                {devices.map((d, i) => (
                  <option key={d.deviceId} value={d.deviceId}>
                    {d.label || `Cámara ${i + 1}`}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
      </div>
    </dialog>
  );
}

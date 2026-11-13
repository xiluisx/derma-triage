"use client";
import { useEffect, useRef, useState } from "react";
import { camHeatmap } from "@/lib/cam";
import type { Prediction } from "@/lib/inference/types";

const SIZE = 336;

export function CamOverlay({ imageUrl, prediction, cls }: { imageUrl: string; prediction: Prediction; cls: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [showCam, setShowCam] = useState(true);

  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      const ctx = ref.current!.getContext("2d")!;
      const { sx, sy, sw, sh } = prediction.crop;
      ctx.clearRect(0, 0, SIZE, SIZE);
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, SIZE, SIZE);
      if (!showCam) return;
      const heat = new OffscreenCanvas(SIZE, SIZE);
      heat.getContext("2d")!.putImageData(new ImageData(camHeatmap(prediction.cam, cls, SIZE), SIZE, SIZE), 0, 0);
      ctx.drawImage(heat, 0, 0);
    };
    img.src = imageUrl;
  }, [imageUrl, prediction, cls, showCam]);

  return (
    <figure className="flex flex-col items-center gap-2">
      <canvas ref={ref} width={SIZE} height={SIZE} className="rounded-lg w-full max-w-[336px] aspect-square" />
      <label className="text-sm flex items-center gap-2">
        <input type="checkbox" checked={showCam} onChange={(e) => setShowCam(e.target.checked)} />
        Mostrar mapa de calor
      </label>
      <figcaption className="text-xs text-neutral-500 text-center max-w-[336px]">
        Zonas cálidas = regiones que más empujaron al modelo hacia la clase mostrada. No indican dónde está la
        enfermedad.
      </figcaption>
    </figure>
  );
}

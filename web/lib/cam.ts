// Turbo aproximado con 5 paradas.
const STOPS: [number, [number, number, number]][] = [
  [0.0, [48, 18, 59]],
  [0.25, [70, 134, 251]],
  [0.5, [27, 229, 181]],
  [0.75, [250, 186, 57]],
  [1.0, [122, 4, 3]],
];

function colormap(v: number): [number, number, number] {
  for (let i = 1; i < STOPS.length; i++) {
    const [t1, c1] = STOPS[i];
    const [t0, c0] = STOPS[i - 1];
    if (v <= t1) {
      const f = (v - t0) / (t1 - t0);
      return [0, 1, 2].map((k) => Math.round(c0[k] + f * (c1[k] - c0[k]))) as [number, number, number];
    }
  }
  return STOPS[STOPS.length - 1][1];
}

/** CAM [K, n, n] de la clase cls -> RGBA out×out (ReLU, normalizado a su máximo, bilineal, alfa según intensidad). */
export function camHeatmap(cam: Float32Array, cls: number, out: number, n = 7): Uint8ClampedArray<ArrayBuffer> {
  const slice = cam.subarray(cls * n * n, (cls + 1) * n * n);
  let max = 0;
  for (const v of slice) max = Math.max(max, v);
  const norm = (i: number) => (max > 0 ? Math.max(slice[i], 0) / max : 0);

  const rgba = new Uint8ClampedArray(out * out * 4);
  const scale = n / out;
  for (let y = 0; y < out; y++) {
    const sy = Math.min(Math.max((y + 0.5) * scale - 0.5, 0), n - 1);
    const y0 = Math.floor(sy), y1 = Math.min(y0 + 1, n - 1), fy = sy - y0;
    for (let x = 0; x < out; x++) {
      const sx = Math.min(Math.max((x + 0.5) * scale - 0.5, 0), n - 1);
      const x0 = Math.floor(sx), x1 = Math.min(x0 + 1, n - 1), fx = sx - x0;
      const v =
        norm(y0 * n + x0) * (1 - fx) * (1 - fy) + norm(y0 * n + x1) * fx * (1 - fy) +
        norm(y1 * n + x0) * (1 - fx) * fy + norm(y1 * n + x1) * fx * fy;
      const [r, g, b] = colormap(v);
      const p = (y * out + x) * 4;
      rgba[p] = r;
      rgba[p + 1] = g;
      rgba[p + 2] = b;
      rgba[p + 3] = Math.round(200 * v);
    }
  }
  return rgba;
}

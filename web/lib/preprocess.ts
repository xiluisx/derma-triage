// Réplica de ml/preprocess.py. No cambiar uno sin el otro: tests/parity.test.ts lo vigila.
export const IMAGE_SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

export interface CropRect {
  sx: number; // en coordenadas de la imagen original
  sy: number;
  sw: number;
  sh: number;
}

export function resizedHW(h: number, w: number, target = IMAGE_SIZE): [number, number] {
  return h <= w ? [target, Math.floor((target * w) / h)] : [Math.floor((target * h) / w), target];
}

interface AxisWeights {
  start: Int32Array;
  size: Int32Array;
  weights: Float64Array[];
}

/** Pesos antialias de PyTorch para las salidas [from, from + count) de un eje inSize -> outSize. */
function axisWeights(inSize: number, outSize: number, from: number, count: number): AxisWeights {
  const scale = inSize / outSize;
  const support = scale >= 1 ? scale : 1;
  const invscale = scale >= 1 ? 1 / scale : 1;
  const start = new Int32Array(count);
  const size = new Int32Array(count);
  const weights: Float64Array[] = [];
  for (let o = 0; o < count; o++) {
    const i = from + o;
    const center = scale * (i + 0.5);
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize);
    const n = xmax - xmin;
    const w = new Float64Array(n);
    let total = 0;
    for (let j = 0; j < n; j++) {
      const x = Math.abs((j + xmin - center + 0.5) * invscale);
      w[j] = x < 1 ? 1 - x : 0;
      total += w[j];
    }
    if (total > 0) for (let j = 0; j < n; j++) w[j] /= total;
    start[o] = xmin;
    size[o] = n;
    weights.push(w);
  }
  return { start, size, weights };
}

/**
 * RGBA (getImageData / pngjs) -> tensor float32 [3, 224, 224] normalizado, más el rectángulo
 * de recorte en coordenadas originales (para dibujar el CAM sobre la foto).
 */
export function preprocess(rgba: Uint8ClampedArray | Uint8Array, width: number, height: number) {
  const S = IMAGE_SIZE;
  const [nh, nw] = resizedHW(height, width);
  const top = Math.floor((nh - S) / 2);
  const left = Math.floor((nw - S) / 2);

  const cols = axisWeights(width, nw, left, S);
  const rows = axisWeights(height, nh, top, S);

  // Pase horizontal: todas las filas de entrada, solo las 224 columnas del recorte.
  const tmp = new Float64Array(3 * height * S);
  for (let y = 0; y < height; y++) {
    const rowBase = y * width * 4;
    for (let o = 0; o < S; o++) {
      const w = cols.weights[o];
      const x0 = cols.start[o];
      let r = 0, g = 0, b = 0;
      for (let j = 0; j < w.length; j++) {
        const p = rowBase + (x0 + j) * 4;
        r += w[j] * (rgba[p] / 255);
        g += w[j] * (rgba[p + 1] / 255);
        b += w[j] * (rgba[p + 2] / 255);
      }
      tmp[(0 * height + y) * S + o] = r;
      tmp[(1 * height + y) * S + o] = g;
      tmp[(2 * height + y) * S + o] = b;
    }
  }

  // Pase vertical: solo las 224 filas del recorte + normalización.
  const input = new Float32Array(3 * S * S);
  for (let c = 0; c < 3; c++) {
    for (let o = 0; o < S; o++) {
      const w = rows.weights[o];
      const y0 = rows.start[o];
      for (let x = 0; x < S; x++) {
        let v = 0;
        for (let j = 0; j < w.length; j++) v += w[j] * tmp[(c * height + y0 + j) * S + x];
        input[(c * S + o) * S + x] = (v - MEAN[c]) / STD[c];
      }
    }
  }

  const crop: CropRect = {
    sx: (left * width) / nw,
    sy: (top * height) / nh,
    sw: (S * width) / nw,
    sh: (S * height) / nh,
  };
  return { input, crop };
}

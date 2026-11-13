import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import * as ort from "onnxruntime-node";
import { PNG } from "pngjs";
import { describe, expect, it } from "vitest";
import { preprocess } from "../lib/preprocess";

// parity-local (imágenes reales) está en .gitignore: en CI solo corren los sintéticos.
const DIRS = ["tests/fixtures/parity", "tests/fixtures/parity-local"];
const MODEL = process.env.PARITY_MODEL ?? "../artifacts/model-fp32.onnx";

interface Fx {
  dir: string;
  file: string;
  probs: number[];
  cam: number[];
}
const expected: Fx[] = DIRS.flatMap((dir) => {
  const p = join(dir, "expected.json");
  if (!existsSync(p)) return [];
  return (JSON.parse(readFileSync(p, "utf8")) as Omit<Fx, "dir">[]).map((e) => ({ ...e, dir }));
});

const maxAbsDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};
const argmax = (a: ArrayLike<number>) => Array.from(a).indexOf(Math.max(...Array.from(a)));

describe.skipIf(expected.length === 0)("paridad de preprocesamiento JS ↔ Python", () => {
  for (const fx of expected) {
    it(`${fx.dir}/${fx.file}: tensor de entrada idéntico (< 1e-4)`, () => {
      const png = PNG.sync.read(readFileSync(join(fx.dir, fx.file)));
      const { input } = preprocess(png.data, png.width, png.height);
      const buf = readFileSync(join(fx.dir, fx.file.replace(".png", ".input.bin")));
      const ref = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
      expect(input.length).toBe(ref.length);
      const d = maxAbsDiff(input, ref);
      console.log(`input ${fx.dir}/${fx.file} maxAbsDiff=${d}`);
      expect(d).toBeLessThan(1e-4);
    });
  }

  it.skipIf(!existsSync(MODEL))("probs y cam del ONNX coinciden (< 1e-3) y misma clase top-1", async () => {
    const session = await ort.InferenceSession.create(MODEL);
    for (const fx of expected) {
      const png = PNG.sync.read(readFileSync(join(fx.dir, fx.file)));
      const { input } = preprocess(png.data, png.width, png.height);
      const out = await session.run({ input: new ort.Tensor("float32", input, [1, 3, 224, 224]) });
      const probs = out.probs.data as Float32Array;
      const cam = out.cam.data as Float32Array;
      const dp = maxAbsDiff(probs, fx.probs);
      const dc = maxAbsDiff(cam, fx.cam);
      console.log(`onnx ${fx.dir}/${fx.file} probs=${dp} cam=${dc}`);
      expect(dp).toBeLessThan(1e-3);
      expect(dc).toBeLessThan(1e-3);
      expect(argmax(probs)).toBe(argmax(fx.probs));
    }
  });
});

import Anthropic from "@anthropic-ai/sdk";
import { getCloudflareContext } from "@opennextjs/cloudflare";
import { z } from "zod";
import { CLASSES, CLASS_LABELS_ES } from "@/lib/classes";

const Body = z
  .object({
    predicted: z.enum(CLASSES),
    probs: z.array(z.number().min(0).max(1)).length(3),
    uncertain: z.boolean(),
  })
  .strict();

const Explanation = z.object({
  resumen: z.string(),
  que_significa: z.string(),
  siguientes_pasos: z.array(z.string()),
  advertencia: z.string(),
});

const OUTPUT_SCHEMA = {
  type: "object",
  properties: {
    resumen: { type: "string" },
    que_significa: { type: "string" },
    siguientes_pasos: { type: "array", items: { type: "string" } },
    advertencia: { type: "string" },
  },
  required: ["resumen", "que_significa", "siguientes_pasos", "advertencia"],
  additionalProperties: false,
};

const SYSTEM = `Eres el componente explicativo de un proyecto académico de IA que clasifica fotos de lesiones de piel en tres categorías: benigno, maligno y no neoplásico. Recibes únicamente la salida numérica del modelo (nunca la imagen).

Escribe en español claro para una persona sin formación médica:
- resumen: una o dos frases sobre lo que indicó el modelo, mencionando la probabilidad de la clase principal.
- que_significa: qué significa esa categoría en términos generales (sin nombrar enfermedades específicas como diagnóstico).
- siguientes_pasos: 2 a 4 pasos prácticos. Siempre incluye consultar a un dermatólogo; si la clase principal es "maligno" o el modelo está incierto, recomienda hacerlo pronto.
- advertencia: recuerda que es un proyecto académico, que no es un diagnóstico y que el modelo puede equivocarse, especialmente en tonos de piel con menos ejemplos de entrenamiento.

Si el campo "incierto" es verdadero, dilo explícitamente y no presentes la clase principal como conclusión. No inventes datos que no estén en la entrada.`;

export async function POST(req: Request) {
  const { env } = getCloudflareContext();

  const ip = req.headers.get("cf-connecting-ip") ?? "local";
  const { success } = await env.EXPLAIN_RATE_LIMITER.limit({ key: ip });
  if (!success) return Response.json({ error: "Demasiadas solicitudes. Intenta en un minuto." }, { status: 429 });

  const parsed = Body.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return Response.json({ error: "Solicitud inválida" }, { status: 400 });
  const { predicted, probs, uncertain } = parsed.data;

  const input = {
    clase_principal: CLASS_LABELS_ES[predicted],
    probabilidades: Object.fromEntries(CLASSES.map((c, i) => [CLASS_LABELS_ES[c], `${Math.round(probs[i] * 100)}%`])),
    incierto: uncertain,
  };

  const client = new Anthropic({ apiKey: env.ANTHROPIC_API_KEY });
  try {
    const response = await client.messages.create({
      model: "claude-haiku-5-5",
      max_tokens: 4096,
      output_config: { effort: "low", format: { type: "json_schema", schema: OUTPUT_SCHEMA } },
      system: SYSTEM,
      messages: [{ role: "user", content: JSON.stringify(input) }],
    });

    if (response.stop_reason === "refusal") {
      return Response.json({ error: "No fue posible generar la explicación. Consulta a un dermatólogo." }, { status: 502 });
    }
    const block = response.content.find((b) => b.type === "text");
    const text = block && block.type === "text" ? block.text : "";
    let json: unknown;
    try {
      json = JSON.parse(text);
    } catch {
      return Response.json({ error: "Respuesta inesperada del modelo" }, { status: 502 });
    }
    const out = Explanation.safeParse(json);
    if (!out.success) return Response.json({ error: "Respuesta inesperada del modelo" }, { status: 502 });
    return Response.json(out.data);
  } catch (err) {
    if (err instanceof Anthropic.RateLimitError) {
      return Response.json({ error: "El servicio está ocupado. Intenta en un momento." }, { status: 503 });
    }
    if (err instanceof Anthropic.APIError) {
      return Response.json({ error: "Error del servicio de explicación" }, { status: 502 });
    }
    throw err;
  }
}

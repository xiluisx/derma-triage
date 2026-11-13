import { Analyzer } from "@/components/Analyzer";

export default function Page() {
  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-2xl font-semibold">Analiza una foto de una lesión de piel</h1>
      <p className="text-sm text-neutral-600 dark:text-neutral-400">
        El modelo corre en tu navegador: la foto no sale de tu dispositivo. Clasifica entre benigno, maligno y no
        neoplásico, y muestra qué tan seguro está.
      </p>
      <p role="note" className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-200">
        <strong>Herramienta académica, no diagnóstica.</strong> Un resultado no sustituye la consulta con un
        dermatólogo. Ante cualquier lesión que te preocupe, acude con un especialista.
      </p>
      <Analyzer />
    </div>
  );
}

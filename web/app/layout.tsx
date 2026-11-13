import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Triaje de piel · Auditoría de sesgo",
  description: "Clasificador de lesiones de piel que corre en tu navegador, con auditoría de equidad por tono de piel.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es">
      <head>
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
      </head>
      <body className="min-h-dvh bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100">
        <header className="border-b">
          <nav className="mx-auto max-w-4xl flex gap-6 px-4 py-3 text-sm">
            <Link href="/" className="font-semibold">Analizar</Link>
            <Link href="/auditoria">Auditoría</Link>
          </nav>
        </header>
        <main className="mx-auto max-w-4xl px-4 py-8">{children}</main>
        <footer className="mx-auto max-w-4xl px-4 pb-8 text-xs text-neutral-500">
          Proyecto académico. <strong>No es una herramienta de diagnóstico</strong> y no sustituye la consulta con un
          dermatólogo. Las fotos se procesan en tu dispositivo y nunca se envían a un servidor.
        </footer>
      </body>
    </html>
  );
}

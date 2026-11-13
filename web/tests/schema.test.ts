import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { AuditSummary } from "../lib/audit-schema";

describe("contrato de auditoría", () => {
  it("el ejemplo compartido con Python es válido", () => {
    const json = JSON.parse(readFileSync("../contracts/audit-summary.example.json", "utf8"));
    expect(() => AuditSummary.parse(json)).not.toThrow();
  });
  it("el summary publicado en la app es válido", () => {
    const json = JSON.parse(readFileSync("public/audit/summary.json", "utf8"));
    expect(() => AuditSummary.parse(json)).not.toThrow();
  });
});

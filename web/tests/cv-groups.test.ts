import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { CvGroups } from "../lib/cv-groups-schema";

describe("cv_groups.json", () => {
  it("cumple el esquema y tiene los grupos de summary.json", () => {
    const cv = CvGroups.parse(JSON.parse(readFileSync("public/audit/cv_groups.json", "utf8")));
    const summary = JSON.parse(readFileSync("public/audit/summary.json", "utf8"));
    for (const run of Object.values(cv.runs)) expect(Object.keys(run.by_group).sort()).toEqual([...summary.groups].sort());
  });
});

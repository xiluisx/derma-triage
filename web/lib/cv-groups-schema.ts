import { z } from "zod";

// Recall maligno por grupo con IC 95% (bootstrap) de las corridas agrupadas de 5 folds.
// Extraído de runs/<config>_cv/metrics_test.json (summary.json no incluye estos IC).
const GroupCI = z
  .object({
    n: z.number().int(),
    n_malignant: z.number().int(),
    recall: z.number(),
    ci_low: z.number(),
    ci_high: z.number(),
  })
  .strict();

const CvRun = z
  .object({
    by_group: z.record(z.string(), GroupCI),
    gap: z.number(),
    gap_ci_low: z.number(),
    gap_ci_high: z.number(),
  })
  .strict();

export const CvGroups = z
  .object({ schema_version: z.literal(1), n: z.number().int(), runs: z.record(z.string(), CvRun) })
  .strict();

export type CvGroups = z.infer<typeof CvGroups>;
export type CvRun = z.infer<typeof CvRun>;

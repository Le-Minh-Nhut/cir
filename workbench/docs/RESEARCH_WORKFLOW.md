# Research workflow

1. Pin source and checkpoint provenance in registry.
2. On GPU host, run unmodified official evaluation.
3. Record command, environment, source commit, checkpoint hash, stdout/stderr and local metrics.
4. Add only observation-only instrumentation.
5. Require metric parity within strict tolerance.
6. Save versioned per-query JSON.
7. Rebuild DuckDB derived index.
8. Select one protocol and compatible runs.
9. Inspect consensus failures, disagreement and common distractors.
10. Annotate cases; save reproducible cohort with run IDs.
11. Export cohort JSON, CSV or Markdown before making research claims.

Never mix protocol families. A noise-trained checkpoint evaluated on clean data remains noise-trained. Paper metrics are sanity references, not local reproductions.

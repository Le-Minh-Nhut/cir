# CIR Failure Analysis Workbench

Local FashionIQ CIR failure-analysis operator tooling. It preserves official upstream evaluators and records guarded source/checkpoint/protocol facts; it does not implement a replacement model or claim reproduction.

Operator manual: [workbench/README.md](workbench/README.md). It covers no-install policy, mock/manual artifacts, guarded real-evaluation phases, model blockers, result provenance, UI, cleanup, and exact script flags.

Tracked code lives under `src/` and `workbench/`. External datasets, upstream source clones, checkpoints, generated result JSON, DuckDB indexes, local annotations/cohorts, and frontend builds are ignored. Current model/reproduction evidence: [workbench/docs/UPSTREAM_AUDIT.md](workbench/docs/UPSTREAM_AUDIT.md); protocol boundaries: [workbench/docs/PROTOCOL_AUDIT.md](workbench/docs/PROTOCOL_AUDIT.md).

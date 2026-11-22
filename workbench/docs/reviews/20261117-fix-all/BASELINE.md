# Initial census — 2026-11-17

Captured before worker edits. Initial checkout:

- Root: `/home/heheboiz/data/cir`
- Branch: `research/cir-failure-workbench`
- Revision: `d3f0a923ba6d9f39f7a2f44aaad583c04fa3c93e`
- Initial dirty paths retained: `CLEANUP_REPORT.md.save`; `workbench/docs/reviews/20261112-parity-repair/raw/third_model/population_habit_dress.npz`, `population_habit_shirt.npz`, `population_habit_toptee.npz`.
- Services already owned by other work: backend PID 1809 on `:8000`, frontend PID 2129 on `:5173`, DB on `:5432`. Do not probe/restart/use those instances.
- GPU census: RTX 5070 Ti, 1070 MiB / 16 GiB used, 19% utilization; desktop clients present. No campaign GPU run authorized.
- Disk: 121 GiB free, 80% used. Required interpreter exists: `workbench/envs/ctl312h/bin/python`; `.venv` points to it.
- Initial expected handoff path `/home/heheboiz/data/cir/CIR_WORKBENCH_HANDOFF_2026-11-14.vi.md` was absent (`ls` exit 2). No root agent instructions beyond auto-loaded `AGENTS.md`; no `workbench/AGENTS.md`.
- CPU heavy-job slot later assigned exclusively for focused tests, max ten threads (`pytest -n10`). Per latest scheduling update m00577, slot presently free for gate worker's narrow check; no tests/builds have been started by this workstream.

Exact initial census commands and full outputs are preserved at `/tmp/cir-release-baseline-20261117.md` (local session file). No checkpoint/raw-asset hashes were computed. No services were touched. All population/dataset claims remain pending source audit.

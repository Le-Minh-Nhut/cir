# PC Reproduction Handoff (Release B)

Release B runs on a provisioned GPU PC only. Start with the [Autopilot Operator Guide](AUTOPILOT_GUIDE.md): it is the primary runbook for paths, permissions, environment grouping, checkpoint policy, GPU choice, budgets, ledger key, launch, resume, reports, and blockers.

## One primary execution command

After configuring the PC-specific YAML and explicitly granting the capabilities required by its selected stages, run:

```bash
./workbench/run.sh autopilot --config /path/to/pc_autopilot.yaml --apply --resume
```

Use same config, artifact roots, selected models, GPU visibility, and HMAC key on resume. Do not run the individual stage scripts as an alternative workflow. Full reproduction is not guaranteed by laptop-side engineering; reports distinguish completed, blocked, and failed models.

## PC-only acceptance still owed

- Real checkpoint loading and official evaluator execution.
- Official Recall reproduction against paper-reported values; reported values remain sanity references.
- Verified model-specific environment and GPU capability on the actual host/device selection.
- Per-query instrumentation remains unavailable; aggregate evaluation alone does not produce schema-v2 per-query result JSON.

## Registry blockers

These are registry/evidence blockers, not tasks Autopilot can bypass:

| Model(s) | Current blocker |
|---|---|
| `ptha_mtst` | No verified FashionIQ fine-tuned checkpoint and no source integration. |
| `tgcir` | No source replay CLI; required ViT-B/16 and preparation assets unresolved. |
| `pair` | FashionIQ mapping for `pair-B1.pt` / `pair-B2.pt` unresolved. |
| `clvc_net` | No standalone replay entrypoint; checkpoint mapping and ResNet-50 asset unresolved. |
| `sprc` | Checkpoint/backbone mapping unresolved. |
| `dcnet` | Checkpoint mapping/hash, author-generated PKLs, and ResNet-50 asset remain unresolved. |
| Combiner / CLIP4Cir | Checkpoint mapping/pairing and RN50x4 base asset unresolved. |
| `encoder` | Direct FashionIQ checkpoint URL/hash unresolved; pinned evaluator imports missing `datasets1.py`; OpenCLIP binary required. |
| Air-Know, ConeSep, HABIT, INTENT | Registered checkpoints are noise-trained, not clean-trained; no audited official command constructor. |
| `hint` | Checkpoint evidence exists; official command constructor remains unaudited. |
| `limn` | Replay implemented, but three checkpoint files, resized images, correction dictionaries, and verified CUDA environment must pass preflight. |
| `csmcir` | Pinned command exists; required COT captions lack verified author source. Registered Qwen captions are acquired only when `allow_network` + `allow_large_downloads` are granted. |

Do not describe these entries as runnable until their recorded blockers are resolved with authoritative evidence. A model blocked in reports stays blocked; do not substitute a checkpoint or invent a command.

## Optional diagnostics

These commands inspect state/readiness only; neither replaces the primary execution command.

```bash
# Optional diagnostic: show accepted autopilot options.
./workbench/run.sh autopilot --help

# Optional diagnostic: inspect latest autopilot report as JSON.
./workbench/run.sh autopilot --config /path/to/pc_autopilot.yaml --report --json

# Optional diagnostic: inspect configured model interpreters.
python workbench/scripts/manage_environment.py --list --json
```

Do not mistake interpreter readiness for model readiness. See [AUTOMATION_RUNBOOK.md](AUTOMATION_RUNBOOK.md) for proof/ledger guarantees and trust boundaries.

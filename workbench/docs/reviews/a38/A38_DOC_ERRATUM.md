# A38 documentation erratum

`workbench/docs/reviews/a37/A37_PC_HANDOFF.md:117,133` omitted `--resume` from composition-eval resume instructions. The flag is opt-in. Corrected command:

```bash
./workbench/run.sh composition-eval --config ~/cir_pc.yaml --models encoder --smoke --resume
```

`workbench/docs/PC_PILOT_HANDOFF.md:108-109,259-260` now states that ENCODER and LIMN declare observers and the other 15 of 17 models are aggregate-only. At `workbench/docs/PC_PILOT_HANDOFF.md:118`, `CAPABILITY_ABSENT_FOR_DIAGNOSTIC` is identified as a `per_query_export_reason`, not a `per_query_state` value.

The A37 review archive was NOT rewritten; historical audit evidence remains unchanged.

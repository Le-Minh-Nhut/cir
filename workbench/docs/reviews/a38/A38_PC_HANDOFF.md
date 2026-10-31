# A38 PC handoff — verified sequence

Target GPU: **NVIDIA RTX 5070 Ti sm_120**. This host reports `NVIDIA GeForce RTX 3050 Laptop GPU, 8.6`; target state remains **`TARGET_GPU_PENDING`**.

Commands below were checked against current `--help` and parser source where the CLI accepts flags. Do not run resource-intensive preparation/evaluation concurrently.

1. **Environment setup** — create and install the controller environment:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r workbench/backend/requirements.txt .
   ```
   These are environment setup commands documented in the A37 handoff; no package installation was run during this audit.

2. **Repository/root verification** — from repository root, copy the cohort profile and edit its six absolute paths; then:
   ```bash
   cp workbench/configs/pc_autopilot.pc_cohort.yaml ~/cir_pc.yaml
   $EDITOR ~/cir_pc.yaml
   cp workbench/config/workbench.env.example workbench/config/workbench.env
   $EDITOR workbench/config/workbench.env
   .venv/bin/python workbench/scripts/print_roots.py --config ~/cir_pc.yaml
   ```
   `print_roots.py --help` confirms `--config`. `git rev-parse HEAD` is a read-only repository identity check; do not assume a pinned SHA from this document.

3. **Dataset preparation** — verify the configured FashionIQ dataset and composition release inventory:
   ```bash
   .venv/bin/python workbench/scripts/doctor.py --scope workbench --json
   ./workbench/run.sh composition-eval --config ~/cir_pc.yaml --prepare-data
   ```
   `doctor.py --help` confirms `--scope` and `--json`; `composition_eval.py --help` confirms `--config` and `--prepare-data`. Preparation reports missing files; it does not download them.

4. **Checkpoint preparation** — stage the declared ENCODER checkpoint `fashioniq` at the configured checkpoint root. `manage_environment.py` does not stage checkpoints; the composition planner checks checkpoint presence. Verify staged files with:
   ```bash
   ./workbench/run.sh composition-eval --config ~/cir_pc.yaml --models encoder --plan
   ```
   The CLI help confirms `--config`, `--models`, and `--plan`. **UNVERIFIED —** no current composition CLI command stages/downloads checkpoints; use the registry's recorded source and checksum through the approved transfer procedure, then rerun the plan.

5. **ENCODER interpreter resolution** — inspect environment, then provision only if necessary and authorized:
   ```bash
   .venv/bin/python workbench/scripts/manage_environment.py --model encoder --json
   .venv/bin/python workbench/scripts/manage_environment.py --model encoder --create --allow-env-install --allow-network
   ```
   `--model`, `--json`, `--create`, `--allow-env-install`, and `--allow-network` appear in actual help. The create command may install packages and requires operator authorization. Runtime resolver itself is `resolve_model_interpreter(model_by_id("encoder"), config)`; interpreter path must be reported by that resolver, not guessed from PATH.

6. **Real CUDA allocation check** — after the interpreter is resolved, execute on the PC:
   ```bash
   <ENCODER_INTERPRETER> -c 'import torch; x=torch.empty((512,512),device="cuda"); torch.cuda.synchronize(); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), x.device)'
   ```
   **UNVERIFIED —** this is a direct Python runtime check, not a repository CLI flag; replace placeholder only with interpreter path resolved in step 5. Requires a real CUDA device.

7. **Real CUDA matmul check**:
   ```bash
   <ENCODER_INTERPRETER> -c 'import torch; a=torch.randn((512,512),device="cuda"); b=a@a; torch.cuda.synchronize(); assert b.isfinite().all().item(); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), "CUDA matmul OK")'
   ```
   **UNVERIFIED —** same host prerequisite and placeholder as step 6.

8. **ENCODER smoke inference** — explicit model selection prevents profile cohort expansion:
   ```bash
   ./workbench/run.sh composition-eval --config ~/cir_pc.yaml --models encoder --smoke
   ```
   Help confirms flags. `--smoke` defaults to one query and preserves full gallery.

9. **Resume verification** — run the same smoke command with opt-in resume:
   ```bash
   ./workbench/run.sh composition-eval --config ~/cir_pc.yaml --models encoder --smoke --resume
   ```
   Confirm result JSON reports `status: REUSED` / `reused: true` for a matching completed rank record. Resume only reuses when saved fingerprint matches, conditions match, and source identity is available; otherwise it computes again. Composition runner has no pipeline `[SKIP] resume` log message.

10. **Full evaluation** — request all composition subsets, explicit ENCODER only, with resume enabled:
    ```bash
    ./workbench/run.sh composition-eval --config ~/cir_pc.yaml --models encoder --subsets full,sf,validated --resume
    ```
    Flags verified in actual help. Do not use `--models selected`: checked-in PC profile has `selected_models: all_registered`.

11. **Result validation** — use reported run identity and verify if it represents a supported reproduction run:
    ```bash
    .venv/bin/python workbench/scripts/verify_run.py --run-id <RUN_ID> --model encoder --checkpoint fashioniq --protocol fashioniq_val_split
    ```
    Actual help requires all four identity flags. **UNVERIFIED —** composition results are not necessarily official evaluator reports accepted by `verify_run.py`; only use this validation if corresponding aggregate report, ledger entry, and completion proof exist. Composition output paths are reported by `--plan` and summaries.

## A37 handoff erratum

`workbench/docs/reviews/a37/A37_PC_HANDOFF.md:117,133` gave the ENCODER smoke command without `--resume` while describing a resume check. Composition `--resume` is opt-in (`composition_eval.py:37`, `action="store_true"`), so those invocations cold-run. The corrected resume command is step 9 above. The prior expected pipeline log `[SKIP] resume: ...` was also incorrect for composition; composition emits structured output with `status: REUSED` when it reuses matching rank records.

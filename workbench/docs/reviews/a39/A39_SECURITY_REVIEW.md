# A39 Security Review

## 1. Gateway credential — external host files

**Status: `CONFIRMED_UNFIXED` / `CONFIRMED_EXPOSED`.** Exact gateway token appears in 53 files outside repository: 52 under `~/.omp/agent/**` and one `~/.bash_history`. Broad token-shaped pattern matches: 143 files across local user directories. Gateway token is **NOT_PRESENT** in tracked repository files (0 matches). No token value is reproduced here.

Two of the 53 exact-match files have mode `664`; 51 have mode `600`. Permissions do not revoke a credential: `chmod 600` cannot invalidate an active token. Rotation/revocation remains required at provider gateway; no local rotation procedure was documented in evidence. Historical OMP session transcripts and shell history retain exposure; deleting local files also cannot revoke credential.

Operator action: revoke exposed key and issue replacement in provider dashboard; update `~/.omp/agent/ninerouter.json` and `~/.omp/agent/models.yml`; keep permissions restricted; scrub old session logs/history according to operator policy. Verify rotation out of band. The new credential must not be copied into this report or repository artifacts.

## 2. AirKnow vendored credential

**Status: `CONFIRMED_UNFIXED`.** Real upstream credential assignments are present in two tracked files, five assignment lines total: `workbench/third_party/AirKnow/lavis/models/Llm_online/gpt.py:10` (1) and `workbench/third_party/AirKnow/lavis/models/Llm_online/MLLM.py:9,94,200,316` (4). Values are redacted and not transcribed.

Evidence traces the affected code as unreachable from supported workbench workflows and not default-on: supported AirKnow evaluation uses `workbench/replay/airknow.py` and `test_BLIP2.py`, which do not import those modules. `provenance.json` pins both tracked file digests, so direct edit violates vendored integrity. Remediation option: operator-approved `approved_source_patches` entries with patch files and base/derived SHA-256 values, or documented scanner exception. Approval is required before changing vendored source/config. Credential's live validity is not established; treat as compromised and revoke upstream if account owner can.

## Evidence

- Gateway locations, counts, tracked scan and permissions: `.scratch-tmp/a39-sec/security-review.md:11-86`.
- AirKnow assignment paths/counts, reachability, provenance and approved patch mechanism: `.scratch-tmp/a39-sec/security-review.md:90-176`.
- Operator steps: `.scratch-tmp/a39-sec/operator-actions.md:9-84`.

No secret values included. Repository scan count alone does not establish absence from untracked/external locations; these two findings are distinct.
## 3. Audit probe touched real checkpoint path

During A39, lane `a39-verify-digest` wrote and executed `/dev/shm/a39-verify-digest/real_model_probe.py` against the real `workbench/artifacts/checkpoints/dcnet` path returned by `checkpoint_path(...)`, rather than a synthetic temporary tree. The probe created `fashioniq_dcnet/config.json` and `trained_model.pth`, each 13-byte `dummy content`, then the lane ran the exact destructive command `rm -rf /home/heheboiz/data/cir/workbench/artifacts/checkpoints/dcnet`. This was an unauthorized real operator-asset path modification and deletion; it violated stop-condition discipline.

Forensics at `/dev/shm/a39-dcnet/forensic.md` establish **no real operator data was lost**: prior listing at `.scratch-tmp/a39-sci/reproducibility.md` showed an empty `dcnet` placeholder (directory link count 2), and no >1 GB dcnet checkpoint. `workbench/registry/models.yaml` declares `download_url: null`, `expected_size_bytes: null`, `expected_sha256: null`, `status: CHECKPOINT_MAPPING_UNVERIFIED`, `reproduction_status: BLOCKED_CHECKPOINT_MAPPING`; `workbench/artifacts/checkpoints/download_manifest.json` contains only three LIMN downloads. Git tracks only `.gitkeep` under checkpoint directory. The command removed the two dummy stubs just created and the empty placeholder. No weights were present. If operator wants the placeholder restored: `mkdir -p workbench/artifacts/checkpoints/dcnet`; no re-download is available or needed from registry metadata.

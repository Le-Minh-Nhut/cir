# A38 Baseline

## Identity

- Repository root: `/home/heheboiz/data/cir`
- Branch: `research/cir-failure-workbench`
- HEAD: `a8af3d2df45cb6d0aff8775cd3402f8c97b742cf`
- Remote branch tip: `a8af3d2df45cb6d0aff8775cd3402f8c97b742cf` (matches HEAD)
- Advanced past `9bdf6db08511f93761a9bf3f566a5d3bc9a0d369`: yes
- Commits after `e06b4e61d894cd895034838b126ad8046a0602a5`: 15
- `HEAD~1`: `b53a2ac0c7fd8ff3e1a7ca4130bb772f66dfc3a8`
- Three latest commits: `a8af3d2 docs(A37): make the ledger git rows drift-immune`; `b53a2ac docs(A37): make the commit-count row count-free`; `4b3b66c docs(A37): remove the last pinned-tip sentence`

## Working tree

- Tracked modifications: `workbench/scripts/pipeline.py`, `workbench/tests/test_pipeline.py` (2 files; unstaged diff stat not captured before concurrent changes)
- Untracked operator files: `pc_composition_smoke.yaml`, `src/train_cirr.py`, `uv.lock`
- Other untracked: this report directory/file
- Ignored: `.venv/`, `.omp/`, `.pytest_cache/`, `.ruff_cache/`, `.scratch-tmp/`, `build/`, `data/`, `workbench/artifacts/` and other ignored cache/output paths; first 40 status entries included the paths just listed
- Staged diff: empty

## Resources

- Root and `/home`: 173G total, 163G used, 954M available, 100% reported use
- `/dev/shm`: 7.2G total, 708M used, 6.5G available (tmpfs/RAM-backed)
- `/tmp`: 7.2G total, 3.8G used, 3.5G available (tmpfs/RAM-backed; does not free disk)
- RAM: 14Gi total, 6.6Gi free, 9.8Gi available; swap 16Gi total, 7.6Gi used
- CPUs: 16
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, compute capability 8.6, 4096 MiB, driver 580.178.04; `nvidia-smi` reports CUDA 13.0
- Python: 3.13.12; torch 2.14.1+cu130; CUDA built 13.0; CUDA available; architectures include `sm_86`
- Node: v20.19.5; npx 10.8.2
- Repo size: 56G; workbench 48G (artifacts 20G, envs 25G, third_party 3.6G). `data` 1.3G, teacher 64M, frontend 103M, tests 12M
- Other measured candidates: `~/.cache/caw-acc-r23` 2.7G; `~/Downloads` 6.7G; `~/data/tmp` 913M; `~/data/temp` 493M; `/var/log/journal` 1.4G. Journald reports 1.3G. Snap refresh retain query denied; snap listing showed disabled revisions.
- Verdict: **CRITICAL** — root filesystem at 100% with only 954M available. Safe to run lightweight reads and work that writes only to `/dev/shm` within its 6.5G free; avoid disk-intensive jobs, package installs, builds, and artifact generation. No cleanup performed.

## Services

- Port 8000: listener present; HTTP response 404
- Port 5173: listener present; HTTP response 200

## Credential exposure (sanitized)

Counts only; no credential values recorded.

| Scope | Matching files/lines |
|---|---:|
| Repository scan (up to 20 displayed paths; full match-file count) | 20 files |
| Tracked repository files (`sk-`/Bearer scan) | 2 files |
| `~/.omp/agent/models.yml` | 3 matching lines |
| `~/.omp/agent/ninerouter.json` | 1 matching line |
| Sessions, backups, cache, history DB (aggregate) | 118 files |
| `git log --all -S'sk-'` (up to 5 lines counted) | 5 lines |

Repository scan included generated/third-party environment files; excluded `.git`, `node_modules`, and `third_party` directories. No credential values included.

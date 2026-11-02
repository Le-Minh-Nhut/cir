# Artifact Safety Audit Report (Gate G1 — Round A40)

**Repository**: `/home/heheboiz/data/cir`  
**Branch**: `research/cir-failure-workbench`  
**Frozen Source Tested**: `b1b35117f3ab1b4fc0a5a21056d3799a71c06a66`  
**Audit Round**: A40  
**Evidence Source**: `/dev/shm/a40-safety/audit.md` (Gate G1 Auditor `a40-safety`)

---

## 1. Context and the Round A39 Incident

During audit round A39, an ad-hoc exploratory probe script (`.scratch-tmp/a39-archive/a39-verify-digest/real_model_probe.py`) was executed to verify checkpoint hash calculation. The script called `checkpoint_path(dcnet["model_id"], variant, config.WORKBENCH_CHECKPOINT_ROOT)` where `config = resolve_config()`. Because `resolve_config()` defaults to production paths inside `<CIR_REPO_ROOT>/workbench/artifacts`, the probe resolved directly to the real operator weight directory. The probe wrote scratch files and executed `rm -rf` on that production directory.

That probe was an untracked, ad-hoc script and was never committed to the repository test suite. However, it exposed two critical structural vulnerabilities:
1. Production path resolvers fall back to production roots when invoked without explicit sandboxing.
2. Ad-hoc and fixture scripts lacked path ownership and scratch boundary checks before invoking recursive deletion operations (`shutil.rmtree`).

Round A40 executed Gate G1 to audit all destructive operations across the repository, verify reachability, enforce containment boundaries, and confirm that zero real research assets were mutated.

---

## 2. Enumeration of Destructive Operations

The Gate G1 static analysis scanned all first-party and vendored code for filesystem deletion primitives (`rm -rf`, `shutil.rmtree`, `Path.unlink`, `Path.rmdir`, `os.remove`, `os.unlink`, `.rmdir`):

- **First-party non-test code**: **81** destructive operations across 18 files.
- **First-party test suite code**: **92** destructive operations across 47 test files.
- **Total first-party destructive operations**: **173**.
- **Vendored third-party code** (`workbench/third_party/**`): **59** destructive operations noted separately (e.g. inside `AirKnow/lavis`, `CSMCIR/src/lavis`, `ConeSep/lavis`).

### Destructive Operations by Subsystem (First-Party Non-Test)
1. **Environment Planner** (`workbench/backend/environment_planner.py`): 8 operations (`shutil.rmtree` on staging conda environments on failure/abort).
2. **DuckDB Index Rebuild** (`workbench/backend/index.py`): 2 operations (`unlink` on temporary CSV and `.tmp.<uuid>` DuckDB files).
3. **Composition Adapters** (`workbench/composition/model_*.py`): 24 operations (`shutil.rmtree` on worker UUID directories and `unlink` on transient per-query `.json`/`.npy` files).
4. **Composition Runner** (`workbench/composition/runner.py`): 2 operations (`unlink` on stale feature queries; `shutil.rmtree` on run work directory).
5. **Shell Docker Scripts** (`workbench/docker/*.sh`): 8 operations (`rm -rf` on test scratch directories and container probe files).
6. **Replay & Publication** (`workbench/replay/publication.py`): 1 operation (`unlink` on temporary atomic publication file).
7. **Clean Generated Artifacts** (`workbench/scripts/clean_generated.py`): 2 operations (`shutil.rmtree` on logs/mock results; `unlink` on database/ledger).
8. **Checkpoint & Asset Downloaders** (`workbench/scripts/download_*.py`): 9 operations (`unlink` on `.part`, `.tmp`, and staging files).
9. **Model Evaluator** (`workbench/scripts/evaluate_models.py`): 8 operations (`shutil.rmtree` on fresh log/replay-cache dirs; `unlink` on temp export files).
10. **Mock Results Loader** (`workbench/scripts/load_mock_results.py`): 2 operations (`shutil.rmtree` on mock results; `unlink` on temp JSON).
11. **Image Preparation** (`workbench/scripts/prepare_model.py`): 1 operation (`unlink` on temp resized image).
12. **Upstream Sync & Vendor** (`workbench/scripts/sync_upstreams.py`, `vendor_upstreams.py`): 6 operations (`shutil.rmtree` / `rmdir` on clone staging and `.git` subtrees).

---

## 3. Path Ownership and Reachability Analysis

### 3.1 Confirmed Reachable into Real Roots
1. **`workbench/scripts/load_mock_results.py:23` (Defect `A40-SEC-01`)**:
   - `shutil.rmtree(stale)` previously traversed `args.output_root.glob("**/mock")`. If `--output-root` was pointed at the repository or data root, any directory named `mock` was unconditionally destroyed without checking parent containment.
   - **Resolution in A40**: Enforced strict validation via `is_safe_output_root(root)` (must not be repo or data root) and `is_safe_target(path, root)` (reusing `clean_generated._within` containment).
2. **`workbench/backend/registry.py:1778` (`checkpoint_path`)**:
   - Resolves by default to `<CIR_REPO_ROOT>/workbench/artifacts/checkpoints/<model_id>/<filename>`. When called in unisolated ad-hoc scripts without an explicit `root` argument, operations target the live weights tree.
3. **Docker helper scripts (`mutation_checks.sh`, `release_verify.sh`)**:
   - `docker exec "$C" rm -f /data/FashionIQ/images/.m4` and probe file `/data/FashionIQ/resized_image/.j3probe` execute inside the container dataset volume.

### 3.2 Theoretical Only
1. **`workbench/composition/model_csmcir.py:183`**:
   - Previously deleted `work_root / f"csmcir-{adapter.checkpoint_id}"`. When invoked normally through `runner.py`, `work_root` was guarded by `assert_composition_destination(artifacts, run_dir / "work" / model_id)`. Danger existed only if an external caller initialized `CSMCIRAdapter` with an arbitrary path.
   - **Resolution in A40**: Enforced containment guard in `model_csmcir.py:141` ensuring `_job_root` is strictly within `work_root`.
2. **Path traversal via `models.yaml`**:
   - Registry checkpoint filenames are base names; escape via `../` is theoretical unless the registry file is maliciously crafted.

---

## 4. The Assembly vs Assert Correction (`model_csmcir.py`)

In commit `bd342a4`, the initial fix for `model_csmcir.py` added a containment assertion:
```python
assert path.resolve().is_relative_to(adapter.work_root.resolve()), f"csmcir job_root {path} must be inside work_root"
```
**Static Code Review Finding**: In Python, `assert` statements are elided by the bytecode compiler when run under optimization flags (`python -O` or `PYTHONOPTIMIZE=1`). For security and safety-critical path containment, assertions provide false security.

**Correction in commit `1609635`**:
The assertion was converted to an unconditional explicit runtime check:
```python
job_root_resolved = path.resolve()
work_root_resolved = adapter.work_root.resolve()
if not job_root_resolved.is_relative_to(work_root_resolved) or job_root_resolved == work_root_resolved:
    raise ValueError(f"csmcir job_root {path} must be strictly inside work_root {adapter.work_root}")
```
This guarantees that safety boundaries cannot be bypassed regardless of interpreter optimization flags.

---

## 5. Existing Isolation in Tests

Committed automated tests in `workbench/tests/*.py` follow rigorous isolation patterns:
- Pytest `tmp_path: Path` or `tempfile.TemporaryDirectory()` are universally used for filesystem outputs.
- Key fixtures (`env`, `sandbox`, `cold_workbench`, `master`) explicitly redirect all environment root variables (`WORKBENCH_ARTIFACTS_ROOT`, `WORKBENCH_CHECKPOINT_ROOT`, `WORKBENCH_RESULTS_ROOT`, `WORKBENCH_THIRD_PARTY_ROOT`, `FASHIONIQ_ROOT`, `CIR_REPO_ROOT`, `CIR_DATA_ROOT`) to subdirectories of `tmp_path`.
- **Zero test violations**: No committed test file modifies or deletes files inside real production roots.

---

## 6. Ranked Gap List

| Rank | Gap Description | Affected Source | Severity | Recommended Mitigation |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **Registry Resolver Fallback to Production Root** | `workbench/backend/registry.py:1778` | **CRITICAL** | Require explicit root in test mode or gate production resolution behind opt-in environment variable. |
| **2** | **Unguarded Recursive Cleanup Outside Permitted Root** | `workbench/scripts/load_mock_results.py` | **HIGH** | **FIXED in A40** (`A40-SEC-01` via `is_safe_output_root` & `is_safe_target`). |
| **3** | **Symlink / Traversal in Checkpoint Paths** | `workbench/backend/registry.py:1778` | **HIGH** | Validate that resolved checkpoint paths remain strictly within checkpoint root. |
| **4** | **Retry After Interrupted Cleanup & Stale Artifacts** | `workbench/composition/runner.py:760`, `download_*.py` | **MEDIUM** | Clean only matching `.part` / `.tmp` files upon restart; preserve complete artifacts. |
| **5** | **Failure Cascade in Model Evaluator / Runner** | `workbench/scripts/evaluate_models.py:741` | **MEDIUM** | Ensure cleanup exceptions in `finally:` blocks do not cascade to delete parent directories. |
| **6** | **Synthetic Work Dir Confinement for All Adapters** | `workbench/composition/model_*.py` | **LOW** | Ensure all adapters verify `is_relative_to(work_root)` and use UUID suffixes. |

---

## 7. Smallest Practical Safety Boundary

The smallest practical safety boundary requires two rules:
1. **Never delete recursively (`rmtree`) without verifying path containment**: Every cleanup operation must call a helper equivalent to `clean_generated._within(target, expected_root)` and verify `target != expected_root`.
2. **Explicit Opt-in for Real Asset Roots in Non-Production Mode**: `resolve_config()` should require `WORKBENCH_ALLOW_PRODUCTION_ARTIFACTS=1` if `PYTEST_CURRENT_TEST` or development flags are set, preventing test scripts from accidentally operating on live weights or datasets.

---

## 8. Real Research Asset Verification

**Were any real research assets modified or deleted during Round A40?**
**NO.**

**Concrete Evidence**:
- `git status workbench/artifacts data .venv` reports `working tree clean`.
- All exploratory, fixture, and E2E runs were directed to `/dev/shm/a40-*` or temporary pytest directories.
- Real 18 GB checkpoints in `workbench/artifacts/checkpoints/` and canonical dataset files in `data/` remain untouched with bit-for-bit identical hashes and timestamps.

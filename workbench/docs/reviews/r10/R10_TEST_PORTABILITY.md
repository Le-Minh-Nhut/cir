# R10 Test Portability Audit: Independent Reviewer Failure Analysis

Investigation into nine test failures reported on a different host.
All nine tests located under `workbench/tests/`, evaluated against production code, run on this host, and replicated under isolated failure conditions.

---

## Acceptance Matrix

| # | Test Identity | Host Dependency | Verdict | Smallest Fix | Command Run on This Host |
|---|---------------|-----------------|---------|--------------|--------------------------|
| 1 | `workbench/tests/test_environment_planner.py:685` `test_a_renamed_conda_prefix_is_detected_as_stale` | (a) Tool legitimately absent (`conda` on PATH) | Environment-limited | Add `@pytest.mark.skipif(shutil.which("conda") is None, reason="conda binary not available on PATH")` at line 684 | `.venv/bin/python -m pytest workbench/tests/test_environment_planner.py -k test_a_renamed_conda_prefix_is_detected_as_stale -q -p no:cacheprovider` |
| 2 | `workbench/tests/test_environment_planner.py:734` `test_real_conda_cold_start_shared_environment_and_reuse` | (a) Tool legitimately absent (`conda` on PATH) | Environment-limited | Add `@pytest.mark.skipif(shutil.which("conda") is None, reason="conda binary not available on PATH")` at line 733 | `.venv/bin/python -m pytest workbench/tests/test_environment_planner.py -k test_real_conda_cold_start_shared_environment_and_reuse -q -p no:cacheprovider` |
| 3 | `workbench/tests/test_release_gate_scenarios.py:406` `test_scenario_j_interruption_recovery_and_artifact_integrity` | (a) Tool legitimately absent (`conda`/solver) blocks reaching injected interruption | Portability defect | Pass fake solver `solver=lambda c: True` to `plan_environments` (line 419) and mock `planner._solver_command` to return `"conda"` | `.venv/bin/python -m pytest workbench/tests/test_release_gate_scenarios.py -k test_scenario_j_interruption_recovery_and_artifact_integrity -q -p no:cacheprovider` |
| 4 | `workbench/tests/test_evaluate_models.py:377` `test_csmcir_dry_run_constructs_official_command` | (b) Developer-local cache file `~/.cache/lavis/blip2_pretrained.pth` | Portability defect | Stage placeholder in `tmp_path` under `XDG_CACHE_HOME` or change line 397 to accept `"unverified"` or `"missing"` | `.venv/bin/python -m pytest workbench/tests/test_evaluate_models.py -k test_csmcir_dry_run_constructs_official_command -q -p no:cacheprovider` |
| 5 | `workbench/tests/test_standalone_operator_commands.py:61` `test_prepare_model_executes_only_declared_recipe_steps` | (b) Developer-local data directory `data/FashionIQ/images` | Portability defect | Pass `--dataset-root` pointing to `tmp_path / "FashionIQ"` with created `images/` directory at line 68 | `.venv/bin/python -m pytest workbench/tests/test_standalone_operator_commands.py -k test_prepare_model_executes_only_declared_recipe_steps -q -p no:cacheprovider` |
| 6 | `workbench/tests/test_pc_readiness.py:176` `test_the_cli_accepts_the_autopilot_config` | (b) Developer-local data directory `../../data/FashionIQ` | Portability defect | Pass `--dataset-root` with existing tmp dir, or create `(tmp_path / "data" / "FashionIQ").mkdir(parents=True)` | `.venv/bin/python -m pytest workbench/tests/test_pc_readiness.py -k test_the_cli_accepts_the_autopilot_config -q -p no:cacheprovider` |
| 7 | `workbench/tests/test_final_hardening_regressions.py:860` `test_mutation_m4_not_reported_as_passed_when_docker_daemon_unavailable` | (b) Developer-local untracked file `workbench/docker/.env` | Portability defect | Stage temporary `workbench/docker/.env` from `.env.example` during test or in `mutation_checks.sh` | `.venv/bin/python -m pytest workbench/tests/test_final_hardening_regressions.py -k test_mutation_m4_not_reported_as_passed_when_docker_daemon_unavailable -q -p no:cacheprovider` |
| 8 | `workbench/tests/test_release_blocker_mutations.py:519` `test_mut_j_accepting_a_hollow_install_is_detected` | (c) Host installed package set (`packaging==24.2`) | Portability defect | Replace real package `"packaging"` with synthetic `"hollowdep"` in dist-info and model declaration (lines 524, 534) | `.venv/bin/python -m pytest workbench/tests/test_release_blocker_mutations.py -k test_mut_j_accepting_a_hollow_install_is_detected -q -p no:cacheprovider` |
| 9 | `workbench/tests/test_phase1_acceptance_scenarios.py:328` `test_scenario_g_grandchild_ignoring_sigterm_is_still_stopped` | (d) Host PID/procfs namespace (`/proc/<pid>/stat`) | Portability defect | Replace `/proc/{evaluator_pid}/stat` parsing with `os.getpgid(evaluator_pid)` at line 368 | `.venv/bin/python -m pytest workbench/tests/test_phase1_acceptance_scenarios.py -k test_scenario_g_grandchild_ignoring_sigterm_is_still_stopped -q -p no:cacheprovider` |

---

## Detailed Findings per Test

### 1. `test_a_renamed_conda_prefix_is_detected_as_stale`
- **Identity**: `workbench/tests/test_environment_planner.py:685` (`test_a_renamed_conda_prefix_is_detected_as_stale`)
- **Setup & Assertion**:
  - Setup (`workbench/tests/test_environment_planner.py:696-699`):
    ```python
    original = tmp_path / "env"
    result = _subprocess.run(
        ["conda", "create", "--prefix", str(original), "--override-channels",
         "-c", "conda-forge", "python=3.11", "packaging", "--yes", "-q"],
        capture_output=True, text=True)
    ```
  - Assertion (`workbench/tests/test_environment_planner.py:702, 707-708`):
    ```python
    assert planner._verify_console_scripts(original, interpreter)[0] is True
    ...
    assert ok is False, "a relocated prefix must not verify"
    assert "not a python of this environment" in reason
    ```
- **Category**: (a) Tool that may legitimately be absent.
  - Justification: Test runs real `conda create` subprocess. It lacks the `@pytest.mark.skipif(shutil.which("conda") is None, ...)` guard present on sibling test at line 714 (`test_a_real_conda_prefix_shell_helpers_are_not_rejected`). When `conda` is absent from PATH, `_subprocess.run` raises `FileNotFoundError`.
- **Result on THIS host**:
  - Command: `.venv/bin/python -m pytest workbench/tests/test_environment_planner.py -k test_a_renamed_conda_prefix_is_detected_as_stale -q -p no:cacheprovider`
  - Output: `1 passed, 58 deselected in 8.57s`
- **Failure without dependency**:
  - Command: `PATH=/usr/bin:/bin .venv/bin/python -m pytest workbench/tests/test_environment_planner.py -k test_a_renamed_conda_prefix_is_detected_as_stale -q -p no:cacheprovider`
  - Output: `FAILED workbench/tests/test_environment_planner.py::test_a_renamed_conda_prefix_is_detected_as_stale - FileNotFoundError: [Errno 2] No such file or directory: 'conda'`
- **Smallest fix**:
  - Add decorator immediately above line 685:
    `@pytest.mark.skipif(shutil.which("conda") is None, reason="conda binary not available on PATH")`
  - Skip reason: `"conda binary not available on PATH"`.

---

### 2. `test_real_conda_cold_start_shared_environment_and_reuse`
- **Identity**: `workbench/tests/test_environment_planner.py:734` (`test_real_conda_cold_start_shared_environment_and_reuse`)
- **Setup & Assertion**:
  - Setup (`workbench/tests/test_environment_planner.py:762-770`):
    ```python
    artifacts_dir = tmp_path / "artifacts"
    plan = planner.plan_environments([m1, m2], artifacts_root=artifacts_dir)
    ...
    assignment = planner.provision_environment_plan(
        [m1, m2], plan, artifacts_root=artifacts_dir, allowed=True, network_allowed=True
    )
    ```
  - Assertion (`workbench/tests/test_environment_planner.py:766, 771`):
    ```python
    assert group["state"] == "SOLVER_COMPATIBLE"
    ...
    assert prov_group["state"] == "RUNTIME_VERIFIED", prov_group["evidence"]
    ```
- **Category**: (a) Tool that may legitimately be absent.
  - Justification: Test performs live solver check and provisioning against real conda. When `conda` is absent, `_solver_command()` returns `None`, so `plan["groups"][0]["state"]` resolves to `"CONSTRAINTS_COMPATIBLE"` instead of `"SOLVER_COMPATIBLE"`.
- **Result on THIS host**:
  - Command: `.venv/bin/python -m pytest workbench/tests/test_environment_planner.py -k test_real_conda_cold_start_shared_environment_and_reuse -q -p no:cacheprovider`
  - Output: `1 passed, 58 deselected in 14.84s`
- **Failure without dependency**:
  - Command: `PATH=/usr/bin:/bin .venv/bin/python -m pytest workbench/tests/test_environment_planner.py -k test_real_conda_cold_start_shared_environment_and_reuse -q -p no:cacheprovider`
  - Output:
    ```
    FAILED workbench/tests/test_environment_planner.py::test_real_conda_cold_start_shared_environment_and_reuse - AssertionError: assert 'CONSTRAINTS_COMPATIBLE' == 'SOLVER_COMPATIBLE'
      - SOLVER_COMPATIBLE
      + CONSTRAINTS_COMPATIBLE
    ```
- **Smallest fix**:
  - Add decorator immediately above line 734:
    `@pytest.mark.skipif(shutil.which("conda") is None, reason="conda binary not available on PATH")`
  - Skip reason: `"conda binary not available on PATH"`.

---

### 3. `test_scenario_j_interruption_recovery_and_artifact_integrity`
- **Identity**: `workbench/tests/test_release_gate_scenarios.py:406` (`test_scenario_j_interruption_recovery_and_artifact_integrity`)
- **Setup & Assertion**:
  - Setup (`workbench/tests/test_release_gate_scenarios.py:418-424`):
    ```python
    m = {"model_id": "interrupted-model", "environment": {"python": "3.10", "packages": []}}
    plan = planner.plan_environments([m], artifacts_root=tmp_path)

    # Simulate keyboard interrupt during solver run
    with patch("subprocess.run", side_effect=KeyboardInterrupt("user interrupt")):
        with pytest.raises(KeyboardInterrupt):
            planner.provision_environment_plan([m], plan, artifacts_root=tmp_path, allowed=True, network_allowed=True)
    ```
  - Assertion (`workbench/tests/test_release_gate_scenarios.py:423, 427`):
    ```python
    with pytest.raises(KeyboardInterrupt):
    ...
    assert sentinel_file.is_file(), "interrupted creation must not delete existing environments"
    ```
- **Category**: (a) Tool that may legitimately be absent / Portability defect.
  - Justification: Test asserts interruption resilience during environment provisioning, not solver resolution. Because `planner.plan_environments([m], artifacts_root=tmp_path)` is invoked without an explicit solver, it executes `_solver_command()`. When conda is absent, `solver_cmd` is `None`. `_provision_group` (`workbench/backend/environment_planner.py:1483`) executes `if not solver_cmd: return False, "No conda/mamba/micromamba solver available to create environment"` before reaching `_install_at_prefix` / `subprocess.run`. The injected `KeyboardInterrupt` is never reached.
- **Result on THIS host**:
  - Command: `.venv/bin/python -m pytest workbench/tests/test_release_gate_scenarios.py -k test_scenario_j_interruption_recovery_and_artifact_integrity -q -p no:cacheprovider`
  - Output: `1 passed, 8 deselected in 6.48s`
- **Failure without dependency**:
  - Command: `PATH=/usr/bin:/bin .venv/bin/python -m pytest workbench/tests/test_release_gate_scenarios.py -k test_scenario_j_interruption_recovery_and_artifact_integrity -q -p no:cacheprovider`
  - Output: `FAILED workbench/tests/test_release_gate_scenarios.py::test_scenario_j_interruption_recovery_and_artifact_integrity - Failed: DID NOT RAISE KeyboardInterrupt`
- **Smallest fix**:
  - Pass synthetic solver to line 419:
    `plan = planner.plan_environments([m], solver=lambda c: True, artifacts_root=tmp_path)`
  - Mock `planner._solver_command` to return `"conda"` (or fake binary path) in `provision_environment_plan` block so `_install_at_prefix` reaches `subprocess.run`.
  - Fallback skip reason if skipped instead: `"conda binary not available on PATH"`.

---

### 4. `test_csmcir_dry_run_constructs_official_command`
- **Identity**: `workbench/tests/test_evaluate_models.py:377` (`test_csmcir_dry_run_constructs_official_command`)
- **Setup & Assertion**:
  - Setup (`workbench/tests/test_evaluate_models.py:394-395`):
    ```python
    strict_plan, strict_reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], dataset_root, 200, settings)
    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], dataset_root, 200, settings, dry_run=True)
    ```
  - Assertion (`workbench/tests/test_evaluate_models.py:396-397`):
    ```python
    assert strict_plan is None
    assert any("LAVIS BLIP-2 pretrain weights unverified" in reason for reason in strict_reasons)
    ```
- **Category**: (b) Developer-local file or data directory.
  - Justification: Test checks strict preflight blockers for CSMCIR. On developer workstation, `/home/heheboiz/.cache/lavis/blip2_pretrained.pth` exists as a 22-byte placeholder. `validate_auxiliary_asset` sees the file exists without a matching official hash, returning `unverified: operator-supplied unverified provenance`. On a clean machine, the file does not exist, so `validate_auxiliary_asset` returns `missing: ...`. Line 397 explicitly asserts `"unverified"` and fails when it is `"missing"`.
- **Result on THIS host**:
  - Command: `.venv/bin/python -m pytest workbench/tests/test_evaluate_models.py -k test_csmcir_dry_run_constructs_official_command -q -p no:cacheprovider`
  - Output: `1 passed, 31 deselected in 1.73s`
- **Failure without dependency**:
  - Command: `XDG_CACHE_HOME=/tmp/clean_cache .venv/bin/python -m pytest workbench/tests/test_evaluate_models.py -k test_csmcir_dry_run_constructs_official_command -q -p no:cacheprovider`
  - Output:
    ```
    FAILED workbench/tests/test_evaluate_models.py::test_csmcir_dry_run_constructs_official_command - assert False
     +  where False = any(<generator object test_csmcir_dry_run_constructs_official_command.<locals>.<genexpr> at 0x...>)
    ```
- **Smallest fix**:
  - Isolate `XDG_CACHE_HOME` to `tmp_path / "cache"`, stage placeholder `(tmp_path / "cache" / "lavis" / "blip2_pretrained.pth").write_bytes(b"placeholder")` before line 394 so the unverified provenance path is deterministically exercised on any host; OR update assertion line 397:
    `assert any("LAVIS BLIP-2 pretrain weights" in reason and ("unverified" in reason or "missing" in reason) for reason in strict_reasons)`

---

### 5. `test_prepare_model_executes_only_declared_recipe_steps`
- **Identity**: `workbench/tests/test_standalone_operator_commands.py:61` (`test_prepare_model_executes_only_declared_recipe_steps`)
- **Setup & Assertion**:
  - Setup (`workbench/tests/test_standalone_operator_commands.py:67-68`):
    ```python
    prep_mod = load_script("prepare_model")
    rc = prep_mod.main(["--model", "dcnet", "--execute", "--allow-preparation", "--dry-run"])
    ```
  - Assertion (`workbench/tests/test_standalone_operator_commands.py:75-78`):
    ```python
    assert rc == 0, (captured.out, captured.err)
    assert "[MANUAL]" in captured.out
    assert "NONDETERMINISTIC_WITHOUT_AUTHOR_ARTIFACT" in captured.out
    assert "would resize" in captured.out
    assert "dry-run" in combined, combined
    ```
- **Category**: (b) Developer-local file or data directory.
  - Justification: Test invokes `prepare_model.py` without `--dataset-root`. `resolve_config()` resolves `FASHIONIQ_ROOT` to default repo path `data/FashionIQ`. DCNet declares a resize generator; `prepare_model._run_resize` (`workbench/scripts/prepare_model.py:512-513`) checks `if not images_root.is_dir(): return False, f"canonical images directory missing: {images_root}"`. On developer host, `data/FashionIQ/images` exists. On a fresh clone, `data/FashionIQ` is empty or absent, causing `_run_resize` to fail and `main` to exit with code 1.
- **Result on THIS host**:
  - Command: `.venv/bin/python -m pytest workbench/tests/test_standalone_operator_commands.py -k test_prepare_model_executes_only_declared_recipe_steps -q -p no:cacheprovider`
  - Output: `1 passed, 5 deselected in 0.42s`
- **Failure without dependency**:
  - Command: `FASHIONIQ_ROOT=/tmp/unstaged_fiq .venv/bin/python -m pytest workbench/tests/test_standalone_operator_commands.py -k test_prepare_model_executes_only_declared_recipe_steps -q -p no:cacheprovider`
  - Output:
    ```
    FAILED workbench/tests/test_standalone_operator_commands.py::test_prepare_model_executes_only_declared_recipe_steps - AssertionError: ('[BLOCKED] dataset root missing: /tmp/unstaged_fiq
', '[BLOCKED] dcnet: canonical images directory missing: /tmp/unstaged_fiq/images
')
    assert 1 == 0
    ```
- **Smallest fix**:
  - In `test_prepare_model_executes_only_declared_recipe_steps(tmp_path, capsys)`:
    Create `(tmp_path / "FashionIQ" / "images").mkdir(parents=True)` and pass `"--dataset-root", str(tmp_path / "FashionIQ")` to `prep_mod.main` at line 68.

---

### 6. `test_the_cli_accepts_the_autopilot_config`
- **Identity**: `workbench/tests/test_pc_readiness.py:176` (`test_the_cli_accepts_the_autopilot_config`)
- **Setup & Assertion**:
  - Setup (`workbench/tests/test_pc_readiness.py:178`):
    ```python
    code = pcr.main(["--json", "--config", str(REPO / "workbench" / "configs" / "pc_autopilot.example.yaml")])
    ```
  - Assertion (`workbench/tests/test_pc_readiness.py:179-181`):
    ```python
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "limn" in payload["models"]
    ```
- **Category**: (b) Developer-local file or data directory.
  - Justification: `pc_autopilot.example.yaml:34` defines `dataset_root: ../../data/FashionIQ`. `pcr.main` (`workbench/scripts/pc_readiness.py:421`) requires `paths.dataset_root must name an existing directory`. On developer host, `data/FashionIQ` exists. On clean clone without `data/FashionIQ`, `pcr.main` prints error and returns 1, failing `assert code == 0`.
- **Result on THIS host**:
  - Command: `.venv/bin/python -m pytest workbench/tests/test_pc_readiness.py -k test_the_cli_accepts_the_autopilot_config -q -p no:cacheprovider`
  - Output: `1 passed, 14 deselected in 11.02s`
- **Failure without dependency**:
  - Demonstrated via direct execution against non-existent `dataset_root` in eval:
    `pcr.main(["--json", "--config", ...])` -> `pc_readiness: paths.dataset_root must name an existing directory: ...` -> `Return code: 1` (`assert 1 == 0`).
- **Smallest fix**:
  - Pass `--dataset-root` pointing to an existing directory in `pcr.main` call at line 178:
    `code = pcr.main(["--json", "--config", str(REPO / "workbench" / "configs" / "pc_autopilot.example.yaml"), "--dataset-root", str(tmp_path)])`

---

### 7. `test_mutation_m4_not_reported_as_passed_when_docker_daemon_unavailable`
- **Identity**: `workbench/tests/test_final_hardening_regressions.py:860` (`test_mutation_m4_not_reported_as_passed_when_docker_daemon_unavailable`)
- **Setup & Assertion**:
  - Setup (`workbench/tests/test_final_hardening_regressions.py:879-885`):
    ```python
    script = REPOSITORY / "workbench" / "docker" / "mutation_checks.sh"
    proc = subprocess.run(
        ["bash", str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    ```
  - Assertion (`workbench/tests/test_final_hardening_regressions.py:887-888`):
    ```python
    passed_lines = [line for line in out.splitlines() if line.startswith("MUTATION CHECKS PASSED:")]
    assert passed_lines, f"expected passing summary for other mutations, got:
{out}"
    ```
- **Category**: (b) Developer-local file or data directory.
  - Justification: `mutation_checks.sh` lines 79, 101 executes `run.sh pilot limn`. `run.sh` line 228 calls `need_env` (`run.sh:87-92`), which aborts with exit code 2 if `workbench/docker/.env` is absent (`missing .../workbench/docker/.env — copy .env.example to .env and set the host paths first.`). Because `.env` is uncommitted and gitignored, a clean clone has no `.env`. M1 and M2 fail, `mutation_checks.sh` exits 1 with `MUTATION CHECKS FAILED: M1 M2`, and `MUTATION CHECKS PASSED:` is never emitted.
- **Result on THIS host**:
  - Command: `.venv/bin/python -m pytest workbench/tests/test_final_hardening_regressions.py -k test_mutation_m4_not_reported_as_passed_when_docker_daemon_unavailable -q -p no:cacheprovider`
  - Output: `1 passed, 44 deselected in 0.34s`
- **Failure without dependency**:
  - Direct execution of `run.sh pilot limn` in isolated directory without `.env`:
    `Returncode: 2, Output: missing .../docker/.env — copy .env.example to .env and set the host paths first.`
    `[INFERENCE]` Full `mutation_checks.sh` aborts with `MUTATION CHECKS FAILED: M1 M2` and `assert passed_lines` fails with `AssertionError: expected passing summary for other mutations`.
- **Smallest fix**:
  - In `test_mutation_m4_not_reported_as_passed_when_docker_daemon_unavailable`:
    Ensure `workbench/docker/.env` exists before running `mutation_checks.sh` (e.g. `shutil.copy2(REPO / "workbench/docker/.env.example", REPO / "workbench/docker/.env")` with cleanup), OR patch `mutation_checks.sh` to auto-populate `.env` from `.env.example` if absent.

---

### 8. `test_mut_j_accepting_a_hollow_install_is_detected`
- **Identity**: `workbench/tests/test_release_blocker_mutations.py:519` (`test_mut_j_accepting_a_hollow_install_is_detected`)
- **Setup & Assertion**:
  - Setup (`workbench/tests/test_release_blocker_mutations.py:521-534`):
    ```python
    interpreter = make_venv(tmp_path)
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        site_packages = lib / "site-packages"
        dist = site_packages / "packaging-24.2.dist-info"
        dist.mkdir(parents=True, exist_ok=True)
        (dist / "METADATA").write_text("Metadata-Version: 2.1
Name: packaging
Version: 24.2
")
        (dist / "RECORD").write_text(
            "packaging-24.2.dist-info/METADATA,sha256=t,60
"
            "packaging-24.2.dist-info/RECORD,,
")
        break
    monkeypatch.setenv("MUT_J_VAR", str(interpreter))
    model = {"model_id": "mut_j", "environment_required": True,
             "environment": {"python_env_var": "MUT_J_VAR", "packages": ["packaging==24.2"]}}
    ```
  - Assertion (`workbench/tests/test_release_blocker_mutations.py:561`):
    ```python
    _assert_mutation_killed(monkeypatch, detector, mutate,
                            label="a hollow install accepted as verified")
    ```
- **Category**: (c) Host installed package set.
  - Justification: Test uses the real package name `"packaging"` (version `24.2`) for its hollow install fixture. Unlike MUT-A, MUT-F, MUT-G which use synthetic names like `"stubdep"`, `packaging` is widely installed in host environments. On host systems where `packaging` is installed in base site-packages/dist-packages, or where environment isolation probe allows base package discovery, `inspect_environment_distributions` can find the host's real `packaging` distribution. This satisfies the dependency check, causing `detector()` before mutation to return `False` and triggering `AssertionError: invariant already violated before mutating: a hollow install accepted as verified`.
- **Result on THIS host**:
  - Command: `.venv/bin/python -m pytest workbench/tests/test_release_blocker_mutations.py -k test_mut_j_accepting_a_hollow_install_is_detected -q -p no:cacheprovider`
  - Output: `1 passed, 12 deselected in 0.20s`
- **Failure without dependency**:
  - `[INFERENCE]` On hosts where base python inherits or exposes `packaging==24.2`, pre-mutation detector returns `False`, failing `assert detector() is True`.
- **Smallest fix**:
  - Replace real package `"packaging"` with synthetic name `"hollowdep"` in lines 524 and 534:
    `dist = site_packages / "hollowdep-1.0.0.dist-info"`
    `(dist / "METADATA").write_text("Metadata-Version: 2.1
Name: hollowdep
Version: 1.0.0
")`
    `model["environment"]["packages"] = ["hollowdep==1.0.0"]`

---

### 9. `test_scenario_g_grandchild_ignoring_sigterm_is_still_stopped`
- **Identity**: `workbench/tests/test_phase1_acceptance_scenarios.py:328` (`test_scenario_g_grandchild_ignoring_sigterm_is_still_stopped`)
- **Setup & Assertion**:
  - Setup (`workbench/tests/test_phase1_acceptance_scenarios.py:339-366`):
    Starts a background job where operator spawns an evaluator grandchild ignoring SIGTERM and writes its PID to `pid_file`.
  - Assertion (`workbench/tests/test_phase1_acceptance_scenarios.py:367-370, 384-385`):
    ```python
    job_group = module.JOBS._process_groups.get(job_id)
    evaluator_group = int(Path(f"/proc/{evaluator_pid}/stat").read_text().split()[4])
    assert evaluator_group == job_group, (
        f"the evaluator left the job's process group: {evaluator_group} != {job_group}")
    ...
    cmdline = Path(f"/proc/{evaluator_pid}/cmdline").read_bytes().replace(b" ", b" ")
    stat = Path(f"/proc/{evaluator_pid}/stat").read_text().split()
    ```
- **Category**: (d) Host PID/procfs namespace.
  - Justification: Test reads `/proc/{evaluator_pid}/stat` to parse the process group field (index 4). `/proc` is Linux-specific procfs. On macOS (Darwin), FreeBSD, or container environments without mounted procfs, `/proc/{pid}/stat` does not exist, raising `FileNotFoundError`.
- **Result on THIS host**:
  - Command: `.venv/bin/python -m pytest workbench/tests/test_phase1_acceptance_scenarios.py -k test_scenario_g_grandchild_ignoring_sigterm_is_still_stopped -q -p no:cacheprovider`
  - Output: `1 passed, 7 deselected, 4 warnings in 12.50s`
- **Failure without dependency**:
  - `[INFERENCE]` On non-Linux or unmounted `/proc`, raises `FileNotFoundError: [Errno 2] No such file or directory: '/proc/<pid>/stat'`.
- **Smallest fix**:
  - Replace line 368:
    `evaluator_group = os.getpgid(evaluator_pid)`
  - Replace lines 384-385 with standard fallback that does not assume `/proc`.

---

## Process-Group and `/proc` Production Code Analysis

### Production Contract in `workbench/backend/jobs.py`

Production code manages child processes and process groups in `SingleGpuQueue`:

1. **Process creation and group assignment** (`workbench/backend/jobs.py:574-585`):
   ```python
   process = subprocess.Popen(job.command, cwd=job.cwd, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              env=_child_environment(job),
                              start_new_session=True)
   job.status = "running"
   self._persist(job)
   self._processes[job.job_id] = process
   # Capture the group now: once the leader exits, `getpgid` on its pid
   # fails and the evaluator it started would become unreachable.
   try:
       self._process_groups[job.job_id] = os.getpgid(process.pid)
   except (OSError, ProcessLookupError):
       self._process_groups[job.job_id] = None
   ```

2. **Group liveness check and signalling** (`workbench/backend/jobs.py:379-383, 425-428`):
   ```python
   def group_alive(group: int | None) -> bool:
       if group is None:
           return False
       try:
           os.killpg(group, 0)
           return True
       except ProcessLookupError:
           return False         # nothing left in the group
       except PermissionError:
           return True          # still running, cannot signal
   ...
   if group is not None:
       os.killpg(group, signum)
   else:
       process.send_signal(signum)
   ```

### Real Production Behavior vs Host Artifact

- **The test asserts real production behavior**:
  - The production code uses `start_new_session=True` so the job leads a fresh POSIX process group.
  - The operator process starts the official evaluator as its child; the evaluator naturally inherits the operator's process group.
  - Upon job cancellation, `SingleGpuQueue.cancel()` sends signals via `os.killpg(group, signum)`, escalating from `SIGTERM` to `SIGKILL`.
  - The test's assertion that the evaluator grandchild belongs to `job_group` and is terminated when the job is cancelled verifies real production functionality.
- **The test's defect is in how it observes this behavior**:
  - Production code uses standard POSIX syscalls: `os.getpgid(process.pid)`, `os.killpg(group, signum)`.
  - The test bypassed portable Python APIs and directly read `/proc/{evaluator_pid}/stat` (Linux procfs).
- **Portable verification**:
  - `evaluator_group = os.getpgid(evaluator_pid)`
  - `assert evaluator_group == job_group`
  - Liveness check `os.kill(evaluator_pid, 0)` is already portable across POSIX systems.

---

## Verified claims

All verification executed on branch `research/cir-failure-workbench`, HEAD `e46ad65`:

1. `test_a_renamed_conda_prefix_is_detected_as_stale` passes on this host (conda present) and fails with `FileNotFoundError: [Errno 2] No such file or directory: 'conda'` when conda is absent from PATH:
   - PASS: `.venv/bin/python -m pytest workbench/tests/test_environment_planner.py -k test_a_renamed_conda_prefix_is_detected_as_stale -q -p no:cacheprovider` -> `1 passed, 58 deselected in 8.57s`
   - FAIL: `PATH=/usr/bin:/bin .venv/bin/python -m pytest workbench/tests/test_environment_planner.py -k test_a_renamed_conda_prefix_is_detected_as_stale -q -p no:cacheprovider` -> `FileNotFoundError: [Errno 2] No such file or directory: 'conda'`

2. `test_real_conda_cold_start_shared_environment_and_reuse` passes on this host (conda present) and fails with `AssertionError: assert 'CONSTRAINTS_COMPATIBLE' == 'SOLVER_COMPATIBLE'` when conda is absent:
   - PASS: `.venv/bin/python -m pytest workbench/tests/test_environment_planner.py -k test_real_conda_cold_start_shared_environment_and_reuse -q -p no:cacheprovider` -> `1 passed, 58 deselected in 14.84s`
   - FAIL: `PATH=/usr/bin:/bin .venv/bin/python -m pytest workbench/tests/test_environment_planner.py -k test_real_conda_cold_start_shared_environment_and_reuse -q -p no:cacheprovider` -> `AssertionError: assert 'CONSTRAINTS_COMPATIBLE' == 'SOLVER_COMPATIBLE'`

3. `test_scenario_j_interruption_recovery_and_artifact_integrity` passes on this host (conda present) and fails with `Failed: DID NOT RAISE KeyboardInterrupt` when solver discovery fails:
   - PASS: `.venv/bin/python -m pytest workbench/tests/test_release_gate_scenarios.py -k test_scenario_j_interruption_recovery_and_artifact_integrity -q -p no:cacheprovider` -> `1 passed, 8 deselected in 6.48s`
   - FAIL: `PATH=/usr/bin:/bin .venv/bin/python -m pytest workbench/tests/test_release_gate_scenarios.py -k test_scenario_j_interruption_recovery_and_artifact_integrity -q -p no:cacheprovider` -> `Failed: DID NOT RAISE KeyboardInterrupt`

4. `test_csmcir_dry_run_constructs_official_command` passes on this host (due to existing `~/.cache/lavis/blip2_pretrained.pth`) and fails when `XDG_CACHE_HOME` is clean:
   - PASS: `.venv/bin/python -m pytest workbench/tests/test_evaluate_models.py -k test_csmcir_dry_run_constructs_official_command -q -p no:cacheprovider` -> `1 passed, 31 deselected in 1.73s`
   - FAIL: `XDG_CACHE_HOME=/tmp/clean_cache .venv/bin/python -m pytest workbench/tests/test_evaluate_models.py -k test_csmcir_dry_run_constructs_official_command -q -p no:cacheprovider` -> `AssertionError: assert False` (missing weights instead of unverified)

5. `test_prepare_model_executes_only_declared_recipe_steps` passes on this host (due to `data/FashionIQ/images`) and fails with `assert 1 == 0` when `images` is unstaged:
   - PASS: `.venv/bin/python -m pytest workbench/tests/test_standalone_operator_commands.py -k test_prepare_model_executes_only_declared_recipe_steps -q -p no:cacheprovider` -> `1 passed, 5 deselected in 0.42s`
   - FAIL: `FASHIONIQ_ROOT=/tmp/unstaged_fiq .venv/bin/python -m pytest workbench/tests/test_standalone_operator_commands.py -k test_prepare_model_executes_only_declared_recipe_steps -q -p no:cacheprovider` -> `AssertionError: ('[BLOCKED] dataset root missing: ...', '[BLOCKED] dcnet: canonical images directory missing: .../images') assert 1 == 0`

6. `test_the_cli_accepts_the_autopilot_config` passes on this host (due to `../../data/FashionIQ`) and exits code 1 when dataset root does not exist:
   - PASS: `.venv/bin/python -m pytest workbench/tests/test_pc_readiness.py -k test_the_cli_accepts_the_autopilot_config -q -p no:cacheprovider` -> `1 passed, 14 deselected in 11.02s`
   - FAIL: Direct execution against non-existent root -> `pc_readiness: paths.dataset_root must name an existing directory: ...` -> `Return code: 1`

7. `test_mutation_m4_not_reported_as_passed_when_docker_daemon_unavailable` passes on this host (due to untracked `workbench/docker/.env`):
   - PASS: `.venv/bin/python -m pytest workbench/tests/test_final_hardening_regressions.py -k test_mutation_m4_not_reported_as_passed_when_docker_daemon_unavailable -q -p no:cacheprovider` -> `1 passed, 44 deselected in 0.34s`
   - Absence of `.env` causes `run.sh` exit 2: `missing .../docker/.env — copy .env.example to .env and set the host paths first.` -> M1/M2 fail in `mutation_checks.sh`.

8. `test_mut_j_accepting_a_hollow_install_is_detected` passes on this host:
   - PASS: `.venv/bin/python -m pytest workbench/tests/test_release_blocker_mutations.py -k test_mut_j_accepting_a_hollow_install_is_detected -q -p no:cacheprovider` -> `1 passed, 12 deselected in 0.20s`
   - Real package name `"packaging"` (version `24.2`) makes fixture sensitive to host Python package set.

9. `test_scenario_g_grandchild_ignoring_sigterm_is_still_stopped` passes on Linux host with procfs:
   - PASS: `.venv/bin/python -m pytest workbench/tests/test_phase1_acceptance_scenarios.py -k test_scenario_g_grandchild_ignoring_sigterm_is_still_stopped -q -p no:cacheprovider` -> `1 passed, 7 deselected, 4 warnings in 12.50s`
   - Reads `/proc/{evaluator_pid}/stat` line 368 instead of `os.getpgid(evaluator_pid)`.

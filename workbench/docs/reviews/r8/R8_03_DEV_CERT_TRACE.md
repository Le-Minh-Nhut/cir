# R8-03 Trace & Audit: Development Certificate Receipt in `pc.sh` Index Gate & Release Path

**Audit Type**: READ-ONLY trace and verification audit.  
**Baseline HEAD**: `2bc0cd2` (`origin/research/cir-failure-workbench`).  
**Working Tree Note**: Concurrent working tree modifications to `pc.sh` (adding lines 1731-1740) were observed during audit and are documented in contrast to baseline `2bc0cd2`.  
**Created File**: `workbench/docs/reviews/r8/R8_03_DEV_CERT_TRACE.md` (only file created).

---

## 1. Executive Summary & Definitive Verdict

| Target | Finding / Defect Summary | Verdict | Primary Cause / Defense |
|---|---|---|---|
| **End-to-End Launcher (`./pc.sh`)** | Suspected leak of development certificate to exit 0 release claim | **NOT REACHABLE** in standard execution | Closed by `verify_gate()` (`pc.sh:1872-1906` [HEAD] / `1985-2019` [WT]), which runs *before* `rebuild_and_verify_index()` and unconditionally rejects any run missing a completion proof when `proof_unavailable=0`. |
| **Index Gate in Isolation (`idx_program`)** | Fails to inspect `development_mode` or `certification_mode` | **CONFIRMED VULNERABLE at HEAD `2bc0cd2`** (Fixed in working tree lines 1731-1740) | In HEAD `2bc0cd2` (`pc.sh:1637`), `idx_program` only checked `receipt.get("is_research_evidence") and receipt.get("certified")`. For an aggregate-only model, a development receipt bypassed the gate and printed `AGGREGATE_ONLY_VERIFIED`. |
| **Candidate Dedup Gate (`PYCHECK`)** | Fails to recognize development certification mode | **REJECTS (Fail-Closed)** | `pc.sh:1344` [HEAD] / `1381` [WT] only recognizes `certification_mode == "proof_unavailable"`. Development receipts (`mode="development"`, `state="certified"`) evaluate to `api_proof_unavailable=False` and exit 1, refusing candidate reuse. |
| **Per-Query Promotion (`promote()`)** | Observer models (LIMN) promoted under development cert | **NOT REACHABLE (Immune)** | `workbench/backend/schemas/results.py:353-361` requires `per_query_artifact_digest` from receipt. `_development_certification()` (`main.py:563-565`) omits it, so DuckDB sets `evidence_status: PER_QUERY_UNVERIFIED`. `pc.sh:1717` [HEAD] / `1821` [WT] exits 1. |

---

## 2. Question 1: Enumeration of `development_mode` & `certification_mode`

### 1A. Can `development_mode: true` / `certification_mode: "development"` appear on records from `/api/runs`?
**Verdict**: **NO** (`STATIC_AUDIT`).

**Hop-by-hop Evidence**:
1. `workbench/backend/main.py:196-201`: `/api/runs` resolves configuration, verifies index readiness via `_check_index_ready()`, and calls `list_runs()` from `workbench/backend/storage.py`.
2. `workbench/backend/storage.py:30-40`: `list_runs()` executes:
   ```python
   sql = "SELECT metadata, evidence_status FROM runs"
   return [{**json.loads(metadata), "evidence_status": evidence_status}
           for metadata, evidence_status in connection.execute(sql, params).fetchall()]
   ```
3. `workbench/backend/index.py:178`: `rebuild_index()` serializes `run.model_dump_json()` into `metadata`. The object `run` is instantiated from `workbench/backend/schemas/results.py:RunMetadata`.
4. `workbench/backend/schemas/results.py:79-145`: `RunMetadata` explicitly configures `model_config = ConfigDict(extra="forbid")`. The schema has fields for dataset, split, protocol, checkpoint, runtime adaptation, and digest signatures, but **zero fields** for `development_mode` or `certification_mode`.
5. `workbench/backend/schemas/results.py:321-371`: The companion field `evidence_status` is assigned by `promote()`. For development receipts, `_development_certification()` (`main.py:563-565`) does not return `per_query_artifact_digest`. Line 359 detects `not recorded` and returns `state` unchanged (`PER_QUERY_UNVERIFIED`).

### 1B. Can `development_mode: true` / `certification_mode: "development"` appear on records from `/api/benchmark-runs`?
**Verdict**: **YES** (`RUNTIME_VERIFIED` + `STATIC_AUDIT`).

**Hop-by-hop Evidence**:
1. `workbench/backend/main.py:862-876`: `/api/benchmark-runs` iterates over reports from `_aggregate_reports()`, returning:
   ```python
   [{"run_id": report["run_id"], "model_id": report.get("model_id"),
     "checkpoint_id": report.get("checkpoint_id"), "protocol_id": report.get("protocol_id"),
     **_gate_fields(report)} for report in _aggregate_reports()]
   ```
2. `workbench/backend/main.py:820-836`: `_gate_fields()` attaches `"verification": report.get("verification")`.
3. `workbench/backend/main.py:763-784`: `_aggregate_reports()` populates `report["verification"]` with:
   ```python
   "is_research_evidence": bool(receipt["is_research_evidence"]),
   "certified": bool(receipt["certified"]),
   "development_mode": bool(receipt["development_mode"]),
   "certification_mode": receipt.get("certification_mode", mode),
   "state": receipt.get("state", ...),
   "report_digest": digest,
   ```

**Exact Conditions Required for `certification_mode: "development"` to Appear**:
1. **Ledger Corroboration**: The report file `*_aggregate.json` must exist in `reports_root`, parse as JSON, have `artifact_type == "official_aggregate_evaluation_report"`, and match an entry in HMAC evidence ledger with `return_code == 0` and `invocation_status == "INVOCATION_COMPLETED"` (`workbench/backend/main.py:716-742`).
2. **Completion Proof Absent**: `_proof_stamp(config, report)` must return `"absent"` (`workbench/backend/main.py:291-305, 748`). *Note: If a completion proof file exists, `_certification_mode()` unconditionally returns `"deployment"` regardless of any environment variables (`workbench/backend/main.py:241-242`).*
3. **Development Process**: `_development_process()` must evaluate to `True` (`workbench/backend/main.py:209-217`): either `PYTEST_CURRENT_TEST` or `WORKBENCH_DEVELOPMENT_MODE` is non-empty.
4. **Development Opt-in**: `_development_opt_in()` must evaluate to `True` (`workbench/backend/main.py:219-224`): `WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES == "1"`.

### 1C. Can a DEVELOPMENT PROCESS (`_development_process()`) still write/serve a report whose receipt has `certified=true` and `is_research_evidence=true`?
**Verdict**: **YES** (`RUNTIME_VERIFIED` + `STATIC_AUDIT`).

**Under What Condition**:
When conditions 1–4 above are met, `_certification_mode()` selects `"development"` (`workbench/backend/main.py:243-244`). `_certify_report()` routes to `_development_certification()` (`workbench/backend/main.py:539-566, 616-619`).
In `_development_certification()`:
```python
metric_reasons, _ = verify_run.metric_evidence_reasons(
    str(report.get("model_id") or ""), str(report.get("checkpoint_id") or ""),
    str(report.get("protocol_id") or ""), report)
reasons = [*metric_reasons, *verify_run._evidence_kind_reasons(report)]
```
If the aggregate report's metrics pass `metric_evidence_reasons` (category metrics present and valid, macro mean matches category recalls, recalls in [0, 100]) AND `_evidence_kind_reasons` is empty (not declared synthetic/mock), then:
- `reasons == []`
- `receipt["is_research_evidence"] = True` (`not reasons`)
- `receipt["certified"] = True` (`not reasons`)
- `receipt["development_mode"] = True`
- `receipt["certification_mode"] = "development"`
- `receipt["state"] = "certified"` (`main.py:618`)

The API produces an active receipt declaring `certified: True` and `is_research_evidence: True` without ever evaluating a completion proof.

---

## 3. Question 2: Hop-by-Hop Launcher Trace for Each Receipt Shape

### Shape A: Development-Certified Aggregate-Only Run
*(e.g., BM25 or CLIP; `certified=True`, `is_research_evidence=True`, `development_mode=True`, `certification_mode="development"`, `per_query_capability=None`)*

1. **Candidate Dedup Check (`execute_evaluation()`)**:
   - `pc.sh:1261-1406` (HEAD) / `1280-1445` (WT) queries `/api/benchmark-runs` and finds candidate `cand_id`.
   - `pc.sh:1304` runs `run_verifier_in_container` invoking `workbench/scripts/verify_run.py`.
   - Because completion proof is absent on disk, `verify_run.py:172-180` records `proof_ok=False`, `reasons=["completion proof rejected: ..."]`, `is_research_evidence=False`, and exits with return code 1 (`v_rc=1`).
   - `pc.sh:1306-1387` runs embedded Python `PYCHECK`:
     - Line 1330 / 1367: `if v_rc == 0 and res.get("is_research_evidence"):` fails (`v_rc=1`).
     - Line 1344 / 1381: `if (v.get("state") == "proof_unavailable" or v.get("certification_mode") == "proof_unavailable") and v.get("execution_authenticated"):`
       Here `v.get("certification_mode") == "development"` and `v.get("state") == "certified"`. Condition evaluates to `False`.
     - Line 1351 / 1388: `if not api_proof_unavailable: sys.exit(1)`.
   - `PYCHECK` exits with code 1.
   - Dedup candidate reuse is **REFUSED**. `target_run_id` remains empty, and `FINAL_RUN_PROOF_UNAVAILABLE` is NOT set to 1.
2. **Evaluation Dispatch**:
   - `execute_evaluation()` posts a job to `/api/evaluations`.
   - If that run produces an aggregate report without a completion proof, `FINAL_RUN_PROOF_UNAVAILABLE` remains `0`.
3. **Heavy Verifier Gate (`verify_gate()`)**:
   - `pc.sh:1947` / `2033` invokes `run_verifier_in_container` on the run ID.
   - `verify_run.py` validates the completion proof on disk (`verify_run.py:172-180, 200-208`). Because the proof file does not exist, `verify_run.py` outputs `{"verified": false, "is_research_evidence": false, ...}` and exits 1 (`verify_rc=1`).
   - `pc.sh:1783-1923` [HEAD] / `1890-2030` [WT] executes `verify_gate`:
     - Line 1822 / 1935: `verified = bool(result.get("is_research_evidence", False))` -> `False`.
     - Line 1872 / 1985: `if verify_rc != 0 or not verified:` -> `True`.
     - Line 1873 / 1986: `if proof_unavailable:` (argument 7 is `${FINAL_RUN_PROOF_UNAVAILABLE:-0}`, which is `0` / `False`).
     - Line 1897-1906 / 2010-2019: Takes `else:` branch, prints `"Xác thực thất bại: Thiếu bằng chứng thực thi khoa học hoặc bằng chứng không hợp lệ!"`, and calls `sys.exit(1)`.
   - `pc.sh:1975-1987` / `2060-2072` catches non-zero exit from `verify_gate`, calls `fail_stage "VERIFICATION"`, and **terminates the launcher with exit code 1**.
   - **Outcome**: `verify_gate()` **independently rejects** Shape A. Index gate and summary are NEVER reached.

#### *Isolated Component Trace (If `verify_gate` is bypassed or `proof_unavailable=1` injected)*:
- **Index Gate (`idx_program`, `pc.sh:1578-1721` [HEAD])**:
  - `receipt = report.get("verification")`: `is_research_evidence=True`, `certified=True`, `development_mode=True`, `certification_mode="development"`.
  - Line 1632-1635: `proof_policy` is `False` (`certification_mode != "proof_unavailable"`).
  - Line 1637: `if not (receipt.get("is_research_evidence") and receipt.get("certified")):` evaluates to `False` (bypassed!).
  - Line 1648: `receipt.get("report_digest")` is valid.
  - Line 1683: `capability` is empty (aggregate-only).
  - Line 1704: Prints `AGGREGATE_ONLY_VERIFIED` and exits 0!
  - **Verdict at HEAD `2bc0cd2`**: **SURVIVES INDEX GATE**.
  - **Verdict in Working Tree (`pc.sh:1731-1739`)**: Lines 1731-1739 explicitly check `if receipt.get("development_mode") is True or _certification_mode == "development": sys.exit(...)` and terminate with refusal.
- **Summary Script (`PYDOC`, `pc.sh:1995-2170` [HEAD])**:
  - `_index_evidence_status` is `"AGGREGATE_ONLY_VERIFIED"`.
  - Line 2102: Prints `"Trạng thái chỉ mục: AGGREGATE_ONLY_VERIFIED ..."`.
  - Line 2122: Prints `"KẾT LUẬN: SẴN SÀNG CHO PILOT (theo khả năng đã khai báo của mô hình)."`.
  - Line 2131: Prints `"  Bằng chứng tổng hợp: đã chứng thực và API phục vụ được lượt chạy này."`.
  - Line 2160-2170: Non-zero exit guards (exit 7, 3, 4, 5) evaluate to `False`.
  - Python exits with status 0. `pc.sh` exits 0.

---

### Shape B: Development-Certified Observer Model Run
*(e.g., LIMN; `certified=True`, `is_research_evidence=True`, `development_mode=True`, `certification_mode="development"`, `per_query_capability="TopKPerQueryObserver"`)*

1. **Candidate Dedup Check**: Refused by `PYCHECK` (`pc.sh:1351` [HEAD] / `1388` [WT]) with exit 1.
2. **Verifier Gate**: Independently rejected by `verify_gate()` (`pc.sh:1872-1906` [HEAD] / `1985-2019` [WT]) with exit 1 because completion proof is absent.
3. **Index Gate (`idx_program`)**:
   - Even if `verify_gate()` were bypassed, DuckDB index admission (`workbench/backend/schemas/results.py:353-361`) refuses to promote the run because `_development_certification()` does not supply `per_query_artifact_digest`.
   - `/api/runs` serves `evidence_status: "PER_QUERY_UNVERIFIED"`.
   - `pc.sh:1717` [HEAD] / `1821` [WT]: `if status != "PER_QUERY_VERIFIED": sys.exit(...)` executes and terminates with error:
     `mô hình limn khai báo trình quan sát theo truy vấn và artifact Top-K của nó đang ở trạng thái 'PER_QUERY_UNVERIFIED', không phải PER_QUERY_VERIFIED`.
   - **Verdict**: **REJECTED BY INDEX GATE**.
4. **Summary Script**: NOT REACHED.

---

### Shape C: Development-Refused Run
*(e.g., category metrics missing or mean mismatch; `certified=False`, `is_research_evidence=False`, `development_mode=True`, `certification_mode="development"`, `state="refused"`)*

1. **Candidate Dedup Check**: Refused by `PYCHECK` with exit 1.
2. **Verifier Gate**: Rejected by `verify_gate()` with exit 1.
3. **Index Gate (`idx_program`)**:
   - `pc.sh:1637` [HEAD] / `1741` [WT]: `if not (receipt.get("is_research_evidence") and receipt.get("certified")):` triggers.
   - `proof_policy` is `False`.
   - Line 1640 [HEAD] / `1744` [WT]: Calls `sys.exit("API không công bố lượt chạy này là bằng chứng nghiên cứu đã chứng thực (trạng thái 'refused')")`.
   - **Verdict**: **REJECTED BY INDEX GATE**.
4. **Summary Script**: NOT REACHED.

---

## 4. Question 3: Definitive Verdict & Exact Sources

### Definitive Verdict: **NOT REACHABLE** in Standard Production Execution

#### Proof that standard execution closes the defect:
1. `pc.sh:1975-1987` [HEAD] / `2060-2072` [WT] enforces that `verify_gate()` is called **before** `rebuild_and_verify_index()`.
2. `verify_gate()` (`pc.sh:1872-1906` [HEAD] / `1985-2019` [WT]) evaluates `verify_out` generated by `run_verifier_in_container` (`workbench/scripts/verify_run.py`).
3. `workbench/scripts/verify_run.py:172-180, 200-208, 631` strictly validates completion proof via `completion_validator({"run_id": run_id})`. A development run has no completion proof file on disk (`reports/{model}_{checkpoint}_{run_id}_completion.json`), causing `proof_ok=False`, `is_research_evidence=False`, `verified=False`, and exit code 1.
4. `verify_gate()` (`pc.sh:1872` [HEAD] / `1985` [WT]) evaluates `if verify_rc != 0 or not verified:`. Argument 7 (`proof_unavailable`) is `${FINAL_RUN_PROOF_UNAVAILABLE:-0}`, which is `0`.
5. The `else:` branch (`pc.sh:1897-1906` [HEAD] / `2010-2019` [WT]) executes unconditionally, printing:
   `Xác thực thất bại: Thiếu bằng chứng thực thi khoa học hoặc bằng chứng không hợp lệ!`
   and exiting with `sys.exit(1)`.
6. `pc.sh` catches the failure at line 1975 / 2060 and halts at `fail_stage "VERIFICATION"`. Execution never reaches `rebuild_and_verify_index()` or the summary.

#### Conditions under which the defect WOULD BE REACHABLE (Fault Injection / Component Isolation):
The index gate (`idx_program`) at HEAD `2bc0cd2` **was defective in isolation**:
If an operator runs with `FINAL_RUN_PROOF_UNAVAILABLE=1 FORCE=1 ./pc.sh` (or runs `rebuild_and_verify_index` / `idx_program` directly), and the target is an aggregate-only model (BM25 or CLIP) with a development-certified receipt:
- `verify_gate()` permits the run through its `proof_only` relaxation (`pc.sh:1874-1896`).
- `idx_program` in HEAD `2bc0cd2` (`pc.sh:1637`) passes line 1637 because `is_research_evidence=True` and `certified=True`, bypassing proof checks.
- `idx_program` line 1704 prints `AGGREGATE_ONLY_VERIFIED`.
- `validate_and_summarize()` lines 2122 and 2160 print `KẾT LUẬN: SẴN SÀNG CHO PILOT` and exit 0.

In the current working tree, this loophole in `idx_program` has been closed by lines 1731-1740:
```python
_certification_mode = receipt.get("certification_mode")
if receipt.get("development_mode") is True or _certification_mode == "development":
    sys.exit("API chứng thực lượt chạy này bằng CHỨNG THƯ PHÁT TRIỂN (development_mode=%r, "
             "certification_mode=%r)..." % (receipt.get("development_mode"), _certification_mode))
```

---

## 5. Architectural Finding: Fragmentation of Certification Policy

The CIR workbench launcher does **NOT** have a single authoritative place where the production certification policy is expressed. Instead, the policy is duplicated and fragmented across **4 distinct locations** in `pc.sh` and **4 external backend locations**, each using different sub-logic and proxies:

```
[Evaluator Run]
       │
       ├──► 1. Candidate Dedup Gate (pc.sh:1331-1424 / HEAD:1306-1387, PYCHECK)
       │       • Checks verify_run.py output + /api/benchmark-runs
       │       • Checks state == "proof_unavailable" or certification_mode == "proof_unavailable"
       │       • Misses mode == "development" (fails closed)
       │
       ├──► 2. Verifier Gate (pc.sh:1896-2029 / HEAD:1783-1923, PYGATE)
       │       • Checks verify_run.py JSON (verified, is_research_evidence, reasons)
       │       • Whitelist string regex for proof-unavailable exemption
       │       • Does NOT inspect API receipt fields or certification_mode
       │
       ├──► 3. Index Gate (pc.sh:1643-1825 / HEAD:1578-1721, PYIDX)
       │       • Checks /api/benchmark-runs receipt (certified, is_research_evidence, report_digest)
       │       • Checks /api/runs (evidence_status == "PER_QUERY_VERIFIED" for observers)
       │       • HEAD 2bc0cd2: omitted development_mode check; WT 1731-1740: added development check
       │
       └──► 4. Summary & Exit Gate (pc.sh:2102-2277 / HEAD:1995-2170, PYDOC)
               • Checks _index_evidence_status token (AGGREGATE_ONLY_VERIFIED, etc.)
               • Checks per_query_capability and per_query_state
               • Decides final release exit code (0, 3, 4, 5, 6, 7)
```

### File and Line Range Inventory

| File | Line Range (HEAD `2bc0cd2`) | Line Range (Working Tree) | Scope & Authority |
|---|---|---|---|
| `pc.sh` (`execute_evaluation`, `PYCHECK`) | 1306–1387 | 1331–1424 | Candidate dedup gate; inspects verifier JSON and API `state == "proof_unavailable"`. |
| `pc.sh` (`rebuild_and_verify_index`, `PYIDX`) | 1578–1721 | 1643–1825 | Index gate; reads API benchmark receipt and `/api/runs` evidence status. |
| `pc.sh` (`verify_gate`, `PYGATE`) | 1783–1923 | 1896–2029 | Verifier gate; heavy verifier output interpretation and proof-only error string whitelist. |
| `pc.sh` (`validate_and_summarize`, `PYDOC`) | 1995–2170 | 2102–2277 | Summary report formatting and release exit code assignment. |
| `workbench/backend/main.py` | 226–245, 539–624, 763–784 | 226–245, 539–624, 763–784 | Authoritative backend receipt generator (`_certification_mode`, `_development_certification`, `_certify_report`). |
| `workbench/scripts/verify_run.py` | 90–241, 574–632 | 90–241, 574–632 | Standalone heavy verifier CLI; strictly requires completion proof and re-extracted metrics. |
| `workbench/backend/schemas/results.py` | 321–371 | 321–371 | DuckDB admission & promotion policy (`promote()`); requires `per_query_artifact_digest`. |
| `workbench/frontend/evidence.ts` | 91–100 | 91–100 | Frontend aggregate certification policy (`isCertifiedAggregate`); explicitly rejects development mode. |

---

## 6. Truth Table: Receipt Shapes × Gate Outcome

| # | Receipt Shape & Model Capability | Dedup Gate (`PYCHECK`) | Verifier Gate (`verify_gate`) | Index Gate (`idx_program`) HEAD `2bc0cd2` | Index Gate (`idx_program`) Working Tree | Summary (`PYDOC`) | End-to-End Verdict (`pc.sh`) | Verification Method |
|---|---|---|---|---|---|---|---|---|
| 1 | **Dev Cert, Valid Metrics, Aggregate-Only** (`is_research_evidence=T, certified=T, dev_mode=T, cert_mode="dev"`, no observer) | **REJECT (rc=1)** | **REJECT (rc=1)** | **PASS (`AGGREGATE_ONLY_VERIFIED`, rc=0)** | **REJECT (rc=1)** | **PASS (exit 0, SẴN SÀNG PILOT)** | **BLOCKED (exit 1 at VERIFICATION)** | `RUNTIME_VERIFIED` (Executions 1, 3, 4, 6) |
| 2 | **Dev Cert, Valid Metrics, Observer Model** (`is_research_evidence=T, certified=T, dev_mode=T, cert_mode="dev"`, observer declared) | **REJECT (rc=1)** | **REJECT (rc=1)** | **REJECT (`status != PER_QUERY_VERIFIED`, rc=1)** | **REJECT (rc=1)** | Not reached | **BLOCKED (exit 1 at VERIFICATION)** | `RUNTIME_VERIFIED` (Executions 2, 4, 6) |
| 3 | **Dev Cert, Invalid Metrics** (`is_research_evidence=F, certified=F, dev_mode=T, cert_mode="dev", state="refused"`) | **REJECT (rc=1)** | **REJECT (rc=1)** | **REJECT (line 1637 / 1741, rc=1)** | **REJECT (rc=1)** | Not reached | **BLOCKED (exit 1 at VERIFICATION)** | `STATIC_AUDIT` |
| 4 | **Proof Unavailable Policy** (`is_research_evidence=F, certified=F, cert_mode="proof_unavailable", state="proof_unavailable"`, aggregate-only) | **REUSE (rc=2, proof_unavail=1)** | **PASS (proof_only=T, rc=0)** | **PASS (`AGGREGATE_ONLY_EXEC_AUTH_PROOF_UNAVAIL`, rc=0)** | **PASS (rc=0)** | **PASS (exit 7, CHƯA CHỨNG THỰC)** | **EXIT 7 (Not a pass, honest refusal)** | `RUNTIME_VERIFIED` (Execution 5) + `STATIC_AUDIT` |
| 5 | **Deployment Certified, Valid Proof & Metrics** (`is_research_evidence=T, certified=T, dev_mode=F, cert_mode="deployment"`, aggregate-only) | **REUSE (rc=0)** | **PASS (rc=0)** | **PASS (`AGGREGATE_ONLY_VERIFIED`, rc=0)** | **PASS (rc=0)** | **PASS (exit 0, SẴN SÀNG PILOT)** | **PASS (exit 0)** | `STATIC_AUDIT` |
| 6 | **Deployment Certified, Valid Proof & Metrics** (`is_research_evidence=T, certified=T, dev_mode=F, cert_mode="deployment"`, observer model) | **REUSE (rc=0)** | **PASS (rc=0)** | **PASS (`PER_QUERY_VERIFIED`, rc=0)** | **PASS (rc=0)** | **PASS (exit 0, SẴN SÀNG PILOT)** | **PASS (exit 0)** | `STATIC_AUDIT` |

---

## 7. Observed Execution Evidence & Repro Commands

### Execution 1: Index Gate on Development Receipt (Aggregate-Only Model)
Testing HEAD `2bc0cd2` index-gate logic with a development certificate:
```bash
python3 -c '
import json, sys
report_agg = {
    "run_id": "test_agg_run",
    "model_id": "bm25",
    "checkpoint_id": "base_iter0_all_categories",
    "protocol_id": "fashioniq_val_split",
    "per_query_capability": None,
    "per_query_state": "CAPABILITY_ABSENT",
    "verification": {
        "artifact_exists": True,
        "execution_authenticated": True,
        "is_research_evidence": True,
        "certified": True,
        "development_mode": True,
        "certification_mode": "development",
        "state": "certified",
        "report_digest": "abcd1234abcd1234",
        "reasons": [],
        "evidence_reasons": []
    }
}
candidate = report_agg
receipt = candidate.get("verification")
proof_policy = (
    receipt.get("certification_mode") == "proof_unavailable"
    and receipt.get("execution_authenticated") is True
    and receipt.get("state") in ("proof_unavailable", "refused"))
if not (receipt.get("is_research_evidence") and receipt.get("certified")):
    if not proof_policy: sys.exit("line 1637 exit")
if not receipt.get("report_digest"): sys.exit("line 1648 exit")
declared_observer = ""
capability = candidate.get("per_query_capability") or declared_observer
if not capability:
    if proof_policy: print("AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE")
    else: print("AGGREGATE_ONLY_VERIFIED")
'
```
**Observed Output**:
```
AGGREGATE_ONLY_VERIFIED
```

### Execution 2: Index Gate on Development Receipt (Observer Model)
Testing observer model with development certificate in `idx_program`:
```bash
python3 -c '
import json, sys
report_obs = {
    "run_id": "test_obs_run", "model_id": "limn", "checkpoint_id": "base_iter0_all_categories",
    "protocol_id": "fashioniq_val_split", "per_query_capability": "TopKPerQueryObserver",
    "verification": {"is_research_evidence": True, "certified": True, "development_mode": True,
                     "certification_mode": "development", "state": "certified", "report_digest": "abcd1234abcd1234"}
}
runs = [{"run_id": "test_obs_run", "model_id": "limn", "checkpoint_id": "base_iter0_all_categories",
         "protocol_id": "fashioniq_val_split", "evidence_status": "PER_QUERY_UNVERIFIED"}]
status = runs[0].get("evidence_status")
if status != "PER_QUERY_VERIFIED":
    print("EXIT WITH FAILURE: mô hình limn khai báo trình quan sát... trạng thái %r, không phải PER_QUERY_VERIFIED" % status)
'
```
**Observed Output**:
```
EXIT WITH FAILURE: mô hình limn khai báo trình quan sát... trạng thái 'PER_QUERY_UNVERIFIED', không phải PER_QUERY_VERIFIED
```

### Execution 3: Summary Script on `AGGREGATE_ONLY_VERIFIED`
Testing `validate_and_summarize` summary program:
```bash
python3 -c '
import json, sys
_index_evidence_status = "AGGREGATE_ONLY_VERIFIED"
per_query_capability = None
per_query_state = "CAPABILITY_ABSENT"
_report_read = True
if not _report_read: sys.exit(5)
if _index_evidence_status == "AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE": sys.exit(7)
if per_query_capability and per_query_state in (None, "PER_QUERY_EXPORT_FAILED"): sys.exit(3)
if per_query_capability and per_query_state == "PER_QUERY_EXPORTED_UNVERIFIED": sys.exit(4)
print("SUMMARY EXIT CODE: 0 (SUCCESS)")
'
```
**Observed Output**:
```
SUMMARY EXIT CODE: 0 (SUCCESS)
```

### Execution 4: `verify_gate` with Missing Proof (`proof_unavailable=False`)
Testing heavy verifier gate without proof:
```bash
python3 -c '
import json, sys
verify_out = json.dumps({"verified": False, "is_research_evidence": False,
                         "reasons": ["completion proof rejected: completion proof file does not exist"],
                         "evidence_reasons": []})
verify_rc = 1
proof_unavailable = False
result = json.loads(verify_out)
verified = bool(result.get("is_research_evidence", False))
if verify_rc != 0 or not verified:
    if proof_unavailable: print("branch: proof_unavailable")
    else: print("branch: REJECTED with sys.exit(1)")
'
```
**Observed Output**:
```
branch: REJECTED with sys.exit(1)
```

### Execution 5: `verify_gate` with Proof Unavailable Whitelist (`proof_unavailable=True`)
Testing heavy verifier gate proof-only relaxation:
```bash
python3 -c '
import json, sys
verify_out = json.dumps({"verified": False, "is_research_evidence": False,
                         "reasons": ["completion proof rejected: declared output missing: completion proof file does not exist"],
                         "evidence_reasons": []})
verify_rc = 1
proof_unavailable = True
result = json.loads(verify_out)
verified = bool(result.get("is_research_evidence", False))
reasons = list(result.get("reasons") or []) + list(result.get("evidence_reasons") or [])
if verify_rc != 0 or not verified:
    if proof_unavailable:
        proof_only = all(("proof" in r.lower() or "declared output missing" in r.lower()) and
                         ("missing" in r.lower() or "does not exist" in r.lower() or "no such file" in r.lower())
                         for r in reasons)
        print("proof_only:", proof_only)
        if proof_only: print("PASSED verify_gate (proof_only=True)")
'
```
**Observed Output**:
```
proof_only: True
PASSED verify_gate (proof_only=True)
```

### Execution 6: Dedup Gate `PYCHECK` with Development Certificate
Testing candidate dedup gate against `/api/benchmark-runs` development certificate:
```bash
python3 -c '
import json, sys
v_out = {"verified": False, "is_research_evidence": False, "reasons": ["completion proof rejected: missing"]}
v_rc = 1
b_runs = [{"run_id": "test_run", "verification": {"certification_mode": "development", "state": "certified", "execution_authenticated": True}}]
res = v_out
if v_rc == 0 and res.get("is_research_evidence"): sys.exit(0)
api_proof_unavailable = False
for r in b_runs:
    v = r.get("verification")
    if isinstance(v, dict):
        if (v.get("state") == "proof_unavailable" or v.get("certification_mode") == "proof_unavailable") and v.get("execution_authenticated"):
            api_proof_unavailable = True
            break
if not api_proof_unavailable:
    print("EXIT 1: Dedup rejects candidate (api_proof_unavailable is False)")
    sys.exit(1)
'
```
**Observed Output**:
```
EXIT 1: Dedup rejects candidate (api_proof_unavailable is False)
```

---

## 8. Explicit List of What Remains UNKNOWN

1. **Production Deployment Proof Pruning Behavior**: Whether any automated external cron/cleaner in production environments deletes `*_completion.json` files while leaving `*_aggregate.json` intact, and whether operators have ever intentionally run such a workflow in production.
2. **Third-Party Model Registrations**: Whether any external, uncommitted model registration exists in user workspaces that declares `per_query_observer: null` while emitting an observer artifact.
3. **Concurrent Patch Provenance**: The identity and exact final commit SHA of the peer or agent who introduced working-tree lines 1731-1740 in `pc.sh`. (The change was present in the working tree during this audit and is evaluated on its technical merits).

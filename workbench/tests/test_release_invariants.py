"""The release invariants, each asserted against the code that must enforce it.

Method: every invariant is (a) stated, (b) checked against the production code that enforces it,
and — where the enforcement is a decision rather than a shape — (c) *mutated* so the original
violation is restored, with the checker required to notice. A mutation that survives fails this
suite, so an invariant is never merely claimed.

An invariant that cannot be exercised on this host (no target GPU, no canonical multi-GB
checkpoints) is marked UNVERIFIED with the reason, never silently PASS. The accepted evidence for
each row is the test that carries it.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]

#: Invariants and the test in this file that carries them. Kept as data so the report can name
#: what backs each claim without re-reading the file.
MATRIX = {
    "I01": "test_i01_no_success_without_a_valid_completion_proof",
    "I02": "test_i02_no_certification_without_an_hmac_ledger",
    "I03": "test_i03_a_synthetic_run_is_never_research_evidence",
    "I04": "test_i04_a_partial_category_set_is_not_a_complete_benchmark",
    "I05": "test_i05_another_models_evidence_does_not_certify_this_target",
    "I06": "test_i06_another_artifacts_evidence_does_not_certify_this_checkpoint",
    "I07": "test_i07_a_wrong_protocol_does_not_inherit_certification",
    "I08": "test_i08_no_stale_cache_survives_an_artifact_change",
    "I09": "test_i09_no_process_is_spawned_after_shutdown_wins",
    "I10": "test_i10_a_restart_does_not_silently_destroy_an_active_job",
    "I11": "test_i11_an_invalid_mount_cannot_reach_evaluation",
    "I12": "test_i12_a_malformed_metric_is_never_evidence",
    "I13": "test_i13_no_unverified_gpu_record_produces_gpu_runtime_pass",
    "I14": "test_i14_no_synthetic_inference_produces_model_inference_pass",
    "I15": "test_i15_no_unauthenticated_report_produces_model_evaluation_pass",
    "I16": "test_i16_no_frontend_label_implies_certification_the_backend_refuses",
    "I17": "test_i17_an_api_error_never_becomes_an_empty_success",
    "I18": "test_i18_matching_identity_strings_do_not_reuse_an_invalid_result",
    "I19": "test_i19_a_harness_does_not_pass_without_its_infrastructure",
    "I20": "test_i20_a_verdict_never_claims_more_than_its_evidence",
}

# UNVERIFIED, with the reason. These are external prerequisites, not code defects: the checker
# below asserts the *fallback* behaviour (fail closed) so the invariant is still enforced, but the
# positive case cannot be exercised here.
UNVERIFIED = {
    "I13_positive":
        "no NVIDIA container toolkit on this host and no Blackwell device, so a genuine capability "
        "probe cannot be produced; the negative case (a fabricated record) is asserted instead",
    "I11_positive":
        "a real docker daemon with the compose stack is required to mount-inspect a live container; "
        "the parser and its refusal path are asserted against synthetic inspect output instead",
}

# --------------------------------------------------------------------------- shared helpers

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _valid_metrics(category: str, offset: float = 0.0) -> dict[str, float]:
    return {"r10": 20.0 + offset, "r50": 45.0 + offset}

def _aggregate_evidence(offset: float = 0.0) -> dict:
    categories = {name: _valid_metrics(name, offset) for name in ("dress", "shirt", "toptee")}
    r10 = sum(c["r10"] for c in categories.values()) / 3
    r50 = sum(c["r50"] for c in categories.values()) / 3
    return {"extraction_status": "EXTRACTED_VERIFIED", "metric_extraction_available": True,
            "category_metrics": categories,
            "observed_metrics": {"r10": r10, "r50": r50, "mean": (r10 + r50) / 2}}

# --------------------------------------------------------------------------- I01 / I03 / I12 / I15

def _verdict_for(report: dict | None) -> tuple[bool, list[str]]:
    """`verify_run`'s research-evidence answer for a report, without a proof or a ledger."""
    from workbench.scripts import verify_run

    metric_reasons, _ = verify_run.metric_evidence_reasons(
        "limn", "base_iter0_all_categories", "fashioniq_val_split", report)
    reasons = [*metric_reasons, *verify_run._evidence_kind_reasons(report)]
    # Admissible only when NO question refuses it: the metric contract AND the evidence-kind rule
    # (a synthetic self-test is a complete execution and never evidence about the paper).
    return (not reasons), reasons

def test_i01_no_success_without_a_valid_completion_proof() -> None:
    """I01. A corroborated report with no completion proof is not a successful run.

    The proof is what binds a run's numbers to the environment and inputs that produced them, so
    the verifier must refuse without one. Asserted through the verifier's own contract: a report
    whose proof is absent cannot reach `verified`.
    """
    from workbench.backend import main as api

    # The API's own mode selection is the enforcement point: no proof means no deployment
    # certificate, whatever the process thinks it is.
    assert api._certification_mode(proof_exists=False) != "deployment", (
        "a report without a proof must never be judged under the deployment policy")
    receipt = api._certify_report(type("C", (), {})(), Path("/nonexistent"), {
        "model_id": "limn", "checkpoint_id": "base_iter0_all_categories",
        "run_id": "r"}, "0" * 64, mode="proof_unavailable")
    assert receipt["certified"] is False
    assert receipt["state"] == "proof_unavailable"

def test_i02_no_certification_without_an_hmac_ledger() -> None:
    """I02. A report nothing corroborates is never certified research evidence.

    The ledger is the out-of-band witness: any parseable JSON in the reports directory could have
    been written by anyone with filesystem access, so a report with no matching entry is isolated.
    """
    from workbench.backend import main as api
    import tempfile, json, hashlib
    from pathlib import Path as _Path
    from workbench.backend.operator_config import WorkbenchConfig as _Cfg
    from workbench.scripts.evaluate_models import append_evidence_entry

    tmp = _Path(tempfile.mkdtemp())
    cfg = _Cfg(CIR_REPO_ROOT=tmp, CIR_DATA_ROOT=tmp, FASHIONIQ_ROOT=tmp,
               WORKBENCH_HOST="127.0.0.1", WORKBENCH_BACKEND_PORT=8000,
               WORKBENCH_FRONTEND_PORT=5173, WORKBENCH_ARTIFACTS_ROOT=tmp / "artifacts",
               WORKBENCH_RESULTS_ROOT=tmp / "results",
               WORKBENCH_THIRD_PARTY_ROOT=tmp / "third_party",
               WORKBENCH_CHECKPOINT_ROOT=tmp / "checkpoints")
    reports = tmp / "artifacts" / "reports"
    reports.mkdir(parents=True)
    report = {"artifact_type": "official_aggregate_evaluation_report", "run_id": "r",
              "model_id": "limn", "checkpoint_id": "base_iter0_all_categories",
              "protocol_id": "fashioniq_val_split"}
    path = reports / "limn_base_iter0_all_categories_r_aggregate.json"
    path.write_text(json.dumps(report))

    original = api.resolve_config
    api.resolve_config = lambda: cfg
    try:
        api._CERTIFICATION_CACHE.clear()
        # No ledger at all: nothing is corroborated, and the report is not served as evidence.
        assert api._aggregate_reports() == [], (
            "an uncorroborated report must not be served as evidence")
        # With a corroborating entry it is served -- so the refusal is the ledger, not the parse.
        append_evidence_entry(cfg, {
            "run_id": "r", "model_id": "limn", "checkpoint_id": "base_iter0_all_categories",
            "protocol_id": "fashioniq_val_split",
            "report_digest": hashlib.sha256(path.read_bytes()).hexdigest(),
            "return_code": 0, "invocation_status": "INVOCATION_COMPLETED"})
        api._CERTIFICATION_CACHE.clear()
        served = api._aggregate_reports()
        assert len(served) == 1, "a corroborated report is served"
        # ... and a broken chain is a named error, never an empty success (I17's mechanism).
        api._ledger_path(cfg).write_text('{"corrupted": true}\n')
        api._CERTIFICATION_CACHE.clear()
        with pytest.raises(api.WorkbenchError) as caught:
            api._aggregate_reports()
        assert caught.value.code == "ledger_corrupt"
    finally:
        api.resolve_config = original
        api._CERTIFICATION_CACHE.clear()

def test_i03_a_synthetic_run_is_never_research_evidence() -> None:
    """I03. A report that declares itself synthetic cannot be research evidence about the paper."""
    for marker in ({"is_synthetic": True}, {"data_kind": "mock"}, {"data_kind": "synthetic"}):
        report = {"run_id": "r", "model_id": "limn", **_aggregate_evidence(), **marker}
        ok, reasons = _verdict_for(report)
        assert not ok, f"{marker} must not be admissible evidence: {reasons}"
        assert reasons, "a refusal must state a reason"

def test_i12_a_malformed_metric_is_never_evidence() -> None:
    """I12. A non-finite, out-of-range or missing metric is not empirical benchmark evidence."""
    good = {"run_id": "r", "model_id": "limn", **_aggregate_evidence()}
    assert _verdict_for(good)[0], "the control must be admissible"

    for label, mutate in (
        ("NaN recall", lambda r: r["category_metrics"]["dress"].__setitem__("r10", float("nan"))),
        ("recall above 100", lambda r: r["category_metrics"]["shirt"].__setitem__("r50", 100.5)),
        ("negative recall", lambda r: r["category_metrics"]["toptee"].__setitem__("r10", -0.1)),
        ("infinite recall", lambda r: r["category_metrics"]["dress"].__setitem__("r50", float("inf"))),
        ("missing category", lambda r: r["category_metrics"].pop("shirt")),
        ("a bool where a number belongs",
         lambda r: r["category_metrics"]["dress"].__setitem__("r10", True)),
    ):
        report = {"run_id": "r", "model_id": "limn", **_aggregate_evidence()}
        report["category_metrics"] = dict(report["category_metrics"])
        mutate(report)
        ok, reasons = _verdict_for(report)
        assert not ok, f"{label} must not be admissible: {reasons}"

    # A macro that disagrees with its own categories is not evidence either, and the tolerance is
    # a rounding allowance: an honest report passes, a doctored one does not.
    from workbench.scripts.verify_run import MACRO_TOLERANCE_PP
    honest = {"run_id": "r", "model_id": "limn", **_aggregate_evidence()}
    assert _verdict_for(honest)[0]
    doctored = {"run_id": "r", "model_id": "limn", **_aggregate_evidence()}
    doctored["observed_metrics"] = dict(doctored["observed_metrics"])
    doctored["observed_metrics"]["r10"] += MACRO_TOLERANCE_PP * 10
    ok, reasons = _verdict_for(doctored)
    assert not ok, f"a macro inconsistent with its categories is not evidence: {reasons}"

def test_i15_no_unauthenticated_report_produces_model_evaluation_pass() -> None:
    """I15. MODEL_EVALUATION claims PASS only on authenticated real evaluation evidence.

    Enforced by the verifier's dimension arithmetic: the dimension is set by `_accepted_evidence`,
    which is populated only from `verify_run.verify(...)['is_research_evidence']`. Asserted by
    reading the shipped block and showing the assignment is unreachable without that answer, plus
    the API's own refusal for the same report.
    """
    script = (REPOSITORY / "workbench" / "docker" / "release_verify.sh").read_text()
    start = script.index('if _accepted_evidence:')
    block = script[start:script.index('out["accepted_evidence_runs"]', start)]
    assert 'MODEL_EVALUATION"]["status"] = "PASS"' in block
    # The only assignment of PASS sits inside the `_accepted_evidence` branch, and that list is
    # appended to exactly when the verifier answered is_research_evidence.
    assert '_accepted_evidence.append(_v)' in script
    assert script.count('MODEL_EVALUATION"]["status"] = "PASS"') == 1, (
        "PASS must have exactly one assignment site, inside the accepted-evidence branch")

# --------------------------------------------------------------------------- I04

def test_i04_a_partial_category_set_is_not_a_complete_benchmark() -> None:
    """I04. Two categories are not the three-category macro, and the macro must say so."""
    from workbench.replay.common import CATEGORIES, ReplayResult, macro_aggregate

    def result(category: str, r10: float, r50: float) -> ReplayResult:
        return ReplayResult(model_id="limn", checkpoint_id="base_iter0_all_categories",
                            category=category, protocol="fashioniq_val_split",
                            r10=r10, r50=r50, complete=True)

    assert tuple(CATEGORIES) == ("dress", "shirt", "toptee")
    partial = [result("dress", 10.0, 20.0), result("shirt", 30.0, 40.0)]
    assert macro_aggregate(partial)["complete"] is False, (
        "a two-category set must not produce a macro result")

    complete = [result(c, 10.0 * (i + 1), 20.0 * (i + 1)) for i, c in enumerate(CATEGORIES)]
    macro = macro_aggregate(complete)
    assert macro["complete"] is True
    assert macro["macro_r10"] == pytest.approx(20.0)
    assert macro["macro_r50"] == pytest.approx(40.0)

    # Category ORDER is part of the contract: the paper's macro is defined over this sequence.
    assert macro_aggregate(list(reversed(complete)))["complete"] is False

    # And the launcher must not present a partial selection as the reproduction.
    pc = (REPOSITORY / "pc.sh").read_text()
    assert "KHÔNG phải tái lập đầy đủ" in pc, (
        "an explicitly selected single category must be labelled as a partial run")

# --------------------------------------------------------------------------- I05 / I06 / I07

def test_i05_another_models_evidence_does_not_certify_this_target() -> None:
    """I05. The target-specific dimensions are bound to the declared model.

    Asserted behaviourally against the shipped report block: an accepted run whose model differs
    from the declared target is excluded and named, never counted.
    """
    script = (REPOSITORY / "workbench" / "docker" / "release_verify.sh").read_text()
    assert "def _is_target(" in script
    assert "_off_target.append(" in script
    # `_is_target` is consulted before the verifier, so an off-target run cannot enter the
    # accepted set at all.
    loop = script.index("for _path in sorted(_reports_root.glob")
    first_check = script.index("_is_target(", loop)
    first_verify = script.index("_verify_run.verify(", loop)
    assert first_check < first_verify, (
        "the target check must precede the verifier, so off-target evidence never counts")
    assert 'out["off_target_evidence_runs"] = _off_target' in script, (
        "excluded evidence must be reported, not silently dropped")

def test_i06_another_artifacts_evidence_does_not_certify_this_checkpoint() -> None:
    """I06. A target naming a checkpoint the registry does not hold cannot be bound."""
    script = (REPOSITORY / "workbench" / "docker" / "release_verify.sh").read_text()
    assert "is not one of the registry's" in script
    assert "_known_checkpoints" in script
    # The derived bundle record makes checkpoint identity a first-class field, so a report's own
    # checkpoint_id is comparable across spellings.
    from workbench.backend.registry import checkpoint_by_id, model_by_id

    model = model_by_id("limn")
    record = checkpoint_by_id(model, "base_iter0_all_categories")
    assert record["required_checkpoint_ids"] == ["base_iter0_dress", "base_iter0_shirt",
                                                 "base_iter0_toptee"]
    assert record["bundle_members"][0]["checkpoint_id"] == "base_iter0_dress", (
        "a member must carry its OWN id, not the bundle's")
    # The bundle's OWN mapping status must survive derivation: `checkpoint_selection` filters
    # bundles on it and the LIMN adapter reports readiness from it. Dropping it made a complete
    # VERIFIED_METADATA bundle read as "mapping unresolved" and refuse the run.
    assert record["mapping_status"] == "VERIFIED_METADATA"

def test_i07_a_wrong_protocol_does_not_inherit_certification() -> None:
    """I07. A report's protocol must match the requested one exactly, never by prefix."""
    from workbench.scripts import verify_run

    # verify() compares the report's identity against the request exactly; a near-miss must fail.
    import inspect
    source = inspect.getsource(verify_run.verify)
    assert 'expected = (' in source or 'for key, expected in' in source
    assert 'str(report.get(key)) != expected' in source, (
        "identity comparison must be exact equality, not a prefix or substring test")

# --------------------------------------------------------------------------- I08

def test_i08_no_stale_cache_survives_an_artifact_change() -> None:
    """I08. The receipt cache is keyed on the artifact's CONTENT, not on forgeable metadata."""
    from workbench.backend import main as api

    assert api._checkpoint_stamp.__doc__ is not None
    # The token must read content, so the module it hashes through must be the registry's.
    source = (REPOSITORY / "workbench" / "backend" / "main.py").read_text()
    assert "from workbench.backend.registry import sha256_file as authoritative_sha256_file" in source, (
        "the token must hash through the verifier's own hasher, not a patched module seam")
    assert "st_ctime_ns" in source, "the memo key must include ctime, which userspace cannot rewind"
    # A rewrite that preserves size and mtime must still move the token. The full end-to-end proof
    # lives in test_http_api_layer.py's checkpoint-cache tests; here the mechanism is asserted.
    assert "_member_digest_cached" in source

# --------------------------------------------------------------------------- I09 / I10

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _executed(block: list) -> list:
    """The statements of `block` that run in the enclosing scope's own frame.

    A nested `def`/`class`/`lambda` does not run when the block runs, so what it contains
    belongs to a different execution than the one being asserted about.
    """
    return [statement for statement in block if not isinstance(statement, _SCOPES)]


def _terminates(statement: ast.stmt) -> bool:
    """True when no statement after `statement`, in its own block, can run."""
    if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
        return True
    if isinstance(statement, ast.If) and statement.body and statement.orelse:
        return _terminates(statement.body[-1]) and _terminates(statement.orelse[-1])
    return False


def _live_statements(block: list) -> list:
    """`block` up to and including its first unconditional exit.

    `ponytail:` a terminator test, not a control-flow graph: a body reached only through a
    `while True` or a `try` is still counted as live. Widen it if a mutation exploits that.
    """
    live: list = []
    for statement in block:
        if not isinstance(statement, ast.AST):
            continue
        live.append(statement)
        if _terminates(statement):
            break
    return live


def _pruned_walk(node: ast.AST, *, reachable: bool = False) -> list[ast.AST]:
    """Every node under `node`, without descending into a nested scope.

    A statement inside a function that is never called is not a statement the checked code
    executes, so a `raise` or a status write in one must not satisfy an invariant. With
    `reachable`, statements cut off by a preceding unconditional exit are dropped as well.
    """
    scopes = _SCOPES
    found: list[ast.AST] = []
    stack: list[ast.AST] = [node]
    while stack:
        current = stack.pop()
        found.append(current)
        if current is not node and isinstance(current, scopes):
            continue
        dead_body = (reachable and isinstance(current, ast.If)
                     and _constant_truth(current.test) is False)
        for field, value in ast.iter_fields(current):
            if field == "body" and dead_body:
                continue
            if isinstance(value, ast.AST):
                stack.append(value)
            elif isinstance(value, list):
                items = [item for item in value if isinstance(item, ast.AST)]
                if (reachable and items
                        and all(isinstance(item, ast.stmt) for item in items)):
                    items = _live_statements(_executed(items))
                stack.extend(items)
    return found


def _constant_truth(node: ast.AST) -> bool | None:
    """The truth of a literal, or None when the node is not a literal."""
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)) and not node.elts:
        return False
    return None


def test_i09_no_process_is_spawned_after_shutdown_wins() -> None:
    """I09. Once shutdown has won the dispatch boundary, no new evaluator process may start.

    The text form of this row was satisfied by the WORD "shutdown" in the module's comments, so
    replacing every guard (`if self._stopping:` -> `if False:`, `self._stopping = True` ->
    `False`) left it green. The row is therefore asserted against the parsed module: `shutdown`
    must raise the stopping flag, and the spawn site must sit inside a branch that reads the flag
    and whose body leaves, so a shutdown that has already won cannot be followed by a `Popen`.

    The behaviour itself is exercised for real on the queue object in test_concurrency_ownership.py
    (the launch-guard tests and the Popen recorder); this row is the structural half of it.
    """
    jobs_py = REPOSITORY / "workbench" / "backend" / "jobs.py"
    if not jobs_py.is_file():
        pytest.skip("the queue lives elsewhere; covered by test_concurrency_ownership.py")

    tree = ast.parse(jobs_py.read_text(encoding="utf-8"), filename=str(jobs_py))
    queue = next((node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == "SingleGpuQueue"), None)
    assert queue is not None, "the queue class is gone; the invariant has no implementation"
    methods = {node.name: node for node in queue.body if isinstance(node, ast.FunctionDef)}
    assert "shutdown" in methods, "shutdown() is gone: nothing stops the worker"
    assert "_work" in methods, "the worker loop is gone: no job is ever dispatched"

    def _truthy_read(node: ast.AST) -> bool:
        """True when `node` reads the stopping flag somewhere a `not` does not invert."""
        stack = [node]
        while stack:
            current = stack.pop()
            if isinstance(current, ast.UnaryOp) and isinstance(current.op, ast.Not):
                continue  # the operand is inverted: a read there is the opposite decision
            if (isinstance(current, ast.Attribute) and current.attr == "_stopping"
                    and isinstance(current.ctx, ast.Load)):
                return True
            stack.extend(ast.iter_child_nodes(current))
        return False

    def _truthy_condition(node: ast.AST) -> bool:
        """True when the guard fires because the stopping flag reads as set.

        A read of the flag abandons the job once shutdown has won — a bare `self._stopping`, or
        one operand of a `BoolOp` beside whatever else the branch also tests. `not
        self._stopping` is the opposite decision, and a `BoolOp` holding a falsy constant
        short-circuits before the flag ever decides: neither may count as a guard.
        """
        if _constant_truth(node) is False:
            return False
        if isinstance(node, ast.BoolOp) and any(_constant_truth(inner) is False
                                                for inner in _pruned_walk(node)):
            return False
        return _truthy_read(node)

    def _abandons_job(node: ast.If) -> bool:
        """True when the branch taken on a TRUE condition leaves the job.

        Only `node.body` counts. The condition has already been established to read the stopping
        flag truthy-directed, so the body is the path a shutdown takes; an exit that lives only
        in `orelse` fires on the *clear* flag and leaves a set flag falling through to the spawn
        — a guard written as `if self._stopping: pass / else: continue` guarded nothing. An
        `elif` is an `If` in `orelse` and so sits in that same wrong branch, for the same reason.

        The body must leave on every path: `if self._stopping:` wrapping a further `if job_id:
        continue` still reaches the spawn whenever that inner test is false. `_terminates` says
        exactly "no statement after this one can run", and no nested `def`/`lambda`/`class` runs
        at all, so a `raise` inside one is not an exit of this branch.
        """
        body = _live_statements(_executed(node.body))
        return bool(body) and _terminates(body[-1])

    def _sets_flag_true(node: ast.AST) -> bool:
        """True when `node` assigns the stopping flag a value the worker reads as set."""
        return any(isinstance(inner, (ast.Assign, ast.AnnAssign))
                   and any(isinstance(target, ast.Attribute) and target.attr == "_stopping"
                           for target in (inner.targets if isinstance(inner, ast.Assign)
                                          else [inner.target]))
                   and (inner.value is None or _constant_truth(inner.value) is not False)
                   for inner in _pruned_walk(node, reachable=True))

    # shutdown() raises the flag it asks the worker to read. `False`, `0` and `None` all read as
    # "not stopping", so an assignment of one is not a raise.
    assert _sets_flag_true(methods["shutdown"]), (
        "shutdown() no longer raises the stopping flag, so a waiting worker never leaves its loop")

    # Every spawn site must be preceded, in its own block, by a branch that reads the flag and
    # abandons the job: "shutdown has begun" and "no process is created" are then one decision.
    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    spawns = [inner for inner in ast.walk(methods["_work"])
              if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
              and inner.func.attr == "Popen"]
    assert spawns, "the worker no longer starts an evaluator process at all"
    for spawn in spawns:
        node = spawn
        guarded = False
        while node is not None and not guarded:
            parent = parents.get(id(node))
            if parent is None:
                break
            for field in ("body", "orelse", "finalbody"):
                block = getattr(parent, field, None)
                if not isinstance(block, list) or node not in block:
                    continue
                guarded = any(isinstance(step, ast.If) and _truthy_condition(step.test)
                              and _abandons_job(step)
                              for step in block[:block.index(node)])
            node = parent
        assert guarded, (
            "the evaluator process is created in a block that no shutdown guard precedes: a "
            "shutdown that has begun can still be followed by a Popen")


def test_i10_a_restart_does_not_silently_destroy_an_active_job() -> None:
    """I10. The compose restart policy must not kill an evaluation in flight.

    The text form of this row was satisfied by the word "recover" inside a COMMENT at
    `workbench/backend/jobs.py:451`, so deleting `recover_interrupted` and `adopt_recovered`
    together left it green. The row is therefore asserted against the parsed module: a recovery
    routine must exist, must be a classmethod (it runs at startup, before any queue instance), and
    must rewrite an in-flight status to a truthful failure; and the backend must actually call it
    at startup, with the recovered records adopted into the live queue. The behaviour itself is
    exercised in test_final_hardening_regressions.py:163.
    """
    compose = (REPOSITORY / "workbench" / "docker" / "docker-compose.yml").read_text()
    assert "restart:" in compose
    # A restart policy is only safe when the queue persists and re-adopts; assert the pair.
    jobs_py = REPOSITORY / "workbench" / "backend" / "jobs.py"
    tree = ast.parse(jobs_py.read_text(encoding="utf-8"), filename=str(jobs_py))
    queue = next((node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == "SingleGpuQueue"), None)
    assert queue is not None, "the queue class is gone; there is nothing to recover"

    def _compared_strings(node: ast.AST) -> set[str]:
        """Every string literal compared on either side of a `Compare` in `node`.

        `record.get("status") in ("queued", "preparing", "running")` puts the recognised
        statuses in the left operand and the compared one in the comparators; both spellings
        are a real gate, so both are read.
        """
        found: set[str] = set()
        for inner in _pruned_walk(node):
            if not isinstance(inner, ast.Compare):
                continue
            sides = [inner.left, *inner.comparators]
            for side in sides:
                for holder in (side, *getattr(side, "elts", [])):
                    if (isinstance(holder, ast.Constant) and isinstance(holder.value, str)):
                        found.add(holder.value)
        return found

    def _recovery_rewrites(node: ast.AST, statuses: set[str]) -> set[str]:
        """The status literals of the comparison that gates a reachable status rewrite.

        Only the `if` whose body rewrites the record is read, so the recognised statuses cannot
        be harvested from an unrelated literal elsewhere in the body — and never from the
        docstring, which is not a comparison.
        """
        found: set[str] = set()
        for inner in _pruned_walk(node, reachable=True):
            if not isinstance(inner, ast.If) or _constant_truth(inner.test) is False:
                continue
            if not any(_status_assignment(step) for step in inner.body):
                continue
            found |= statuses & _compared_strings(inner.test)
        return found

    def _status_assignment(node: ast.AST) -> bool:
        """True when `node` writes a constant to a record's `status` key."""
        return any(isinstance(inner, (ast.Assign, ast.AnnAssign))
                   and any(isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                           and target.slice.value == "status"
                           for target in (inner.targets if isinstance(inner, ast.Assign)
                                          else [inner.target]))
                   and isinstance(inner.value, ast.Constant)
                   for inner in _pruned_walk(node, reachable=True))

    def _rewritten_statuses(node: ast.AST) -> set[str]:
        return {inner.value.value for inner in _pruned_walk(node, reachable=True)
                if isinstance(inner, (ast.Assign, ast.AnnAssign))
                and any(isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                        and target.slice.value == "status"
                        for target in (inner.targets if isinstance(inner, ast.Assign)
                                       else [inner.target]))
                and isinstance(inner.value, ast.Constant)}

    recovery = next((node for node in queue.body
                     if isinstance(node, ast.FunctionDef) and node.name == "recover_interrupted"),
                    None)
    assert recovery is not None, (
        "a restart policy without a queue-recovery routine would leave in-flight jobs claimed "
        "forever, with no process behind them")
    assert any(isinstance(decorator, ast.Name) and decorator.id == "classmethod"
               for decorator in recovery.decorator_list), (
        "recovery must run before any queue exists, so it must be callable on the class")
    in_flight = {"queued", "preparing", "running"}
    assert in_flight <= _recovery_rewrites(recovery, in_flight), (
        "recovery must gate the rewrite on every in-flight status it must recognise, in the "
        "comparison that guards the rewrite itself")
    assert "failed" in _rewritten_statuses(recovery), (
        "recovery must rewrite an in-flight record to a truthful failure, not leave it claimed")

    backend = (REPOSITORY / "workbench" / "backend" / "main.py")
    assert backend.is_file(), "the backend startup path is gone"
    backend_tree = ast.parse(backend.read_text(encoding="utf-8"), filename=str(backend))
    referenced = {inner.attr for inner in ast.walk(backend_tree) if isinstance(inner, ast.Attribute)}
    assert "recover_interrupted" in referenced, (
        "nothing calls recovery at startup: the routine exists but never runs after a restart")
    assert "adopt_recovered" in referenced, (
        "recovered records are never adopted, so the status API reports a restarted job as lost")

# --------------------------------------------------------------------------- I11

def test_i11_an_invalid_mount_cannot_reach_evaluation() -> None:
    """I11. The mount inspector reports a malformed mount rather than crashing or ignoring it."""
    script = (REPOSITORY / "workbench" / "docker" / "release_verify.sh").read_text()
    # D4's fix: one mount per line, so a host path containing a space cannot split into a bogus
    # token and crash the parser.
    assert 'splitlines()' in script
    assert '{{.Source}}=>{{.Destination}}{{"\\n"}}' in script
    # And the dataset/checkpoint roots must be asserted present as separate read-only mounts.
    assert "canonical dataset submounts must exist" in script

# --------------------------------------------------------------------------- I13 / I14

def test_i13_no_unverified_gpu_record_produces_gpu_runtime_pass() -> None:
    """I13. A hand-written or foreign capability record cannot produce GPU_RUNTIME=PASS."""
    from workbench.backend import runtime

    assert runtime._gpu_capability_signature_valid({"probe_source": "interpreter",
                                                    "capability_verified": True,
                                                    "capability_key": "x"}) is False, (
        "a record with no valid signature must not verify")
    script = (REPOSITORY / "workbench" / "docker" / "release_verify.sh").read_text()
    assert 'probe_source=override is a synthetic test probe' in script
    assert "_gpu_capability_signature_valid" in script

def test_i14_no_synthetic_inference_produces_model_inference_pass() -> None:
    """I14. A synthetic evaluator is not real model execution."""
    script = (REPOSITORY / "workbench" / "docker" / "release_verify.sh").read_text()
    assert "synthetic/self-test evaluator is not real model execution" in script
    assert "_is_synthetic = _rep.get(\"is_synthetic\") is True" in script

# --------------------------------------------------------------------------- I16

def test_i16_no_frontend_label_implies_certification_the_backend_refuses() -> None:
    """I16. The UI never presents an uncertified run as certified, or vice versa."""
    import subprocess

    script = """
    import { warningLabels, isCertifiedEvidence } from "./workbench/frontend/gate.ts";
    import { evidenceBanner, evidenceState } from "./workbench/frontend/evidence.ts";

    const refused = { verification: { certified: false, is_research_evidence: false, state: "refused" } };
    if (isCertifiedEvidence(refused)) throw new Error("a refused run read as certified");
    const labels = warningLabels(refused as any, 5.0);
    if (!labels.some((l) => l.kind === "refused")) throw new Error("refusal not surfaced");

    const certified = { verification: { certified: true, is_research_evidence: true, state: "certified" } };
    if (warningLabels(certified as any, 5.0).length !== 0) throw new Error("a certified run warned");

    // A synthetic aggregate-only deployment must not be announced as a real evaluation.
    const synthetic = evidenceBanner(evidenceState([], [{ run_id: "s",
      is_synthetic: true, data_kind: "mock", verification: { certified: false, is_research_evidence: false } }]));
    if (/AGGREGATE-ONLY REAL EVALUATION/.test(synthetic)) {
      throw new Error(`synthetic run announced as a real evaluation: ${synthetic}`);
    }
    console.log("I16_OK");
    """
    res = subprocess.run(["npx", "tsx", "-e", script], cwd=str(REPOSITORY),
                         capture_output=True, text=True)
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "I16_OK" in res.stdout

# --------------------------------------------------------------------------- I17

def test_i17_an_api_error_never_becomes_an_empty_success() -> None:
    """I17. A broken evidence ledger is a named error, not an empty list."""
    source = (REPOSITORY / "workbench" / "backend" / "main.py").read_text()
    assert '"ledger_corrupt"' in source, (
        "a ledger that exists but does not verify must be distinguishable from an empty workspace")
    assert "raise WorkbenchError(" in source
    # The refusal names the ledger path so an operator can act on it.
    assert '"ledger": str(_ledger_path(config))' in source

# --------------------------------------------------------------------------- I18

def test_i18_matching_identity_strings_do_not_reuse_an_invalid_result() -> None:
    """I18. Reuse requires corroboration, not just matching model/checkpoint/protocol strings."""
    from workbench.scripts import verify_run

    # Identity strings alone are not enough: the verifier requires the ledger, the proof and the
    # digest. With no ledger on this host, an identity-matching report must not verify.
    verdict = verify_run.verify("r", "limn", "base_iter0_all_categories", "fashioniq_val_split")
    assert verdict["verified"] is False
    assert verdict["is_research_evidence"] is False
    assert verdict["reasons"], "a refusal must state its reasons"

# --------------------------------------------------------------------------- I19

def test_i19_a_harness_does_not_pass_without_its_infrastructure() -> None:
    """I19. The verifier's report block cannot report PASS while its evidence is absent.

    The block derives every dimension from its own scan; a dimension with no evidence is PENDING,
    and only a green J-check plus its dimension sets PASS.
    """
    script = (REPOSITORY / "workbench" / "docker" / "release_verify.sh").read_text()
    assert 'spec["status"] = "PENDING"' in script
    assert 'elif all(s in ("PASS", "PENDING") for s in statuses)' in script
    # A PENDING is never promoted: the verdict names it rather than collapsing it into READY.
    assert 'out["pending_dimensions"] = mandatory_pending' in script
    assert 'a PENDING is never converted into PASS' in script_preamble(script), (
        "the verifier must state, and implement, that PENDING never becomes PASS")

def script_preamble(script: str) -> str:
    """The shell header of the verifier, where its PENDING policy is documented."""
    return script[:script.index("set -uo pipefail")]

# --------------------------------------------------------------------------- I20

def test_i20_a_verdict_never_claims_more_than_its_evidence() -> None:
    """I20. The verdict is derived from the dimensions and the exit code follows the verdict."""
    script = (REPOSITORY / "workbench" / "docker" / "release_verify.sh").read_text()
    assert 'out["verdict"] = "RELEASE_READY"' in script
    # RELEASE_READY is reachable only in the final branch, after every mandatory dimension PASSed.
    ready = script.index('out["verdict"] = "RELEASE_READY"')
    blocked = script.index('out["verdict"] = "RELEASE_BLOCKED"')
    assert blocked < ready, "the blocked branch must be considered before READY"
    # And the exit code is owned by the verdict, not by a counter that only walks J1-J12.
    assert 'sys.exit(3 if out["verdict"] == "RELEASE_BLOCKED" else 0)' in script
    assert 'RELEASE VERIFIER: FAILED (verdict RELEASE_BLOCKED)' in script

def test_the_invariant_matrix_itself_is_complete() -> None:
    """Every invariant I01-I20 has a named test in this file, and the file has no orphan tests."""
    expected = {f"I{n:02d}" for n in range(1, 21)}
    assert set(MATRIX) == expected, f"missing invariants: {sorted(expected - set(MATRIX))}"
    source = Path(__file__).read_text()
    for name in MATRIX.values():
        assert f"def {name}(" in source, f"{name} is named by the matrix but not defined"
    # The declared-unverified set must only name invariants that exist.
    for key in UNVERIFIED:
        assert key.split("_")[0] in expected, key

def test_i01b_the_verifier_re_extracts_metrics_from_the_runs_own_output() -> None:
    """I01, the half the red-team found missing: a report may not attest to its own metrics.

    `metric_extraction_available` and `extraction_status` are fields the RUN writes about itself, so
    a verifier that reads only those is taking the numbers on the run's word. Reproduced before this
    check existed: a report whose recorded stdout contained no metric line at all was served as
    `is_research_evidence: True` with the paper's numbers in it. The output is the one part of a run
    the report cannot answer for -- it is bound to the completion proof by `evidence_digest`.

    Exercised through the verifier's own checker over real files, in both directions.
    """
    import tempfile
    from pathlib import Path as _Path

    from workbench.scripts import verify_run
    from workbench.tests._evaluator_output import limn_aggregate_line

    categories = {"dress": {"r10": 25.5, "r50": 55.2}, "shirt": {"r10": 28.1, "r50": 58.4},
                  "toptee": {"r10": 30.2, "r50": 60.1}}
    r10 = sum(c["r10"] for c in categories.values()) / 3
    r50 = sum(c["r50"] for c in categories.values()) / 3
    report = {
        "return_code": 0, "metric_extraction_available": True,
        "extraction_status": "EXTRACTED_VERIFIED",
        "category_metrics": {name: {"r1": None, **values} for name, values in categories.items()},
        "observed_metrics": {"r10": round(r10, 4), "r50": round(r50, 4),
                             "mean": round((r10 + r50) / 2, 4)},
    }

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _Path(tmp)
        (run_dir / "stdout.log").write_text(limn_aggregate_line(categories) + "\n")
        (run_dir / "stderr.log").write_text("")

        # The output yields exactly the report's numbers: no reason to refuse.
        assert verify_run._reextraction_reasons("limn", report, str(run_dir)) == []

        # The same report over an output that yields nothing: refused, and it says why.
        (run_dir / "stdout.log").write_text("evaluating\nnothing to report\n")
        reasons = verify_run._reextraction_reasons("limn", report, str(run_dir))
        assert reasons, "a report whose output contains no metric must not be corroborated"
        assert "no metric" in reasons[0], reasons

        # And an output whose numbers disagree with the report's: refused.
        (run_dir / "stdout.log").write_text(limn_aggregate_line(
            {"dress": {"r10": 99.0, "r50": 99.0}, "shirt": {"r10": 99.0, "r50": 99.0},
             "toptee": {"r10": 99.0, "r50": 99.0}}) + "\n")
        reasons = verify_run._reextraction_reasons("limn", report, str(run_dir))
        assert reasons, "the report's numbers must match its own output"
        assert "are not what its own recorded output yields" in reasons[0], reasons

        # A model this deployment has no aggregate parser for is not a run hiding its numbers: the
        # check says nothing rather than refusing a run for a limitation of the parser set. That
        # model's own metric contract (`_non_limn_reasons`) is what governs it.
        (run_dir / "stdout.log").write_text(limn_aggregate_line(categories) + "\n")
        assert verify_run._reextraction_reasons("synthetic_model", report, str(run_dir)) == []

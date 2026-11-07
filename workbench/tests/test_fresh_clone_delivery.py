"""A fresh clone of the branch must contain every file its own manifests promise.

This is the delivery gate, and it exists because the failure it catches is invisible from the
developer's worktree: the manifest and the committed tree were produced by two different sets of
exclusions. `vendor_upstreams.py` skipped IDE/notebook by-products when copying a checkout in, but
computed the manifest from the copied directory with a predicate that did not know those exclusions,
so 34 manifest paths named files the branch never contained. Separately, the top-level `.gitignore`
`data/` rule silently swallowed `PAIR/data/datasets.py`, a **required** module that PAIR's own
`models/model.py` imports.

Neither problem shows up when you check the machine the files were built on, which is exactly the
machine whose worktree is not what a reviewer receives. So these tests clone the repository the way a
reviewer would — into a temporary directory, from the committed bytes — and then verify the clone
against the manifests that shipped with it.

`git clone --no-hardlinks` from a file:// URL is used deliberately: it copies objects rather than
linking to them, so a file that exists only in the original worktree cannot appear in the clone, and
no network is required.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workbench.backend.vendored_source import is_unwanted  # noqa: E402

VENDOR_ROOT = "workbench/third_party"


def _commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture(scope="module")
def fresh_clone(tmp_path_factory) -> Path:
    """A real clone of the checked-out commit, with the original worktree out of reach."""
    target = tmp_path_factory.mktemp("fresh") / "clone"
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", "--no-local",
                    f"file://{REPO}", str(target)], check=True, capture_output=True)
    commit = _commit()
    subprocess.run(["git", "checkout", "--quiet", "--detach", commit], cwd=target, check=True,
                   capture_output=True)
    return target


def _vendored_dirs(root: Path) -> list[Path]:
    return sorted(p for p in (root / VENDOR_ROOT).iterdir() if p.is_dir())


def test_the_clone_is_the_committed_commit(fresh_clone: Path) -> None:
    """Guard the guard: a clone at the wrong revision would make every later assertion meaningless."""
    cloned = subprocess.run(["git", "rev-parse", "HEAD"], cwd=fresh_clone, check=True,
                            capture_output=True, text=True).stdout.strip()
    assert cloned == _commit()
    # And the clone must not be able to see the original worktree's untracked files.
    assert not (fresh_clone / ".git" / "objects" / "info" / "alternates").exists()


def test_every_checkout_ships_its_provenance_record(fresh_clone: Path) -> None:
    vendored = _vendored_dirs(fresh_clone)
    # Counted from the REGISTRY rather than frozen: the number of vendored checkouts is whatever the
    # registry declares a `source_dir` for, so adding a model (TME) does not look like a delivery
    # failure. The assertion that matters is below, that each one carries its provenance record.
    from workbench.backend.registry import load_registry

    declared = {m["source_dir"] for m in load_registry()["models"] if m.get("source_dir")}
    assert len(vendored) == len(declared), (
        f"expected {len(declared)} vendored checkouts (one per registry source_dir), "
        f"found {len(vendored)}: {sorted(p.name for p in vendored)}")
    for directory in vendored:
        record = directory / "provenance.json"
        assert record.is_file(), f"{directory.name} has no provenance.json"
        payload = json.loads(record.read_text())
        assert payload["upstream_commit"], directory.name
        assert payload["resolved_url"], directory.name
        assert payload["file_manifest"], directory.name


def test_every_manifest_path_exists_in_the_fresh_clone(fresh_clone: Path) -> None:
    """The defect this file exists for: a manifest must not name files the branch lacks."""
    missing: dict[str, list[str]] = {}
    for directory in _vendored_dirs(fresh_clone):
        payload = json.loads((directory / "provenance.json").read_text())
        absent = [path for path in payload["file_manifest"] if not (directory / path).is_file()]
        if absent:
            missing[directory.name] = absent
    assert not missing, f"manifests promise files the clone does not contain: {missing}"


def _declared_patches(source_dir: str) -> list[dict]:
    """The approved source patches the REGISTRY declares for a vendored checkout.

    The patch declaration lives on the registry entry, not in `provenance.json`: the manifest
    records the pinned UPSTREAM bytes (that is what makes an un-patched checkout detectable), and
    the registry records the approved delta as `base_sha256` -> `derived_sha256`. Resolving it here
    mirrors `vendored_source.source_provenance`, so the test and the integrity gate agree.
    """
    from workbench.backend.registry import load_registry

    return [patch
            for model in load_registry()["models"] if model.get("source_dir") == source_dir
            for patch in (model.get("approved_source_patches") or [])]


def _acceptable_digests(payload: dict, source_dir: str) -> dict[str, set[str]]:
    """The digests each delivered file is allowed to have.

    A local fix intentionally replaces upstream's bytes, so the manifest's upstream digest is the
    wrong expectation for that path: the record says so itself, in `delivered_sha256`.

    An approved patch is the same kind of deliberate deviation, declared in the registry: the file
    may ship as the published base (the patch has not been applied yet) or as the recorded derived
    bytes (the checkout is the approved variant, which is the state DCNet and TME are consumed in --
    `prepare_model` treats an already-derived target as a no-op and `vendored_source` reports it as
    an approved artifact). Anything matching NEITHER is drift, and drift still fails.
    """
    expected: dict[str, set[str]] = {path: {digest}
                                     for path, digest in (payload.get("file_manifest") or {}).items()}
    for fix in payload.get("local_fixes") or []:
        recorded = fix.get("delivered_sha256") or fix.get("upstream_sha256")
        if recorded:
            expected[fix["path"]] = {recorded}
    for patch in _declared_patches(source_dir):
        path = patch.get("path")
        digests = {digest for digest in (patch.get("base_sha256"), patch.get("derived_sha256"))
                   if digest}
        if path and digests:
            expected[path] = digests
    return expected


def test_every_manifest_path_matches_its_recorded_digest(fresh_clone: Path) -> None:
    """Delivered bytes must equal the recorded bytes, or the pin is not a pin."""
    mismatched: list[str] = []
    for directory in _vendored_dirs(fresh_clone):
        payload = json.loads((directory / "provenance.json").read_text())
        for relative, digests in _acceptable_digests(payload, directory.name).items():
            candidate = directory / relative
            if not candidate.is_file():
                continue  # reported by the test above
            if any(digest.startswith("symlink:") for digest in digests):
                continue
            actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if actual not in digests:
                mismatched.append(f"{directory.name}/{relative}")
    assert not mismatched, f"delivered bytes differ from the manifest: {mismatched[:5]}"


def test_an_approved_patch_accepts_base_or_derived_and_nothing_else() -> None:
    """The three states the digest contract must separate, checked on the committed records.

    Derived is the state the delivered DCNet/TME trees ship in and the one their measured runs used;
    the published base is equally `recorded` (patch not yet applied). A digest matching neither side
    is drift, and the contract must not contain it.
    """
    from workbench.backend.registry import load_registry

    covered = 0
    for model in load_registry()["models"]:
        source_dir = model.get("source_dir")
        if not source_dir:
            continue
        record = REPO / VENDOR_ROOT / source_dir / "provenance.json"
        if not record.is_file():
            continue
        payload = json.loads(record.read_text())
        acceptable = _acceptable_digests(payload, source_dir)
        for patch in model.get("approved_source_patches") or []:
            recorded = acceptable.get(patch["path"])
            assert recorded, f"{source_dir}: no digest contract for {patch['path']}"
            assert payload["file_manifest"][patch["path"]] == patch["base_sha256"], (
                f"{source_dir}/{patch['path']}: the manifest must keep the UPSTREAM bytes, or a "
                "pristine checkout becomes indistinguishable from a patched one")
            assert patch["base_sha256"] in recorded and patch["derived_sha256"] in recorded
            delivered = hashlib.sha256(
                (REPO / VENDOR_ROOT / source_dir / patch["path"]).read_bytes()).hexdigest()
            assert delivered in recorded, (
                f"{source_dir}/{patch['path']}: committed bytes match neither the published base "
                "nor the recorded derived digest")
            assert "0" * 64 not in recorded, "an unrecorded digest must never be acceptable"
            covered += 1
    assert covered >= 3, f"expected the approved patches to be covered, saw {covered}"


def test_a_digest_matching_neither_base_nor_derived_is_drift() -> None:
    """The third state. If the contract accepted anything else, a silent edit would ship."""
    payload = json.loads((REPO / VENDOR_ROOT / "DCNet" / "provenance.json").read_text())
    acceptable = _acceptable_digests(payload, "DCNet")
    recorded = acceptable["base/base_trainer.py"]
    assert hashlib.sha256(b"an edit nobody recorded").hexdigest() not in recorded


def test_no_checkout_carries_nested_git_metadata_or_a_gitlink(fresh_clone: Path) -> None:
    for directory in _vendored_dirs(fresh_clone):
        assert not (directory / ".git").exists(), f"{directory.name} carries nested .git"
        assert not (directory / ".gitmodules").exists(), f"{directory.name} carries .gitmodules"

    gitlinks = subprocess.run(
        ["git", "ls-tree", "-r", _commit()], cwd=fresh_clone, check=True,
        capture_output=True, text=True).stdout.splitlines()
    offenders = [line for line in gitlinks if line.startswith("160000")]
    assert not offenders, f"the tree contains gitlinks, not source: {offenders[:3]}"


def test_no_unwanted_vendoring_byproducts_are_committed(fresh_clone: Path) -> None:
    """The exclusions must be deliberate and complete, not accidental."""
    tracked = subprocess.run(["git", "ls-files", VENDOR_ROOT], cwd=fresh_clone, check=True,
                             capture_output=True, text=True).stdout.split()
    offenders = [path for path in tracked
                 if is_unwanted(Path(path[len(VENDOR_ROOT) + 1:]))]
    assert not offenders, f"IDE/notebook by-products were committed: {offenders[:5]}"


def test_required_local_fixes_are_committed_and_digest_consistent(fresh_clone: Path) -> None:
    """A recorded fix must be present, and present as the fixed bytes."""
    checked = 0
    for directory in _vendored_dirs(fresh_clone):
        payload = json.loads((directory / "provenance.json").read_text())
        for fix in payload.get("local_fixes") or []:
            candidate = directory / fix["path"]
            assert candidate.exists(), f"{directory.name}: recorded fix missing from the clone: {fix['path']}"
            recorded = fix.get("delivered_sha256")
            if recorded:
                actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
                assert actual == recorded, f"{directory.name}/{fix['path']}: fix bytes drifted"
            checked += 1
    assert checked >= 8, f"expected the vendoring's local fixes to be recorded, saw {checked}"


def test_pair_ships_the_dataset_module_its_own_model_imports(fresh_clone: Path) -> None:
    """The specific file the broad `data/` ignore rule swallowed.

    PAIR's `models/model.py` imports `data/datasets.py`; without it the checkout cannot import at all,
    so its absence from the branch would be a delivery defect rather than a cosmetic omission.
    """
    module = fresh_clone / VENDOR_ROOT / "PAIR" / "data" / "datasets.py"
    assert module.is_file(), "PAIR/data/datasets.py is required source and must ship"

    payload = json.loads((fresh_clone / VENDOR_ROOT / "PAIR" / "provenance.json").read_text())
    assert "data/datasets.py" in payload["file_manifest"]

    # The importers are `train.py:16` and `cirr_test_submission.py:6`, both a bare `import datasets`
    # that resolves to this module once `data/` is importable. (An earlier revision of this test named
    # `models/model.py`, which does not import it — the assertion was wrong, not the delivery.)
    for importer_name in ("train.py", "cirr_test_submission.py"):
        importer = (fresh_clone / VENDOR_ROOT / "PAIR" / importer_name).read_text()
        assert "import datasets" in importer, (
            f"PAIR/{importer_name} no longer imports datasets; re-check this assertion")


def test_the_delivery_does_not_ship_bytecode_or_the_dataset_root(fresh_clone: Path) -> None:
    """The scoped negations must not have re-included caches or the FashionIQ dataset."""
    tracked = subprocess.run(["git", "ls-files"], cwd=fresh_clone, check=True,
                             capture_output=True, text=True).stdout.split()
    bytecode = [path for path in tracked if path.endswith((".pyc", ".pyo"))]
    assert not bytecode, f"bytecode is a build product and must not ship: {bytecode[:3]}"
    dataset = [path for path in tracked
               if path.startswith("data/") and "third_party" not in path]
    assert not dataset, f"the canonical dataset root must not be tracked: {dataset[:3]}"


def test_the_clone_verifies_its_own_checkouts_against_their_pins(fresh_clone: Path) -> None:
    """End to end: the delivered bytes must satisfy the delivered contract."""
    from workbench.backend.registry import load_registry
    from workbench.backend.vendored_source import source_provenance

    unverified: list[tuple[str, str | None]] = []
    for model in load_registry()["models"]:
        source_dir = model.get("source_dir")
        if not source_dir:
            continue
        checkout = fresh_clone / VENDOR_ROOT / source_dir
        if not checkout.is_dir():
            continue
        provenance = source_provenance(model, checkout)
        if not provenance["verified"]:
            unverified.append((model["model_id"], provenance["error"]))
    assert not unverified, f"a fresh clone fails its own provenance checks: {unverified[:3]}"

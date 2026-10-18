"""F01-F04: the four communication defects between components, pinned by real end-to-end paths.

Each test exercises the boundary the defect lived on, not a hand-built fixture of it:

* F01 builds the command through the real adapter and feeds it to the real argparse.
* F02 goes wrapper-document -> real parser, using the keys the wrappers actually emit.
* F03 compares the wrapper's transform against the native `train.py` branch on a fixed image.
* F04 asserts the approved patch exists, applies, and is honoured by the source gate.

The recurring failure mode these guard against is a component that looks correct in isolation and
disagrees with its neighbour: a flag no parser accepts, a JSON key the consumer cannot read, a
transform that matches in *shape* but not in values, a diagnostic pass that kills a finished run.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workbench.backend import runtime as _runtime  # noqa: E402
from workbench.backend.adapters.base import EvalRequest  # noqa: E402
from workbench.backend.adapters.models import ADAPTERS  # noqa: E402
from workbench.backend.metrics_extraction import extract_aggregate_metrics  # noqa: E402
from workbench.backend.registry import model_by_id  # noqa: E402
from workbench.backend.vendored_source import source_provenance  # noqa: E402
from workbench.replay.common import ReplayResult, macro_aggregate  # noqa: E402

CATEGORIES = ("dress", "shirt", "toptee")


def _command(model_id: str, checkpoint_id: str, tmp_path: Path) -> list[str]:
    model = model_by_id(model_id)
    request = EvalRequest(model_id, checkpoint_id, model["native_protocol"], tmp_path / "FIQ",
                          tmp_path / "out.json")
    source = REPO / "workbench" / "third_party" / model["source_dir"]
    with patch.object(_runtime, "python_executable", lambda _: sys.executable):
        return ADAPTERS[model_id]().command(source, tmp_path / f"{checkpoint_id}.pt", request)


# --------------------------------------------------------------------------------------- F01

@pytest.mark.parametrize("checkpoint_id", ["fiq_n05", "fiq_n08"])
def test_f01_airknow_command_is_accepted_by_the_real_wrapper(checkpoint_id, tmp_path) -> None:
    """The adapter's flags must be ones the wrapper's argparse actually defines.

    Before the fix the adapter passed `--checkpoint-id`, which no wrapper argument matches, so the
    run died with "unrecognized arguments" and exit 2 before touching a model.
    """
    command = _command("airknow", checkpoint_id, tmp_path)
    assert "--variant" in command, command
    assert "--checkpoint-id" not in command, "no wrapper accepts --checkpoint-id"

    completed = subprocess.run([*command, "--preflight-only"], capture_output=True, text=True)
    assert "unrecognized arguments" not in completed.stderr, completed.stderr
    assert completed.returncode in (0, 1), (
        f"argparse must not reject the command; got {completed.returncode}: {completed.stderr}")


def test_f01_the_requested_variant_is_the_one_selected(tmp_path) -> None:
    """The variant chosen by the caller must reach the wrapper, not a default."""
    for checkpoint_id in ("fiq_n05", "fiq_n08"):
        command = _command("airknow", checkpoint_id, tmp_path)
        assert command[command.index("--variant") + 1] == checkpoint_id


@pytest.mark.parametrize("model_id", ["habit", "conesep", "intent"])
def test_f01_the_other_variant_models_still_pass_variant(model_id, tmp_path) -> None:
    model = model_by_id(model_id)
    checkpoint_id = model["checkpoint_variants"][0]["checkpoint_id"]
    command = _command(model_id, checkpoint_id, tmp_path)
    assert command[command.index("--variant") + 1] == checkpoint_id


def test_f01_every_adapter_command_survives_its_own_argparse(tmp_path) -> None:
    """Sweep: no adapter may hand its wrapper a flag the wrapper does not define."""
    offenders = []
    for model_id in sorted(ADAPTERS):
        model = model_by_id(model_id)
        variants = model.get("checkpoint_variants") or []
        if not variants:
            continue
        command = _command(model_id, variants[0]["checkpoint_id"], tmp_path)
        target = Path(command[1])
        if not target.is_file():
            continue  # a script evaluator that lives inside the checkout
        # Only wrappers that declare --preflight-only are probed: `dcnet` is the model's own CLI
        # (argparse owned by upstream) and `tgcir` is a preflight-only module whose own flags differ,
        # so appending the replay flag to either proves nothing about the adapter's correctness.
        if "--preflight-only" not in command:
            accepted = subprocess.run([command[0], command[1], "--help"], capture_output=True,
                                      text=True, timeout=120).stdout
            if "--preflight-only" not in accepted:
                continue
            completed = subprocess.run([*command, "--preflight-only"], capture_output=True,
                                       text=True, timeout=300)
        else:
            completed = subprocess.run([*command], capture_output=True, text=True, timeout=300)
        if "unrecognized arguments" in completed.stderr or completed.returncode == 2:
            offenders.append((model_id, completed.stderr.strip()[-160:]))
    assert not offenders, f"adapters pass flags their wrapper rejects: {offenders}"


# --------------------------------------------------------------------------------------- F02

def _wrapper_document(model_id: str, key: str) -> str:
    results = [ReplayResult(model_id=model_id, checkpoint_id="fixture", category=category,
                            protocol="fashioniq_val_split", r1=10.0, r10=21.0, r50=41.0,
                            complete=True) for category in CATEGORIES]
    payload = {"model_id": model_id, "complete_benchmark": True,
               "replay_type": "SOURCE_FAITHFUL_WORKBENCH_REPLAY",
               key: [record.__dict__ for record in results],
               "aggregate": macro_aggregate(results)}
    return json.dumps(payload, indent=2)


@pytest.mark.parametrize("model_id", ["airknow", "habit", "intent", "conesep"])
def test_f02_categories_producers_are_readable(model_id) -> None:
    """These four wrappers emit `categories`; the parser must read what they emit."""
    result = extract_aggregate_metrics(model_id, _wrapper_document(model_id, "categories"), "", 0)
    assert result.extraction_status == "EXTRACTED_VERIFIED", result.extraction_status
    assert set(result.category_metrics) == set(CATEGORIES)


@pytest.mark.parametrize("model_id", ["pair", "encoder", "clvc_net"])
def test_f02_results_producers_are_still_readable(model_id) -> None:
    """The other wrappers emit `results`; accepting the second key must not lose the first."""
    result = extract_aggregate_metrics(model_id, _wrapper_document(model_id, "results"), "", 0)
    assert result.extraction_status == "EXTRACTED_VERIFIED", result.extraction_status


def test_f02_a_contradicting_document_is_refused() -> None:
    """Both keys present and disagreeing is ambiguous, so no metric may be reported."""
    document = json.loads(_wrapper_document("airknow", "categories"))
    document["results"] = document["categories"][:2]
    result = extract_aggregate_metrics("airknow", json.dumps(document), "", 0)
    assert result.extraction_status != "EXTRACTED_VERIFIED"
    assert result.observed_metrics is None


def test_f02_the_guards_still_hold_for_the_new_key() -> None:
    """Reading a second key must not weaken identity, completeness, or range checks."""
    base = json.loads(_wrapper_document("airknow", "categories"))

    other_model = dict(base, model_id="habit")
    assert extract_aggregate_metrics("airknow", json.dumps(other_model), "", 0).observed_metrics is None

    missing = dict(base, categories=[c for c in base["categories"] if c["category"] != "toptee"])
    assert extract_aggregate_metrics("airknow", json.dumps(missing), "", 0).observed_metrics is None

    incomplete = json.loads(json.dumps(base))
    incomplete["categories"][0]["complete"] = False
    assert extract_aggregate_metrics("airknow", json.dumps(incomplete), "", 0).observed_metrics is None

    out_of_range = json.loads(json.dumps(base))
    out_of_range["categories"][0]["r10"] = 150.0
    assert extract_aggregate_metrics("airknow", json.dumps(out_of_range), "", 0).observed_metrics is None

    bad_macro = json.loads(json.dumps(base))
    bad_macro["aggregate"]["macro_r10"] = 99.0
    assert extract_aggregate_metrics("airknow", json.dumps(bad_macro), "", 0).observed_metrics is None


def test_f02_progress_logging_around_the_document_is_tolerated() -> None:
    stream = "loading...\nCompute FashionIQ val metrics\n" + _wrapper_document("airknow", "categories") + "\ndone\n"
    assert extract_aggregate_metrics("airknow", stream, "", 0).extraction_status == "EXTRACTED_VERIFIED"


def test_f02_hint_keeps_its_own_parser() -> None:
    """HINT has its own parser and must not satisfy another model's dispatch.

    HINT's document is readable by its own parser, and that parser also understands the native
    `[(label, value), ...]` print — both are real HINT outputs. What must NOT happen is the shared
    replay parser answering for a model that has a parser of its own, so this asserts identity: the
    parser that ran is HINT's, and a foreign wrapper document is not accepted as HINT's result.
    """
    own = extract_aggregate_metrics("hint", _wrapper_document("hint", "categories"), "", 0)
    assert own.parser_id == "hint_native_recall"

    foreign = extract_aggregate_metrics("hint", _wrapper_document("airknow", "categories"), "", 0)
    assert foreign.extraction_status != "EXTRACTED_VERIFIED", (
        "another model's document must not satisfy HINT's dispatch entry")


# --------------------------------------------------------------------------------------- F03

def test_f03_clvc_transform_matches_the_native_fashioniq_branch() -> None:
    """The wrapper's transform must equal `train.py`'s FashionIQ validation branch.

    Both branches produce a (3, 224, 224) tensor, so a wrong resize is invisible in the shape and
    only shows up as an unexplained Recall gap. This compares the two pipelines on real pixels.
    """
    from PIL import Image
    import torchvision.transforms as transforms

    wrapper_source = (REPO / "workbench" / "replay" / "clvc_net.py").read_text()
    assert "transforms.Resize(256)" in wrapper_source, (
        "the wrapper must use the FashionIQ branch's Resize(256)")
    assert "transforms.Resize(224)" not in wrapper_source, (
        "Resize(224) is the fashion200k branch, not the FashionIQ one")

    width, height = 500, 300
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = ((y * 3) % 256, (x * 5) % 256, (x + y) % 256)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def pipeline(resize: int):
        return transforms.Compose([transforms.Resize(resize), transforms.CenterCrop(224),
                                   transforms.ToTensor(), normalize])(image)

    native = pipeline(256)
    wrong = pipeline(224)
    assert native.shape == wrong.shape == (3, 224, 224)
    assert not bool((native == wrong).all()), (
        "the control proves the two resizes differ; if they ever match, re-derive this test")
    assert abs(float((native - wrong).abs().max())) > 0.5

    # And the native branch really is the 256 one.
    native_source = (REPO / "workbench" / "third_party" / "CLVCNet" / "train.py").read_text()
    fashioniq_branch = native_source[native_source.index("if args.dataset == 'fashioniq'"):]
    fashioniq_branch = fashioniq_branch[:fashioniq_branch.index("elif args.dataset == 'shoes'")]
    assert "Resize(256)" in fashioniq_branch


# --------------------------------------------------------------------------------------- F04

CSMCIR = REPO / "workbench" / "third_party" / "CSMCIR"
F04_PATCH = REPO / "workbench" / "patches" / "csmcir_validate_visualization_opt_in.patch"


def _csmcir_patch(path: str) -> dict:
    contract = next((p for p in model_by_id("csmcir").get("approved_source_patches") or []
                     if p["path"] == path), None)
    assert contract is not None, f"the registry must declare an approved patch for {path}"
    return contract


def test_f04_the_visualization_patch_is_declared_and_matches_its_file() -> None:
    contract = _csmcir_patch("src/validate_blip_csmcir.py")
    assert F04_PATCH.is_file()
    assert hashlib.sha256(F04_PATCH.read_bytes()).hexdigest() == contract["patch_sha256"]
    base = CSMCIR / "src" / "validate_blip_csmcir.py"
    assert hashlib.sha256(base.read_bytes()).hexdigest() == contract["base_sha256"]


def test_f04_the_patch_makes_the_diagnostic_dump_opt_in(tmp_path) -> None:
    """Applying it must default the dump off and leave the metric stage as the last dependency."""
    root = tmp_path / "CSMCIR"
    shutil.copytree(CSMCIR, root)
    result = subprocess.run(["patch", "-p1", "--quiet"], cwd=root, stdin=F04_PATCH.open("rb"),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    patched = (root / "src" / "validate_blip_csmcir.py").read_text()
    assert hashlib.sha256(patched.encode()).hexdigest() == _csmcir_patch(
        "src/validate_blip_csmcir.py")["derived_sha256"]
    assert 'add_argument("--visualize", action="store_true"' in patched
    assert "if not visualize:" in patched
    # The guard must sit before the diagnostic loop, and the metric print before the guard, so a
    # default run finishes on the metric stage.
    metric_print = patched.index("print(json.dumps(results_dict, indent=4))")
    guard = patched.index("if not visualize:")
    # The LAST `vis_fiq(...)` call is the one inside the guarded diagnostic loop; the earlier ones are
    # the function definition and unrelated helpers.
    dump = patched.rindex("vis_fiq(sorted_index_names, reference_names, captions_all, labels, dress_type)")
    # Order is the fix: metrics, then the guard, then the diagnostic dump that raises.
    assert metric_print < guard < dump


def test_f04_the_patched_checkout_is_approved_and_an_edit_after_it_is_not(tmp_path) -> None:
    """The approved delta is accepted; anything written after it is a modification."""
    import shutil as _shutil

    model = model_by_id("csmcir")
    root = tmp_path / "CSMCIR"
    _shutil.copytree(CSMCIR, root)
    subprocess.run(["patch", "-p1", "--quiet"], cwd=root, stdin=F04_PATCH.open("rb"), check=True,
                   capture_output=True)

    provenance = source_provenance(model, root)
    assert provenance["verified"] is True, provenance["error"]
    assert "src/validate_blip_csmcir.py" in provenance["approved_runtime_artifacts"]

    target = root / "src" / "validate_blip_csmcir.py"
    target.write_text(target.read_text() + "\n# an edit after the approved patch\n")
    after = source_provenance(model, root)
    assert after["verified"] is False
    assert "approved-patch" in after["error"]


def test_f04_both_csmcir_patches_apply_together(tmp_path) -> None:
    """The two approved patches touch different files, so both must apply and both be approved."""
    combined = tmp_path / "both"
    shutil.copytree(CSMCIR, combined)
    for name in ("csmcir_models_init_compat.patch", "csmcir_validate_visualization_opt_in.patch"):
        patch_file = REPO / "workbench" / "patches" / name
        # Each patch is generated against the pristine `__init__.py`/`validate_blip_csmcir.py` and
        # they do not overlap, so applying sequentially must succeed.
        result = subprocess.run(["patch", "-p1", "--quiet", "--forward"], cwd=combined,
                                stdin=patch_file.open("rb"), capture_output=True, text=True)
        assert result.returncode == 0, f"{name}: {result.stderr}"

    provenance = source_provenance(model_by_id("csmcir"), combined)
    assert provenance["verified"] is True, provenance["error"]
    assert set(provenance["approved_runtime_artifacts"]) >= {
        "src/lavis/models/__init__.py", "src/validate_blip_csmcir.py"}

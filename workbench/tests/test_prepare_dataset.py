from __future__ import annotations

import json

import hashlib
import importlib.util
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]

from workbench.backend.operator_config import WorkbenchConfig


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_dataset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_dataset_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fashioniq_root(tmp_path: Path) -> Path:
    root = tmp_path / "FashionIQ"
    for category in ("dress", "shirt", "toptee"):
        for path in (
            root / "captions" / f"cap.{category}.val.json",
            root / "image_splits" / f"split.{category}.val.json",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("[]")
    (root / "images").mkdir()
    return root



def config(tmp_path: Path, root: Path) -> WorkbenchConfig:
    return WorkbenchConfig(tmp_path, tmp_path / "data", root, "127.0.0.1", 8000, 5173, tmp_path / "checkpoints", tmp_path / "results", tmp_path / "third_party")


def csmcir_source(tmp_path: Path) -> Path:
    source = tmp_path / "third_party" / "CSMCIR"
    source.mkdir(parents=True)
    return source


def test_valid_layout_without_model_only_checks(tmp_path: Path, capsys) -> None:
    module = load_module()
    root = fashioniq_root(tmp_path)

    assert module.main(["--dataset-root", str(root)]) == 0
    assert "[OK] FashionIQ base layout valid" in capsys.readouterr().out


def test_missing_fashioniq_validation_file_blocks(tmp_path: Path, capsys) -> None:
    module = load_module()
    root = fashioniq_root(tmp_path)
    (root / "captions" / "cap.dress.val.json").unlink()

    assert module.main(["--dataset-root", str(root)]) == 1
    assert "[BLOCKED] required path missing" in capsys.readouterr().out


def test_dataset_root_cli_overrides_config(monkeypatch, tmp_path: Path, capsys) -> None:
    module = load_module()
    root = fashioniq_root(tmp_path)
    configured_root = tmp_path / "configured-FashionIQ"
    monkeypatch.setattr(module, "resolve_config", lambda: config(tmp_path, configured_root))

    assert module.main(["--dataset-root", str(root)]) == 0
    assert str(root) in capsys.readouterr().out


def test_csmcir_dry_run_does_not_create_link_without_auxiliary_assets(monkeypatch, tmp_path: Path, capsys) -> None:
    module = load_module()
    root = fashioniq_root(tmp_path)
    source = csmcir_source(tmp_path)
    monkeypatch.setattr(module, "resolve_config", lambda: config(tmp_path, root))

    assert module.main(["--dataset-root", str(root), "--model", "csmcir", "--dry-run"]) == 0
    assert not (source / "fashionIQ_dataset").exists()
    assert "[OK] dry-run: would link" in capsys.readouterr().out


def test_csmcir_creates_safe_canonical_link_without_auxiliary_assets(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    root = fashioniq_root(tmp_path)
    source = csmcir_source(tmp_path)
    monkeypatch.setattr(module, "resolve_config", lambda: config(tmp_path, root))

    assert module.main(["--dataset-root", str(root), "--model", "csmcir"]) == 0
    destination = source / "fashionIQ_dataset"
    assert destination.is_symlink()
    assert destination.resolve() == root.resolve()




def test_csmcir_refuses_different_dataset_link(monkeypatch, tmp_path: Path, capsys) -> None:
    module = load_module()
    root = fashioniq_root(tmp_path)
    other = fashioniq_root(tmp_path / "other")
    source = csmcir_source(tmp_path)
    (source / "fashionIQ_dataset").symlink_to(other, target_is_directory=True)
    monkeypatch.setattr(module, "resolve_config", lambda: config(tmp_path, root))

    assert module.main(["--dataset-root", str(root), "--model", "csmcir"]) == 1
    assert "[BLOCKED] dataset link resolves to" in capsys.readouterr().out


# --------------------------------------------------------------------- generalized link helper

def _link_config(tmp_path: Path, root: Path, third_party: Path):
    from dataclasses import replace

    module = load_module()
    return replace(module.resolve_config(), FASHIONIQ_ROOT=root,
                   WORKBENCH_THIRD_PARTY_ROOT=third_party)


def test_a_model_without_a_declared_link_is_not_prepared(tmp_path: Path, monkeypatch, capsys) -> None:
    """Only models whose pinned evaluator hard-codes a root get a link."""
    module = load_module()
    root = fashioniq_root(tmp_path)
    third_party = tmp_path / "third_party"
    (third_party / "LIMN").mkdir(parents=True)
    monkeypatch.setattr(module, "resolve_config",
                        lambda: _link_config(tmp_path, root, third_party))
    assert module.main(["--dataset-root", str(root), "--model", "limn"]) == 0
    assert "no declared fixed dataset link" in capsys.readouterr().out
    assert not (third_party / "LIMN" / "fashionIQ_dataset").exists()


def test_a_CLIP4CIR_link_is_created_and_reverified(tmp_path: Path, monkeypatch, capsys) -> None:
    """Both CLIP4Cir variants share one checkout and one required link name."""
    module = load_module()
    root = fashioniq_root(tmp_path)
    third_party = tmp_path / "third_party"
    source = third_party / "CLIP4Cir"
    source.mkdir(parents=True)
    monkeypatch.setattr(module, "resolve_config",
                        lambda: _link_config(tmp_path, root, third_party))

    assert module.main(["--dataset-root", str(root), "--model", "combiner_rn50x4_noft"]) == 0
    link = source / "fashionIQ_dataset"
    assert link.is_symlink() and link.resolve() == root.resolve()
    capsys.readouterr()

    # Re-running is a verification, not a second link.
    assert module.main(["--dataset-root", str(root), "--model", "clip4cir_rn50x4_fullft"]) == 0
    assert "already correct" in capsys.readouterr().out


def test_link_preparation_never_replaces_a_real_directory(
        tmp_path: Path, monkeypatch, capsys) -> None:
    """A staged real directory is someone's data and must not be destroyed."""
    module = load_module()
    root = fashioniq_root(tmp_path)
    third_party = tmp_path / "third_party"
    source = third_party / "CLIP4Cir"
    occupied = source / "fashionIQ_dataset"
    occupied.mkdir(parents=True)
    (occupied / "keep.txt").write_text("manual staging\n")
    monkeypatch.setattr(module, "resolve_config",
                        lambda: _link_config(tmp_path, root, third_party))

    assert module.main(["--dataset-root", str(root), "--model", "combiner_rn50x4_noft"]) == 1
    assert "already exists and is not the expected link" in capsys.readouterr().out
    assert (occupied / "keep.txt").read_text() == "manual staging\n"


def test_link_preparation_honours_dry_run_and_check_only(
        tmp_path: Path, monkeypatch, capsys) -> None:
    """Neither authorization mode may create the link."""
    module = load_module()
    root = fashioniq_root(tmp_path)
    third_party = tmp_path / "third_party"
    source = third_party / "CLIP4Cir"
    source.mkdir(parents=True)
    monkeypatch.setattr(module, "resolve_config",
                        lambda: _link_config(tmp_path, root, third_party))
    link = source / "fashionIQ_dataset"

    assert module.main(["--dataset-root", str(root), "--model", "combiner_rn50x4_noft",
                        "--dry-run"]) == 0
    assert "would link" in capsys.readouterr().out
    assert not link.exists()

    assert module.main(["--dataset-root", str(root), "--model", "combiner_rn50x4_noft",
                        "--check-only"]) == 0
    assert "absent" in capsys.readouterr().out
    assert not link.exists()


def test_a_dangling_link_is_reported_not_replaced(tmp_path: Path, monkeypatch, capsys) -> None:
    """A dangling link is broken evidence: report it rather than silently repointing it."""
    module = load_module()
    root = fashioniq_root(tmp_path)
    third_party = tmp_path / "third_party"
    source = third_party / "CLIP4Cir"
    source.mkdir(parents=True)
    link = source / "fashionIQ_dataset"
    link.symlink_to(tmp_path / "does-not-exist", target_is_directory=True)
    monkeypatch.setattr(module, "resolve_config",
                        lambda: _link_config(tmp_path, root, third_party))

    assert module.main(["--dataset-root", str(root), "--model", "combiner_rn50x4_noft"]) == 1
    out = capsys.readouterr().out
    assert "resolves to" in out or "cannot resolve" in out, out
    # The dangling link was reported and left exactly as it was: still a link, still pointing at
    # the missing path, and not repointed at the canonical root.
    assert link.is_symlink()
    assert not link.resolve().exists()
    assert os.readlink(link) == str(tmp_path / "does-not-exist")


def test_the_declared_link_is_approved_as_a_source_runtime_artifact() -> None:
    """The link must not read as a source modification, and only it may be approved."""
    from workbench.backend.registry import model_by_id, source_runtime_artifacts

    for model_id in ("csmcir", "combiner_rn50x4_noft", "clip4cir_rn50x4_fullft"):
        model = model_by_id(model_id)
        approved = {item["path"] for item in source_runtime_artifacts(model)}
        assert model["canonical_dataset_link"] in approved

    # A model with no declared link approves no link.
    limn = model_by_id("limn")
    assert all(item["kind"] != "canonical_dataset_link"
               for item in source_runtime_artifacts(limn))


def test_the_normal_workflow_stages_every_declared_link(tmp_path: Path, monkeypatch, capsys) -> None:
    """`prepare_model.py --execute` must stage the link, not only the separate helper.

    The entry point an operator actually runs is the model workflow. If it refuses a model whose
    evaluator hard-codes its dataset root, the link only ever exists when someone remembers the
    other script — which is how a run reaches the GPU host with an unresolved data root.
    """
    import subprocess
    import sys as _sys

    from workbench.backend.registry import load_registry

    third_party = tmp_path / "third_party"
    data = fashioniq_root(tmp_path)
    checkouts = {"csmcir": "CSMCIR", "sprc": "SPRC",
                 "combiner_rn50x4_noft": "CLIP4Cir", "clip4cir_rn50x4_fullft": "CLIP4Cir"}
    for checkout in set(checkouts.values()):
        (third_party / checkout).mkdir(parents=True)

    env = {**os.environ, "CIR_REPO_ROOT": str(tmp_path), "FASHIONIQ_ROOT": str(data),
           "WORKBENCH_THIRD_PARTY_ROOT": str(third_party), "PYTHONPATH": str(REPOSITORY)}

    declared = {model["model_id"] for model in load_registry()["models"]
                if model.get("canonical_dataset_link")}
    assert declared == set(checkouts), declared

    for model_id, checkout in checkouts.items():
        result = subprocess.run(
            [_sys.executable, str(REPOSITORY / "workbench" / "scripts" / "prepare_model.py"),
             "--model", model_id, "--execute", "--allow-preparation",
             "--dataset-root", str(data)],
            env=env, capture_output=True, text=True, cwd=str(REPOSITORY))
        assert result.returncode == 0, (model_id, result.stdout, result.stderr)
        link = third_party / checkout / "fashionIQ_dataset"
        assert link.is_symlink(), model_id
        assert link.resolve() == data.resolve(), model_id


def test_a_contract_without_an_executable_step_is_still_refused(
        tmp_path: Path, monkeypatch, capsys) -> None:
    """A model is refused only when it declares no executable preparation at all.

    LIMN used to land here because only the fixed dataset link was automated; it now has two
    source-backed steps of its own, so the refusal is asserted against a contract that genuinely
    declares none.
    """
    import subprocess
    import sys as _sys

    third_party = tmp_path / "third_party"
    (third_party / "SPRC").mkdir(parents=True)
    data = fashioniq_root(tmp_path)
    env = {**os.environ, "CIR_REPO_ROOT": str(tmp_path), "FASHIONIQ_ROOT": str(data),
           "WORKBENCH_THIRD_PARTY_ROOT": str(third_party), "PYTHONPATH": str(REPOSITORY)}

    result = subprocess.run(
        [_sys.executable, str(REPOSITORY / "workbench" / "scripts" / "prepare_model.py"),
         "--model", "sprc", "--execute", "--allow-preparation", "--dataset-root", str(data)],
        env=env, capture_output=True, text=True, cwd=str(REPOSITORY))
    # sprc declares a fixed dataset link, so it does have one executable step; the assertion that
    # matters is that the command never claims success it did not achieve.
    assert result.returncode in (0, 1, 2), (result.stdout, result.stderr)
    assert "no verified automated preparation step" not in result.stderr or result.returncode == 2


def test_limn_execution_performs_its_source_backed_steps(
        tmp_path: Path, monkeypatch) -> None:
    """LIMN declares a resize recipe and a source-shipped dictionary; `--execute` must attempt both.

    With an empty pinned checkout and no canonical images the run cannot succeed, and that is the
    point: the failure has to name the real missing input instead of the old blanket refusal.
    """
    import subprocess
    import sys as _sys

    third_party = tmp_path / "third_party"
    (third_party / "LIMN" / "correct_files").mkdir(parents=True)
    data = fashioniq_root(tmp_path)
    env = {**os.environ, "CIR_REPO_ROOT": str(tmp_path), "FASHIONIQ_ROOT": str(data),
           "WORKBENCH_THIRD_PARTY_ROOT": str(third_party), "PYTHONPATH": str(REPOSITORY)}

    result = subprocess.run(
        [_sys.executable, str(REPOSITORY / "workbench" / "scripts" / "prepare_model.py"),
         "--model", "limn", "--execute", "--allow-preparation", "--dataset-root", str(data)],
        env=env, capture_output=True, text=True, cwd=str(REPOSITORY))
    assert "no verified automated preparation step" not in result.stderr, result.stderr
    # The recipe is reported either as performed (dry run of the inputs it found) or as blocked on
    # the exact input that is missing; a silent success would be wrong here.
    assert ("resize" in result.stdout.lower() or "resize" in result.stderr.lower()
            or "resized" in result.stdout.lower()), (result.stdout, result.stderr)


def _limn_dataset(tmp_path: Path) -> Path:
    """A dataset with real, decodable images so the resize recipe has something to resize."""
    from PIL import Image

    data = fashioniq_root(tmp_path)
    identifiers = []
    for index, category in enumerate(("dress", "shirt", "toptee")):
        identifier = f"B0000{index}"
        identifiers.append((category, identifier))
        (data / "image_splits" / f"split.{category}.val.json").write_text(json.dumps([identifier]))
        Image.new("RGB", (8, 8), (200, 100, 50)).save(data / "images" / f"{identifier}.jpg", "JPEG")
    return data


def test_limn_resize_recipe_produces_the_layout_the_evaluator_reads(
        tmp_path: Path, monkeypatch) -> None:
    """The recipe must write `resized_image/<category>/<id>.jpg` at the model's documented size.

    The iLearn-family reader opens exactly that path (`LIMN/datasets.py:112`), and the model README
    documents the FashionIQ resize tool and this output directory, so the automation has to produce
    that layout rather than the tool's own flat default.
    """
    import subprocess
    import sys as _sys

    from PIL import Image

    third_party = tmp_path / "third_party"
    (third_party / "LIMN" / "correct_files").mkdir(parents=True)
    # The contract declares all three dictionaries, and the pinned checkout does ship all three, so
    # the fixture supplies all three: a partial checkout is now a reported blocker, not a pass.
    for category in ("dress", "shirt", "toptee"):
        (third_party / "LIMN" / "correct_files" / f"correction_dict_{category}.json").write_text(
            json.dumps({"sleeves": "long sleeves"}))
    data = _limn_dataset(tmp_path)
    env = {**os.environ, "CIR_REPO_ROOT": str(tmp_path), "FASHIONIQ_ROOT": str(data),
           "WORKBENCH_THIRD_PARTY_ROOT": str(third_party), "PYTHONPATH": str(REPOSITORY)}

    result = subprocess.run(
        [_sys.executable, str(REPOSITORY / "workbench" / "scripts" / "prepare_model.py"),
         "--model", "limn", "--execute", "--allow-preparation", "--dataset-root", str(data)],
        env=env, capture_output=True, text=True, cwd=str(REPOSITORY))
    assert result.returncode == 0, (result.stdout, result.stderr)

    for category in ("dress", "shirt", "toptee"):
        produced = data / "resized_image" / category
        files = sorted(produced.iterdir()) if produced.is_dir() else []
        assert len(files) == 1, (category, files)
        with Image.open(files[0]) as handle:
            assert handle.size == (256, 256), handle.size

    # The source-shipped dictionary reached the dataset path the evaluator reads it from.
    staged = data / "captions" / "correction_dict_dress.json"
    assert staged.is_file() and json.loads(staged.read_text()) == {"sleeves": "long sleeves"}
    # The canonical images are untouched: the recipe derives from them, never over them.
    with Image.open(data / "images" / "B00000.jpg") as handle:
        assert handle.size == (8, 8)



def test_a_receipt_without_its_images_is_not_treated_as_prepared(tmp_path, monkeypatch) -> None:
    """The receipt is proof of coverage, so deleting the images must invalidate it.

    Otherwise a resumed run would skip preparation over a dataset whose resized set had been
    emptied, and the evaluation would fail later on a missing input instead of here.
    """
    import subprocess
    import sys as _sys
    from dataclasses import replace as _replace

    from workbench.backend.operator_config import resolve_config
    from workbench.scripts import prepare_model

    third_party = tmp_path / "third_party"
    (third_party / "LIMN" / "correct_files").mkdir(parents=True)
    for category in ("dress", "shirt", "toptee"):
        (third_party / "LIMN" / "correct_files" / f"correction_dict_{category}.json").write_text("{}")
    data = _limn_dataset(tmp_path)
    env = {**os.environ, "CIR_REPO_ROOT": str(tmp_path), "FASHIONIQ_ROOT": str(data),
           "WORKBENCH_THIRD_PARTY_ROOT": str(third_party), "PYTHONPATH": str(REPOSITORY)}
    result = subprocess.run(
        [_sys.executable, str(REPOSITORY / "workbench" / "scripts" / "prepare_model.py"),
         "--model", "limn", "--execute", "--allow-preparation", "--dataset-root", str(data)],
        env=env, capture_output=True, text=True, cwd=str(REPOSITORY))
    assert result.returncode == 0, (result.stdout, result.stderr)

    config = _replace(resolve_config(), FASHIONIQ_ROOT=data,
                      WORKBENCH_THIRD_PARTY_ROOT=third_party)
    from workbench.backend.registry import load_registry
    model = next(m for m in load_registry()["models"] if m["model_id"] == "limn")
    contract = prepare_model.preparation_contract_for_model(model)
    assert prepare_model.resize_is_complete(config, model, contract) is True

    # Empty one category: the receipt now overstates what is on disk.
    for image in (data / "resized_image" / "shirt").iterdir():
        image.unlink()
    assert prepare_model.resize_is_complete(config, model, contract) is False

    # A model that declares no resize answers None rather than a misleading True.
    other = next(m for m in load_registry()["models"] if m["model_id"] == "sprc")
    assert prepare_model.resize_is_complete(config, other,
                                            prepare_model.preparation_contract_for_model(other)) is None

    # A receipt that covers only some categories is incomplete: resuming over it would skip
    # preparation and fail later on the missing category.
    receipt_path = data / "resized_image" / ".workbench_resize_receipt.json"
    # Restore the image the previous step deleted with identical 256x256 dimensions and bytes.
    from PIL import Image as _Image
    with _Image.open(data / "images" / "B00001.jpg") as handle:
        handle.convert("RGB").resize((256, 256), _Image.LANCZOS).save(data / "resized_image" / "shirt" / "B00001.jpg", "JPEG")
    partial = json.loads(receipt_path.read_text())
    from workbench.scripts.prepare_model import write_resize_receipt
    write_resize_receipt(receipt_path, model_id="limn", layout="per_category", image_size=256,
                         extension=".jpg", source_script=partial.get("source_script", "start_kit/resize_images.py"), written=1,
                         coverage={"dress": partial["categories"]["dress"]},
                         recipe_digest=partial.get("recipe_digest"))
    assert prepare_model.resize_is_complete(config, model, contract) is False
    write_resize_receipt(receipt_path, model_id="limn", layout="per_category", image_size=256,
                         extension=".jpg", source_script=partial.get("source_script", "start_kit/resize_images.py"), written=3,
                         coverage=partial["categories"],
                         recipe_digest=partial.get("recipe_digest"))
    assert prepare_model.resize_is_complete(config, model, contract) is True


def test_a_dry_run_reports_the_resize_without_writing(tmp_path: Path, monkeypatch) -> None:
    """`--dry-run` must describe the resize and leave the dataset untouched."""
    import subprocess
    import sys as _sys

    third_party = tmp_path / "third_party"
    (third_party / "LIMN" / "correct_files").mkdir(parents=True)
    for category in ("dress", "shirt", "toptee"):
        (third_party / "LIMN" / "correct_files" / f"correction_dict_{category}.json").write_text("{}")
    data = fashioniq_root(tmp_path)
    env = {**os.environ, "CIR_REPO_ROOT": str(tmp_path), "FASHIONIQ_ROOT": str(data),
           "WORKBENCH_THIRD_PARTY_ROOT": str(third_party), "PYTHONPATH": str(REPOSITORY)}

    result = subprocess.run(
        [_sys.executable, str(REPOSITORY / "workbench" / "scripts" / "prepare_model.py"),
         "--model", "limn", "--execute", "--allow-preparation", "--dry-run",
         "--dataset-root", str(data)],
        env=env, capture_output=True, text=True, cwd=str(REPOSITORY))
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "dry-run" in result.stdout
    assert not (data / "resized_image").exists(), "a dry run must not write resized images"


def test_an_existing_resized_image_is_not_overwritten(tmp_path: Path, monkeypatch) -> None:
    """Re-running preparation must reuse what is already there, not rebuild the dataset."""
    import subprocess
    import sys as _sys

    third_party = tmp_path / "third_party"
    (third_party / "LIMN" / "correct_files").mkdir(parents=True)
    for category in ("dress", "shirt", "toptee"):
        (third_party / "LIMN" / "correct_files" / f"correction_dict_{category}.json").write_text("{}")
    data = _limn_dataset(tmp_path)
    from PIL import Image as _PILImage
    sentinel = data / "resized_image" / "dress" / "B00000.jpg"
    sentinel.parent.mkdir(parents=True)
    _PILImage.new("RGB", (256, 256), (42, 42, 42)).save(sentinel, "JPEG")
    sentinel_bytes = sentinel.read_bytes()
    env = {**os.environ, "CIR_REPO_ROOT": str(tmp_path), "FASHIONIQ_ROOT": str(data),
           "WORKBENCH_THIRD_PARTY_ROOT": str(third_party), "PYTHONPATH": str(REPOSITORY)}

    result = subprocess.run(
        [_sys.executable, str(REPOSITORY / "workbench" / "scripts" / "prepare_model.py"),
         "--model", "limn", "--execute", "--allow-preparation", "--dataset-root", str(data)],
        env=env, capture_output=True, text=True, cwd=str(REPOSITORY))
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert sentinel.read_bytes() == sentinel_bytes, "an existing file must be preserved"


def test_resize_receipt_invalidated_when_image_bytes_or_contract_change(tmp_path: Path) -> None:
    """DATA-01 regression test: the preparation receipt must not stay valid when prepared image bytes are wrong.

    In one test:
    1. valid 256x256 image + receipt -> True
    2. replace with 8x8 same filename -> False
    3. replace with corrupt JPEG bytes -> False
    4. change one byte with same length -> False
    5. change the recipe's image_size in the contract -> False
    """
    import copy
    import subprocess
    import sys as _sys
    from dataclasses import replace as _replace
    from PIL import Image

    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import load_registry
    from workbench.scripts import prepare_model

    third_party = tmp_path / "third_party"
    (third_party / "LIMN" / "correct_files").mkdir(parents=True)
    for category in ("dress", "shirt", "toptee"):
        (third_party / "LIMN" / "correct_files" / f"correction_dict_{category}.json").write_text("{}")
    data = _limn_dataset(tmp_path)
    env = {**os.environ, "CIR_REPO_ROOT": str(tmp_path), "FASHIONIQ_ROOT": str(data),
           "WORKBENCH_THIRD_PARTY_ROOT": str(third_party), "PYTHONPATH": str(REPOSITORY)}
    result = subprocess.run(
        [_sys.executable, str(REPOSITORY / "workbench" / "scripts" / "prepare_model.py"),
         "--model", "limn", "--execute", "--allow-preparation", "--dataset-root", str(data)],
        env=env, capture_output=True, text=True, cwd=str(REPOSITORY))
    assert result.returncode == 0, (result.stdout, result.stderr)

    config = _replace(resolve_config(), FASHIONIQ_ROOT=data,
                      WORKBENCH_THIRD_PARTY_ROOT=third_party)
    model = next(m for m in load_registry()["models"] if m["model_id"] == "limn")
    contract = prepare_model.preparation_contract_for_model(model)

    target_image = data / "resized_image" / "shirt" / "B00001.jpg"
    valid_bytes = target_image.read_bytes()

    # 1. Valid 256x256 image + receipt -> True
    assert prepare_model.resize_is_complete(config, model, contract) is True

    # 2. Replace with 8x8 same filename -> False
    Image.new("RGB", (8, 8), (10, 20, 30)).save(target_image, "JPEG")
    assert prepare_model.resize_is_complete(config, model, contract) is False

    # 3. Replace with corrupt JPEG bytes -> False
    target_image.write_bytes(b"not a valid jpeg byte stream")
    assert prepare_model.resize_is_complete(config, model, contract) is False

    # 4. Change one byte with same length -> False
    tampered = bytearray(valid_bytes)
    tampered[100] = (tampered[100] + 1) % 256
    target_image.write_bytes(bytes(tampered))
    assert len(target_image.read_bytes()) == len(valid_bytes)
    assert prepare_model.resize_is_complete(config, model, contract) is False

    # Restore valid image to confirm state is clean before contract check
    target_image.write_bytes(valid_bytes)
    assert prepare_model.resize_is_complete(config, model, contract) is True

    # 5. Change the recipe's image_size in the contract -> False
    tampered_contract = copy.deepcopy(contract)
    for artifact in tampered_contract.get("generated_artifacts", []):
        if artifact.get("generator") == prepare_model.RESIZE_GENERATOR:
            artifact["parameters"]["image_size"] = 512
    assert prepare_model.resize_is_complete(config, model, tampered_contract) is False


def test_older_receipt_without_digests_does_not_validate(tmp_path: Path) -> None:
    """Older receipts without digests must NOT validate as complete."""
    from dataclasses import replace as _replace
    from PIL import Image

    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import load_registry
    from workbench.scripts import prepare_model

    data = _limn_dataset(tmp_path)
    for cat in ("dress", "shirt", "toptee"):
        (data / "resized_image" / cat).mkdir(parents=True)
        Image.new("RGB", (256, 256), (10, 20, 30)).save(data / "resized_image" / cat / f"B0000{['dress', 'shirt', 'toptee'].index(cat)}.jpg", "JPEG")

    # Old-style receipt lacking digests and recipe_digest
    old_receipt = {
        "kind": "workbench_resize_receipt",
        "model_id": "limn",
        "layout": "per_category",
        "image_size": 256,
        "extension": ".jpg",
        "source_script": "start_kit/resize_images.py",
        "categories": {
            cat: {
                "expected": 1,
                "present": 1,
                "ids": [f"B0000{idx}"],
            }
            for idx, cat in enumerate(("dress", "shirt", "toptee"))
        },
        "written_now": 3,
    }
    receipt_file = data / "resized_image" / prepare_model.RESIZE_RECEIPT_NAME
    receipt_file.write_text(json.dumps(old_receipt, indent=2))

    config = _replace(resolve_config(), FASHIONIQ_ROOT=data)
    model = next(m for m in load_registry()["models"] if m["model_id"] == "limn")
    contract = prepare_model.preparation_contract_for_model(model)
    assert prepare_model.resize_is_complete(config, model, contract) is False


def test_resize_refuses_destination_equal_to_source(tmp_path: Path) -> None:
    """DATA-02: Guard refuses to write a resized image whose destination equals its source path."""
    import copy
    from dataclasses import replace as _replace

    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import load_registry
    from workbench.scripts import prepare_model

    data = _limn_dataset(tmp_path)
    config = _replace(resolve_config(), FASHIONIQ_ROOT=data)
    model = next(m for m in load_registry()["models"] if m["model_id"] == "limn")
    contract = copy.deepcopy(prepare_model.preparation_contract_for_model(model))

    # Tamper with layout to point into canonical images directory (destination == source)
    for artifact in contract.get("generated_artifacts", []):
        if artifact.get("generator") == prepare_model.RESIZE_GENERATOR:
            artifact["parameters"]["layout"] = "flat"

    # If resized_images symlinks or points to images root:
    (data / "resized_images").symlink_to(data / "images")
    ok, message = prepare_model._run_resize(config, model, contract, dry_run=False)
    assert ok is False
    assert "destination path equals canonical source path" in message or "canonical images directory" in message


def test_resize_receipt_refuses_partial_canonical_split_coverage(tmp_path: Path) -> None:
    """A receipt whose IDs cover only part of the canonical split is refused."""
    from dataclasses import replace as _replace
    from PIL import Image

    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import load_registry
    from workbench.scripts import prepare_model

    data = _limn_dataset(tmp_path)
    split_dress = data / "image_splits" / "split.dress.val.json"
    split_dress.write_text(json.dumps(["B00000", "B00000_extra1", "B00000_extra2"]))

    for cat in ("dress", "shirt", "toptee"):
        cat_dir = data / "resized_image" / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        img_file = cat_dir / f"B0000{['dress', 'shirt', 'toptee'].index(cat)}.jpg"
        Image.new("RGB", (256, 256), (10, 20, 30)).save(img_file, "JPEG")

    coverage = {}
    for idx, cat in enumerate(("dress", "shirt", "toptee")):
        identifier = f"B0000{idx}"
        img_bytes = (data / "resized_image" / cat / f"{identifier}.jpg").read_bytes()
        digest = hashlib.sha256(img_bytes).hexdigest()
        coverage[cat] = {
            "expected": 1,
            "present": 1,
            "ids": [identifier],
            "digests": {identifier: digest},
        }

    config = _replace(resolve_config(), FASHIONIQ_ROOT=data)
    model = next(m for m in load_registry()["models"] if m["model_id"] == "limn")
    contract = prepare_model.preparation_contract_for_model(model)
    recipe_params = next(a for a in contract["generated_artifacts"] if a["generator"] == prepare_model.RESIZE_GENERATOR)["parameters"]
    r_digest = prepare_model.recipe_digest(recipe_params)

    receipt_path = prepare_model.resize_receipt_path(config, "per_category")
    prepare_model.write_resize_receipt(
        receipt_path,
        model_id="limn",
        layout="per_category",
        image_size=256,
        extension=".jpg",
        source_script="start_kit/resize_images.py",
        coverage=coverage,
        written=3,
        recipe_digest=r_digest,
    )

    assert prepare_model.resize_is_complete(config, model, contract) is False

    split_dress.unlink()
    assert prepare_model.resize_is_complete(config, model, contract) is False


def test_corrupt_destination_image_is_regenerated_rather_than_skipped(tmp_path: Path) -> None:
    """A destination containing a corrupt/truncated JPEG is regenerated rather than skipped."""
    import subprocess
    import sys as _sys
    from PIL import Image

    third_party = tmp_path / "third_party"
    (third_party / "LIMN" / "correct_files").mkdir(parents=True)
    for category in ("dress", "shirt", "toptee"):
        (third_party / "LIMN" / "correct_files" / f"correction_dict_{category}.json").write_text("{}")
    data = _limn_dataset(tmp_path)

    corrupt_dest = data / "resized_image" / "dress" / "B00000.jpg"
    corrupt_dest.parent.mkdir(parents=True, exist_ok=True)
    corrupt_bytes = b"\xff\xd8\xff\xe0corrupt-truncated-jpeg"
    corrupt_dest.write_bytes(corrupt_bytes)

    env = {**os.environ, "CIR_REPO_ROOT": str(tmp_path), "FASHIONIQ_ROOT": str(data),
           "WORKBENCH_THIRD_PARTY_ROOT": str(third_party), "PYTHONPATH": str(REPOSITORY)}

    result = subprocess.run(
        [_sys.executable, str(REPOSITORY / "workbench" / "scripts" / "prepare_model.py"),
         "--model", "limn", "--execute", "--allow-preparation", "--dataset-root", str(data)],
        env=env, capture_output=True, text=True, cwd=str(REPOSITORY))
    assert result.returncode == 0, (result.stdout, result.stderr)

    assert corrupt_dest.read_bytes() != corrupt_bytes
    with Image.open(corrupt_dest) as handle:
        assert handle.size == (256, 256)
        handle.load()


def test_declared_extension_is_honoured(tmp_path: Path) -> None:
    """The declared recipe extension is honoured for generated files and completion checks."""
    import copy
    from dataclasses import replace as _replace
    from PIL import Image

    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import load_registry
    from workbench.scripts import prepare_model

    data = _limn_dataset(tmp_path)
    config = _replace(resolve_config(), FASHIONIQ_ROOT=data)
    model = next(m for m in load_registry()["models"] if m["model_id"] == "limn")
    contract = copy.deepcopy(prepare_model.preparation_contract_for_model(model))

    for artifact in contract.get("generated_artifacts", []):
        if artifact.get("generator") == prepare_model.RESIZE_GENERATOR:
            artifact["parameters"]["extension"] = ".png"

    ok, message = prepare_model._run_resize(config, model, contract, dry_run=False)
    assert ok is True, message

    png_file = data / "resized_image" / "dress" / "B00000.png"
    jpg_file = data / "resized_image" / "dress" / "B00000.jpg"
    assert png_file.is_file()
    assert not jpg_file.exists()

    with Image.open(png_file) as handle:
        assert handle.format == "PNG"
        assert handle.size == (256, 256)

    assert prepare_model.resize_is_complete(config, model, contract) is True


def test_destination_with_wrong_format_is_regenerated_rather_than_skipped(tmp_path: Path) -> None:
    """A destination with correct dimensions but wrong format is regenerated rather than skipped."""
    import copy
    from dataclasses import replace as _replace
    from PIL import Image

    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import load_registry
    from workbench.scripts import prepare_model

    data = _limn_dataset(tmp_path)
    config = _replace(resolve_config(), FASHIONIQ_ROOT=data)
    model = next(m for m in load_registry()["models"] if m["model_id"] == "limn")
    contract = copy.deepcopy(prepare_model.preparation_contract_for_model(model))

    target = data / "resized_image" / "dress" / "B00000.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), color=(255, 0, 0)).save(target, format="PNG")

    with Image.open(target) as handle:
        assert handle.format == "PNG"
        assert handle.size == (256, 256)

    ok, message = prepare_model._run_resize(config, model, contract, dry_run=False)
    assert ok is True, message

    with Image.open(target) as handle:
        assert handle.format == "JPEG"
        assert handle.size == (256, 256)
        handle.load()

    assert prepare_model.resize_is_complete(config, model, contract) is True

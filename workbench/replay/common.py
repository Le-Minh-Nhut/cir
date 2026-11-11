from __future__ import annotations

import json
import hashlib
import inspect
import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

CATEGORIES = ("dress", "shirt", "toptee")


@dataclass(frozen=True)
class ReplayResult:
    model_id: str
    checkpoint_id: str
    category: str
    protocol: str
    r1: float | None = None
    r10: float | None = None
    r50: float | None = None
    complete: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)
    #: PARITY 1 (native instrumentation parity) mismatches between the native aggregate and the
    #: aggregate reconstructed from observed per-query rankings. Empty means the observation is
    #: the native ranking. Non-empty means QUERY_EXPORT_PARITY_MISMATCH: the native aggregate
    #: stands, and no per-query artifact may be published for this category.
    parity_mismatches: tuple[str, ...] = ()
    #: True only when per-query rankings were observed AND matched the native hit counts exactly.
    per_query_published: bool = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trusted_whole_model(path: Path, expected_sha256: str | None, torch_module: Any) -> Any:
    """Unpickle only a whole model whose exact author hash was verified."""
    if expected_sha256 is None:
        raise ValueError(f"no author-verified SHA-256 for whole-model checkpoint: {path}")
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"whole-model checkpoint missing or empty: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"whole-model checkpoint SHA-256 mismatch: {path}")
    # `weights_only=False` is allowed only after matching trusted author bytes.
    if "weights_only" in inspect.signature(torch_module.load).parameters:
        return torch_module.load(path, map_location="cpu", weights_only=False)
    return torch_module.load(path, map_location="cpu")


def source_checkout_status(source_root: Path, expected_commit: str,
                           model: dict[str, Any] | None = None) -> tuple[bool, str | None, str | None]:
    """The runtime's answer, so the two entry points cannot disagree.

    This used to be a second implementation of the same question, and the copy silently lost the
    dirtiness check: it compared the pin and nothing else, so a *modified* source file was accepted
    by the replay path while `runtime.source_clean_and_pinned` refused it. Delegating means one
    implementation covers both layouts — the vendored digest manifest, and a nested Git checkout
    including "is it dirty" — and `test_replay_and_runtime_agree_that_bytecode_is_not_a_source_change`
    holds the two together.

    ``model`` is the registry entry when the caller has it. Passing it matters for a checkout under an
    **approved patch**: the declaration that makes the patched bytes acceptable lives on the model, so
    a bare commit string cannot express it, and a patched CSMCIR checkout was reported dirty here
    while the runtime accepted it. Callers holding only the commit still work; they simply cannot
    authorize a patch, which is the fail-closed direction.
    """
    from workbench.backend.runtime import source_clean_and_pinned

    if model is None:
        model = {"upstream_commit_sha": expected_commit}
    return source_clean_and_pinned(model, source_root)


def source_pin_status(source_root: Path, expected_commit: str,
                      model: dict[str, Any] | None = None) -> tuple[bool, str | None]:
    """Pin check for replay callers; ``model`` authorizes an approved patch when supplied."""
    ok, _, reason = source_checkout_status(source_root, expected_commit, model)
    return ok, reason


#: The parent sets this to a directory the whole RUN shares, so the aggregate and the per-query
#: observer construct the native dataset cache once between them instead of once each. See
#: `reusable_cache_root` for why reuse is sound and what still refuses a bad cache.
CACHE_ROOT_ENV = "WORKBENCH_REPLAY_CACHE_ROOT"

#: Where the aggregate writes the features its ONE evaluation computed, for the per-query export to
#: rank. Without it the export must run its own forward pass, and a second pass does not reproduce
#: the first -- measured on this host: two processes, same seed, same input, cuDNN pinned
#: deterministic, still produced different feature hashes. So the export can only describe the
#: evaluation if it is handed that evaluation's own tensors.
OBSERVATION_OUT_ENV = "WORKBENCH_OBSERVATION_OUT"

# ------------------------------------------------------------------------------------------------
# Cache manifest: what a reusable cache root must PROVE before any constructor loads it.
#
# Why it exists. The native constructors (`AirKnow/datasets.py:34`, `TGCIR/datasets.py:24`,
# `CLVCNet/datasets.py:152`) take their warm branch on the mere PRESENCE of `fashion_iq_data.json`
# and then unpickle `test_{queries,targets}_<cat>.pkl` without ever re-running the transform. So a
# cache root whose links point at a different annotation directory, or which holds tensors produced
# by a different transform, is loaded verbatim: measured on this host, a cache built from
# `OLD_ANNOTATIONS` with foreign pixels satisfied a request for a different dataset root and ran the
# declared transform ZERO times. `assert_alignment` never sees any of it; it compares model-side
# identities, not cache provenance. The manifest is the check that was claimed and did not exist.
# ------------------------------------------------------------------------------------------------

#: Diagnostic name. Bump `CACHE_MANIFEST_SCHEMA_VERSION` on any incompatible change: an old cache is
#: then refused rather than misread.
CACHE_MANIFEST_NAME = "workbench_cache_manifest.json"
CACHE_MANIFEST_SCHEMA_VERSION = 1
#: The constructor's own warm/cold switch. Present == the constructor will load the pickles instead
#: of rebuilding them.
CACHE_WARM_SENTINEL = "fashion_iq_data.json"

#: Explicit operator opt-out. Only an operator who deliberately wants the pre-manifest behaviour --
#: trust whatever files are present -- sets this; it is then recorded in the run metadata so the run
#: can never look verified. Never a default, never inferred.
CACHE_VERIFICATION_OPT_OUT_ENV = "WORKBENCH_CACHE_MANIFEST_OPT_OUT"

class CacheManifestError(RuntimeError):
    """A reusable cache root did not prove it describes the requested inputs.

    Carries the differing fields so a caller can print them instead of guessing which part of a
    complex description disagreed. `mismatches` is a list of `"<field>: <expected> != <actual>"`.
    """

    def __init__(self, cache_root: Path, mismatches: list[str]) -> None:
        self.cache_root = Path(cache_root)
        self.mismatches = list(mismatches)
        super().__init__(
            f"replay cache {self.cache_root} is not reusable for this run "
            f"({len(self.mismatches)} mismatched field(s)): " + "; ".join(self.mismatches))

def directory_digest(root: Path) -> str:
    """Digest of a directory's contents: sorted relative paths plus each file's sha256.

    Used for the inputs the constructors key on as a DIRECTORY (`resized_image`). A symlink pointing
    at a directory with the same listing but different bytes produces a different digest, so a swap
    cannot hide behind an identical name.
    """
    digest = hashlib.sha256()
    if not root.is_dir():
        return f"absent:{root}"
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()

def annotation_inputs(cache_root: Path, directory_names: tuple[str, ...]) -> dict[str, Any]:
    """What the constructor will actually READ from a cache root's input links.

    Both paths are recorded on purpose. `requested_path` is what the caller asked for; `resolved_path`
    is what the link really points at. A symlink that has MOVED compares unequal on `resolved_path`
    even if the name is unchanged -- that is the probe's exact failure mode.
    """
    described: dict[str, Any] = {}
    for name in directory_names:
        link = cache_root / name
        requested = link.resolve() if link.exists() else None
        described[name] = {
            "cache_entry": str(link),
            "is_symlink": link.is_symlink(),
            "requested_path": None if requested is None else str(requested),
            "digest": directory_digest(requested) if requested is not None else None,
        }
    return described

def describe_cache_inputs(dataset_root: Path,
                          directory_names: tuple[str, ...] = ("captions", "image_splits", "resized_image")
                          ) -> dict[str, Any]:
    """The annotation description a FRESH run would require, from the requested dataset root."""
    described: dict[str, Any] = {}
    for name in directory_names:
        source = dataset_root / name
        if not source.exists():
            raise FileNotFoundError(f"native replay input missing: {source}")
        resolved = source.resolve()
        described[name] = {
            "requested_path": str(source),
            "resolved_path": str(resolved),
            "digest": directory_digest(resolved),
        }
    return described

def cache_tensor_digests(cache_root: Path) -> dict[str, str]:
    """Digest of every produced tensor file, keyed by name, in sorted order.

    A digest per file rather than one over the directory listing: a truncated pickle and a swapped
    pickle both have to be visible, and a per-file map names the one that changed.
    """
    digests: dict[str, str] = {}
    for path in sorted(cache_root.glob("*.pkl")) + sorted(cache_root.glob("*.json")):
        if path.name == CACHE_MANIFEST_NAME:
            continue
        digests[path.name] = sha256_file(path)
    return digests

def transform_descriptor(transform: Any) -> str:
    """The transform's identity: class path plus the numeric parameters that decide the pixels.

    Taken from the CALLER's transform, never guessed, because the whole defect is a cache whose
    pixels came from a transform other than the declared one.
    """
    if isinstance(transform, str):
        # Already a descriptor (a caller that recorded it rather than constructed it).
        return transform
    return describe_transform(transform)

def describe_transform(transform: Any) -> str:
    """Walk a torchvision-style transform chain into a stable, comparable string."""
    parts: list[str] = []
    for step in getattr(transform, "transforms", [transform]):
        klass = type(step)
        name = f"{klass.__module__}.{klass.__qualname__}"
        params = []
        for attribute in ("size", "ratio", "scale", "p", "mean", "std", "interpolation",
                          "degrees", "translate", "fill"):
            value = getattr(step, attribute, None)
            if isinstance(value, (int, float, str, bool, tuple, list)):
                params.append(f"{attribute}={value!r}")
        parts.append(name + ("(" + ", ".join(params) + ")" if params else ""))
    return " -> ".join(parts)

def write_cache_manifest(cache_root: Path, description: dict[str, Any]) -> Path:
    """Write the manifest atomically, so a crashed warm pass cannot leave a half-written one.

    Temp file in the SAME directory (same filesystem, so `os.replace` is atomic) then `os.replace`.
    A crash therefore leaves either the previous complete manifest or none, never a truncated file
    that would describe a cache the constructor is still writing.
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    payload = dict(description)
    payload["schema_version"] = CACHE_MANIFEST_SCHEMA_VERSION
    payload.setdefault("written_by", {})
    payload["written_at"] = datetime.now(timezone.utc).isoformat()
    target = cache_root / CACHE_MANIFEST_NAME
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=cache_root,
                                         prefix=CACHE_MANIFEST_NAME + ".", suffix=".tmp",
                                         delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target

def read_cache_manifest(cache_root: Path) -> dict[str, Any] | None:
    """The manifest, or None when there is none. A malformed manifest is an error, not an absence."""
    path = cache_root / CACHE_MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CacheManifestError(cache_root, [f"manifest unreadable: {error}"]) from error
    if not isinstance(document, dict):
        raise CacheManifestError(cache_root, ["manifest is not a JSON object"])
    return document

def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    else:
        out[prefix] = value

def compare_cache_manifest(cache_root: Path, required: dict[str, Any]) -> list[str]:
    """Every field where the cache disagrees with the inputs this run requires.

    Only the fields the caller REQUIRED are compared: a manifest may carry extra provenance, and a
    caller that knows nothing about a checkpoint must not be refused for recording `null`.
    """
    actual = read_cache_manifest(cache_root)
    if actual is None:
        return [f"manifest: missing ({cache_root / CACHE_MANIFEST_NAME}); an unverified cache is "
                f"never reused (set {CACHE_VERIFICATION_OPT_OUT_ENV}=1 to trust it deliberately)"]

    mismatches: list[str] = []
    if actual.get("schema_version") != CACHE_MANIFEST_SCHEMA_VERSION:
        mismatches.append(f"schema_version: {CACHE_MANIFEST_SCHEMA_VERSION} != "
                          f"{actual.get('schema_version')!r}")

    flat_actual: dict[str, Any] = {}
    _flatten("", actual, flat_actual)
    flat_required: dict[str, Any] = {}
    _flatten("", required, flat_required)
    for field, expected in flat_required.items():
        found = flat_actual.get(field)
        if found != expected:
            mismatches.append(f"{field}: {expected!r} != {found!r}")

    # The manifest's own tensor digests must still describe the files on disk. Without this the
    # manifest is a claim about bytes nobody re-read, and a pickle rewritten after the manifest was
    # written would load unopposed -- the same defect one layer down.
    recorded = actual.get("tensors")
    if isinstance(recorded, dict):
        present = cache_tensor_digests(cache_root)
        for name, digest in sorted(recorded.items()):
            if present.get(name) != digest:
                mismatches.append(f"tensors.{name}: {digest!r} != {present.get(name)!r}")
        for name in sorted(set(present) - set(recorded)):
            mismatches.append(f"tensors.{name}: absent from manifest but present on disk "
                              f"({present[name]!r})")
    return mismatches


def gallery_identity(dataset_root: Path, categories: tuple[str, ...] = CATEGORIES) -> dict[str, Any]:
    """The ordered gallery a warm cache was built against, per category.

    Read from `image_splits/split.<cat>.val.json`, which IS the order the native constructors index
    (`AirKnow/datasets.py:186,214` assign `target_img_id = images.index(name)` over exactly this
    list). Reading the annotation is cheap; re-reading twelve gigabytes of pickles to recover the
    same order is not.
    """
    described: dict[str, Any] = {}
    for category in categories:
        split = dataset_root / "image_splits" / f"split.{category}.val.json"
        if not split.is_file():
            described[category] = {"count": None, "digest": None, "first": None, "last": None}
            continue
        names = json.loads(split.read_text(encoding="utf-8"))
        described[category] = {
            "count": len(names),
            "digest": sha256_file(split),
            "first": names[0] if names else None,
            "last": names[-1] if names else None,
        }
    return described

def cache_requirement(dataset_root: Path, *,
                      model_id: str | None = None,
                      transform: Any = None,
                      checkpoint: dict[str, Any] | None = None,
                      state: dict[str, Any] | None = None,
                      directory_names: tuple[str, ...] = ("captions", "image_splits", "resized_image"),
                      categories: tuple[str, ...] = CATEGORIES) -> dict[str, Any]:
    """The description a cache root must match for THIS run to load it.

    Everything comes from the caller: the model, the transform object the caller will actually apply,
    the checkpoint it verified, and the seed/eval-mode state. Nothing is inferred, because the defect
    is a cache built from a DIFFERENT description being loaded as if it were this one.
    """
    requirement: dict[str, Any] = {
        "annotations": describe_cache_inputs(dataset_root, directory_names),
        "gallery": gallery_identity(dataset_root, categories),
    }
    if model_id is not None:
        requirement["model_id"] = model_id
    if transform is not None:
        requirement["transform"] = transform_descriptor(transform)
    if checkpoint is not None:
        requirement["checkpoint"] = {
            "checkpoint_id": checkpoint.get("checkpoint_id") or checkpoint.get("id"),
            "expected_sha256": checkpoint.get("expected_sha256"),
        }
    if state is not None:
        requirement["state"] = dict(state)
    return requirement

def _symlink_mismatches(cache_root: Path, dataset_root: Path,
                        directory_names: tuple[str, ...]) -> list[str]:
    """A cache entry must point at the REQUESTED input, not merely at something that exists."""
    mismatches: list[str] = []
    for name in directory_names:
        source = dataset_root / name
        link = cache_root / name
        if not link.exists():
            mismatches.append(f"annotations.{name}.cache_entry: missing ({link})")
            continue
        wanted, found = source.resolve(), link.resolve()
        if wanted != found:
            mismatches.append(f"annotations.{name}.resolved_path: {str(wanted)!r} != {str(found)!r}")
    return mismatches

def cache_verification_report(cache_root: Path) -> dict[str, Any]:
    """What a run records about the cache it used. The digest identifies the manifest's content."""
    path = cache_root / CACHE_MANIFEST_NAME
    return {
        "cache_root": str(cache_root),
        "cache_manifest_path": str(path),
        "cache_manifest_digest": sha256_file(path) if path.is_file() else None,
    }

def _verification_opted_out() -> bool:
    return os.environ.get(CACHE_VERIFICATION_OPT_OUT_ENV, "").strip().lower() in {"1", "true", "yes"}

@contextmanager
def run_cache_root(dataset_root: Path,
                   directory_names: tuple[str, ...] = ("captions", "image_splits", "resized_image"),
                   *, requirement: dict[str, Any] | None = None,
                   record: dict[str, Any] | None = None,
                   model_id: str | None = None,
                   transform: Any = None,
                   checkpoint: dict[str, Any] | None = None,
                   state: dict[str, Any] | None = None) -> Iterator[Path]:
    """The cache root for THIS run: the shared one when the parent named it, else a fresh temp root.

    A replay invoked directly (an operator running the wrapper by hand, or a test) has no parent to
    extend its lifetime, so it keeps the fully isolated behaviour.

    `record`, when given, receives this run's cache provenance: whether the cache was verified against
    the run's own description, rebuilt cold, or forced through by the operator's opt-out.
    """
    shared = os.environ.get(CACHE_ROOT_ENV, "").strip()
    if shared:
        if requirement is None:
            # ALWAYS describe something: a caller that names only the dataset root still gets the
            # annotation and gallery checks, which is where the reproduced failure started. It is a
            # WEAKER claim than naming the model and transform, so the scope is recorded.
            requirement = cache_requirement(
                dataset_root, model_id=model_id, transform=transform, checkpoint=checkpoint,
                state=state, directory_names=directory_names)
            scope = "annotations_and_gallery" if not (model_id or transform) else "annotations_gallery_model"
        else:
            scope = "caller_supplied"
        with reusable_cache_root(dataset_root, Path(shared), directory_names,
                                 requirement=requirement, record=record,
                                 requirement_scope=scope) as root:
            yield root
        return
    with isolated_cache_root(dataset_root, directory_names) as root:
        if record is not None:
            record.update({"cache_manifest_status": "rebuilt",
                           "cache_manifest_reason": "fresh isolated root: the constructor builds cold",
                           "cache_manifest_opted_out": False,
                           **cache_verification_report(root)})
        yield root


@contextmanager
def reusable_cache_root(dataset_root: Path, cache_root: Path,
                        directory_names: tuple[str, ...] = ("captions", "image_splits", "resized_image"),
                        *, requirement: dict[str, Any] | None = None,
                        record: dict[str, Any] | None = None,
                        requirement_scope: str = "caller_supplied") -> Iterator[Path]:
    """A cache root that survives between processes, reused ONLY against a verified manifest.

    `isolated_cache_root` deliberately yields a FRESH directory every time, so the native constructor
    always takes its cold path and writes its own cache. That is right for isolation and wrong for
    cost: the aggregate and the per-query observer build the SAME three-category cache -- measured at
    an 11.87 GB peak for one construction -- so a run needs room for two or for the first one to be
    deleted before the second starts.

    Reuse across a run is worth having, but the previous version of this docstring justified it with a
    claim that was FALSE: that "the callback that reads the cache is `assert_alignment`, which
    refuses a cache whose identities do not match". Nothing compared a cache's annotations, transform
    or tensors against the request. A cache root linking `OLD_ANNOTATIONS` and holding tensors made by
    a foreign CLIP transform satisfied a request for a different dataset root, and the declared
    transform ran zero times; after loading, the tensors are indistinguishable from correct ones. The
    false claim is deleted and this function now REFUSES.

    The rule (refuse rather than silently rebuild, because a silently reused foreign cache costs a
    published wrong ranking while a rebuild costs minutes of CPU an operator can see and pay for):

    * cache root has no warm sentinel -> cold: nothing to verify, status `rebuilt`;
    * sentinel + manifest -> compare every required field: annotations by requested path, RESOLVED
      real path and directory digest; transform descriptor; model/checkpoint; gallery count, order
      and digest; seed/eval state. Any difference raises `CacheManifestError` naming each field;
    * sentinel + no manifest -> unverified, refused as `not_present`;
    * input symlinks must resolve to the REQUESTED dataset_root entry -- always checked, because a
      moved link is the first step of the failure above;
    * `WORKBENCH_CACHE_MANIFEST_OPT_OUT` restores trust-whatever-is-present for an operator who
      deliberately accepts it, and the run then records `not_verified_opt_out`, so a run can never
      look verified while skipping the check.
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    for name in directory_names:
        source = dataset_root / name
        if not source.exists():
            raise FileNotFoundError(f"native replay input missing: {source}")
        link = cache_root / name
        if not link.exists():
            link.symlink_to(source.resolve(), target_is_directory=source.is_dir())

    mismatches = _symlink_mismatches(cache_root, dataset_root, directory_names)
    if mismatches:
        raise CacheManifestError(cache_root, mismatches)

    warm = (cache_root / CACHE_WARM_SENTINEL).is_file()
    opted_out = _verification_opted_out()
    if opted_out:
        status = "not_verified_opt_out"
        reason = f"{CACHE_VERIFICATION_OPT_OUT_ENV} set: cache contents trusted without verification"
    elif not warm:
        status = "rebuilt"
        reason = "no warm sentinel in the cache root: the constructor builds cold"
    else:
        manifest_mismatches = compare_cache_manifest(cache_root, requirement or {})
        if manifest_mismatches:
            raise CacheManifestError(cache_root, manifest_mismatches)
        status = "verified"
        reason = "manifest matches the run's own description"

    if record is not None:
        record.update({"cache_manifest_status": status,
                       "cache_manifest_reason": reason,
                       "cache_manifest_opted_out": opted_out,
                       # What was actually checked. `annotations_and_gallery` is honest about not
                       # covering the model or the transform.
                       "cache_manifest_scope": requirement_scope,
                       "cache_manifest_checked_fields": sorted(requirement) if requirement else [],
                       **cache_verification_report(cache_root)})
    try:
        yield cache_root
    finally:
        # A COLD construction just wrote tensors into this root. Describe them on the way out,
        # otherwise the next child sees a warm sentinel with no manifest and is refused -- which
        # would make sharing a root impossible rather than safe.
        if status in {"rebuilt", "not_verified_opt_out"} and (cache_root / CACHE_WARM_SENTINEL).is_file():
            if requirement is not None:
                tensors = cache_tensor_digests(cache_root)
                write_cache_manifest(cache_root, {**requirement, "tensors": tensors,
                                                  "cache_files": sorted(tensors),
                                                  "written_by": {"script": "workbench.replay.common",
                                                                 "version": 1,
                                                                 "caller": "cold construction"}})
                if record is not None:
                    record.update(cache_verification_report(cache_root))
            elif record is not None:
                record["cache_manifest_status"] = "rebuilt_unverified"
                record["cache_manifest_reason"] = (
                    "the constructor wrote tensors but the caller supplied no description, so no "
                    "manifest could be written; a later child will refuse this root")


@contextmanager
def isolated_cache_root(dataset_root: Path,
                        directory_names: tuple[str, ...] = ("captions", "image_splits", "resized_image"),
                        *, image_source: str | None = None,
                        image_extension: str = ".jpg") -> Iterator[Path]:
    """Use a fresh temp root, symlinking immutable inputs; optionally expose category images flat."""
    with tempfile.TemporaryDirectory(prefix="cir-replay-cache-") as temporary:
        cache_root = Path(temporary)
        for name in directory_names:
            source = dataset_root / name
            if not source.exists():
                raise FileNotFoundError(f"native replay input missing: {source}")
            (cache_root / name).symlink_to(source.resolve(), target_is_directory=source.is_dir())
        if image_source is not None:
            source_root = dataset_root / image_source
            if not source_root.is_dir():
                raise FileNotFoundError(f"native replay image input missing: {source_root}")
            images = cache_root / "images"
            images.mkdir()
            from workbench.backend.fashioniq_layout import CATEGORIES

            names = []
            if image_source == "resized_image":
                for category in CATEGORIES:
                    split = dataset_root / "image_splits" / f"split.{category}.val.json"
                    if not split.is_file():
                        raise FileNotFoundError(f"native replay split input missing: {split}")
                    names.extend(json.loads(split.read_text(encoding="utf-8")))
                    category_root = source_root / category
                    if not category_root.is_dir():
                        raise FileNotFoundError(f"native replay resized images missing: {category_root}")
                for name in dict.fromkeys(names):
                    if not isinstance(name, str) or not name or "/" in name or "\\" in name:
                        raise ValueError(f"unusable FashionIQ image identifier in split: {name!r}")
                    candidates = [source_root / category / f"{name}{image_extension}" for category in CATEGORIES]
                    source = next((path for path in candidates if path.is_file()), None)
                    if source is None:
                        raise FileNotFoundError(
                            f"native replay resized FashionIQ image missing: {source_root}/<category>/{name}{image_extension}")
                    (images / f"{name}.jpg").symlink_to(source.resolve())
            elif image_source == "images":
                for category in CATEGORIES:
                    split = dataset_root / "image_splits" / f"split.{category}.val.json"
                    if not split.is_file():
                        raise FileNotFoundError(f"native replay split input missing: {split}")
                    names.extend(json.loads(split.read_text(encoding="utf-8")))
                for name in dict.fromkeys(names):
                    if not isinstance(name, str) or not name or "/" in name or "\\" in name:
                        raise ValueError(f"unusable FashionIQ image identifier in split: {name!r}")
                    source = source_root / f"{name}{image_extension}"
                    if not source.is_file():
                        raise FileNotFoundError(f"native replay canonical FashionIQ image missing: {source}")
                    (images / f"{name}.jpg").symlink_to(source.resolve())
            else:
                raise ValueError(f"unsupported native image source: {image_source}")
        yield cache_root


def fashioniq_image_binding_blockers(dataset_root: Path, split: str = "val",
                                     image_source: str = "resized_image",
                                     image_extension: str = ".jpg") -> list[str]:
    """Check that FashionIQ gallery names resolve through TME's flat `images/<id>.jpg` view."""
    from workbench.backend.fashioniq_layout import CATEGORIES

    root = dataset_root / image_source
    blockers = []
    for category in CATEGORIES:
        split_path = dataset_root / "image_splits" / f"split.{category}.{split}.json"
        if not split_path.is_file():
            blockers.append(f"native replay split input missing: {split_path}")
            continue
        if image_source == "resized_image":
            category_root = root / category
            if not category_root.is_dir():
                blockers.append(f"native replay resized images missing: {category_root}")
                continue
        elif not root.is_dir():
            blockers.append(f"native replay canonical images missing: {root}")
            continue
        try:
            names = json.loads(split_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            blockers.append(f"native replay split file unreadable: {split_path}")
            continue
        if not isinstance(names, list):
            blockers.append(f"native replay split is not an image list: {split_path}")
            continue
        for name in names:
            if not isinstance(name, str) or not name or "/" in name or "\\" in name or name in {".", ".."}:
                blockers.append(f"unusable FashionIQ image identifier in {split_path}: {name!r}")
            elif image_source == "resized_image":
                if not any((root / item / f"{name}{image_extension}").is_file() for item in CATEGORIES):
                    blockers.append(f"native replay resized FashionIQ image missing: {root}/<category>/{name}{image_extension}")
            elif not (root / f"{name}{image_extension}").is_file():
                blockers.append(f"native replay canonical FashionIQ image missing: {root}/{name}{image_extension}")
    return blockers


def fashioniq_gallery_digests(dataset_root: Path, split: str, image_extension: str) -> dict[str, str]:
    """Hash actual gallery bytes in native per-category file order."""
    from workbench.backend.fashioniq_layout import CATEGORIES

    result = {}
    for category in CATEGORIES:
        names = json.loads((dataset_root / "image_splits" / f"split.{category}.{split}.json")
                           .read_text(encoding="utf-8"))
        digest = hashlib.sha256()
        for name in names:
            path = dataset_root / "images" / f"{name}{image_extension}"
            encoded = name.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            with path.open("rb") as image:
                for block in iter(lambda: image.read(1 << 20), b""):
                    digest.update(block)
        result[category] = digest.hexdigest()
    return result



def constructor_input_blockers(dataset_root: Path, message: Callable[[Path], str],
                               categories: tuple[str, ...] = CATEGORIES,
                               train_captions: bool = True) -> list[str]:
    """The inputs the native FashionIQ constructor opens, independent of the evaluated categories.

    Every wrapper in this family calls `datasets.FashionIQ(...)`, whose `__init__` runs
    `train_init_process()` and then `get_test_data()` for **all three** categories
    (`HINT/datasets.py:35-47,59-73`) with no category argument anywhere in its signature. So
    `--category dress` selects what is *scored*, never what is *opened*: a root that satisfies the
    dress files alone loads a checkpoint and only then dies on `cap.shirt.train.json`.

    `isolated_cache_root` makes this reachable on every run -- it yields an empty cache root, so
    `fashion_iq_data.json` never exists there and the constructor always takes the cold path. The
    evaluated subset and the constructor's input set are therefore two different questions, and this
    is the second one; the caller keeps asking the first about the categories it will score.

    `message` is the caller's own phrasing for one missing path, so the two questions produce
    identical text for the paths both of them cover and the caller can deduplicate.
    """
    blockers: list[str] = []
    for category in categories:
        required = [f"captions/cap.{category}.val.json",
                    f"captions/correction_dict_{category}.json",
                    f"image_splits/split.{category}.val.json"]
        if train_captions:
            required.append(f"captions/cap.{category}.train.json")
        for relative in required:
            path = dataset_root / relative
            if not path.is_file():
                blockers.append(message(path))
        image_directory = dataset_root / "resized_image" / category
        if not image_directory.is_dir():
            blockers.append(message(image_directory))
    return blockers


def macro_aggregate(results: list[ReplayResult]) -> dict[str, float | None]:
    """Compute source-compatible arithmetic category macro metrics, never query-weighted."""
    if len(results) != len(CATEGORIES) or not all(result.complete for result in results):
        return {"macro_r10": None, "macro_r50": None, "macro_mean": None, "complete": False}
    if tuple(result.category for result in results) != CATEGORIES:
        return {"macro_r10": None, "macro_r50": None, "macro_mean": None, "complete": False}
    if any(result.r10 is None or result.r50 is None for result in results):
        return {"macro_r10": None, "macro_r50": None, "macro_mean": None, "complete": False}
    r10 = sum(result.r10 for result in results) / len(CATEGORIES)
    r50 = sum(result.r50 for result in results) / len(CATEGORIES)
    return {"macro_r10": r10, "macro_r50": r50, "macro_mean": (r10 + r50) / 2, "complete": True}


#: Fixed seed for the dataset constructors whose query pipeline is STOCHASTIC.
#:
#: The iLearn-family constructors bake the transformed query image INTO the dataset at construction
#: time: `datasets.py::get_test_data` calls `get_img(..., stage=0)`, and the replay supplies
#: `transform = [preprocess_train, preprocess_val]`, so the query inputs are drawn through
#: `preprocess_train` -- whose first step is `RandomResizedCrop`. Every construction therefore draws
#: a different crop, and a per-query export that rebuilds the dataset in its own process observes a
#: DIFFERENT input population than the aggregate it is supposed to reconstruct.
#:
#: Measured on this host with the pinned transform and the real annotations: two constructions of the
#: dress category produced 3994 differing query tensors out of 4034, while every gallery input
#: (stage=1, the deterministic validation transform) was byte-identical. That is the whole cause of
#: the ENCODER artifact-versus-aggregate disagreement, and it is a preprocessing-population
#: difference, not the GPU nondeterminism it was previously attributed to: normalization, the dot
#: product and `argsort` all happen in NumPy after the features return to CPU.
#:
#: Seeding global RNG immediately before construction makes each construction reproduce the same
#: draws. It must be applied at the construction site and not merely at process start, because the
#: constructor consumes randomness from the same global generators that a model load or a CUDA
#: context can also touch first.
DATASET_CONSTRUCTION_SEED = 0


@contextmanager
def seeded_dataset_construction(seed: int = DATASET_CONSTRUCTION_SEED) -> Iterator[None]:
    """Make a stochastic dataset construction reproducible across processes.

    Seed state is saved and restored, so seeding a construction does not leak into whatever the
    caller does next (a differently seeded evaluation, or a caller that seeded deliberately).
    """
    import random

    python_state = random.getstate()
    numpy_state = None
    try:
        import numpy as np

        numpy_state = np.random.get_state()
    except Exception:
        np = None  # type: ignore[assignment]

    torch_state = None
    try:
        import torch

        torch_state = torch.random.get_rng_state()
    except Exception:
        torch = None  # type: ignore[assignment]

    random.seed(seed)
    if numpy_state is not None:
        np.random.seed(seed)
    if torch_state is not None:
        torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        if numpy_state is not None:
            np.random.set_state(numpy_state)
        if torch_state is not None:
            torch.random.set_rng_state(torch_state)


def enable_deterministic_inference(torch) -> None:
    """Make a CUDA forward pass reproducible across processes.

    The per-query observer re-derives the ranking from features it computes ITSELF, with the same
    calls and the same batching the native `test()` makes. Those are two separate forward passes, and
    cuDNN's default reduction order is not fixed, so the two disagreed on a handful of queries out of
    six thousand even once both saw byte-identical inputs (measured: 5 differing hit counts across
    four category Recall values). A ranking that depends on kernel scheduling is not reproducible
    evidence, and the artifact-versus-aggregate check is right to refuse it.

    This does not change what the protocol computes -- normalization, the similarity dot product,
    `argsort` and the metric arithmetic are unchanged -- only the order in which the same additions
    are performed. It costs speed, which is the price of an artifact that can be reproduced.

    `CUBLAS_WORKSPACE_CONFIG` is required by cuBLAS for deterministic reductions on some versions and
    must be set before the handle is created, so it is set here as well as the torch flags.
    """
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass


class FeatureRecorder:
    """Capture the features a native evaluation actually computes, so an observation need not
    recompute them.

    The per-query observer and the native `test()` must reconstruct the SAME rankings. When the
    observer makes its own forward pass, it cannot: CUDA feature extraction is not reproducible across
    processes even with cuDNN pinned deterministic and a fixed seed. Measured on the pinned ENCODER
    model, the same input and seed produced different feature hashes in two processes
    (`82dc55fc…` vs `af784fa2…`), so a second pass can differ by a hit or two out of six thousand no
    matter how carefully it is configured.

    Recording from the actual call removes the second pass entirely: the observation is derived from
    the very tensors the native metric was computed on, which is what the per-query export claims to
    describe. It is a wrapper around the model's own methods, so preprocessing, batching, dtype and
    normalization are unchanged.
    """

    def __init__(self, model):
        self.model = model
        self.query_batches: list = []
        self.target_batches: list = []
        self._original_compose = None
        self._original_target = None

    def __enter__(self):
        self._original_compose = self.model.extract_retrieval_compose
        self._original_target = self.model.extract_retrieval_target

        def compose(imgs, mods, *args, **kwargs):
            out = self._original_compose(imgs, mods, *args, **kwargs)
            self.query_batches.append(out)
            return out

        def target(imgs, *args, **kwargs):
            out = self._original_target(imgs, *args, **kwargs)
            self.target_batches.append(out)
            return out

        self.model.extract_retrieval_compose = compose
        self.model.extract_retrieval_target = target
        return self

    def __exit__(self, *exc):
        if self._original_compose is not None:
            self.model.extract_retrieval_compose = self._original_compose
        if self._original_target is not None:
            self.model.extract_retrieval_target = self._original_target
        return False

    def features(self):
        """The captured batches as two arrays, or None when the native pass did not run."""
        import numpy as np

        if not self.query_batches or not self.target_batches:
            return None, None
        queries = np.concatenate([b.data.cpu().numpy() if hasattr(b, "data") else np.asarray(b)
                                  for b in self.query_batches])
        targets = np.concatenate([b.data.cpu().numpy() if hasattr(b, "data") else np.asarray(b)
                                  for b in self.target_batches])
        return queries, targets

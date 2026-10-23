"""Shared model and dataset readiness requirements for operator entry points."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def model_requirements(
    model: dict,
    config: Any,
    *,
    dataset_root: Path | None = None,
    canonical_root: Path | None = None,
) -> dict[str, Any]:
    """Resolve data, preparation, external asset, and fixed-link requirements."""
    from workbench.backend.fashioniq_layout import image_extension_problems, missing_paths
    from workbench.backend.registry import (
        auxiliary_assets_for_model, auxiliary_destination, caption_alias_evidence,
        external_asset_statuses,
        fashioniq_required_paths, preparation_contract_for_model, preparation_paths,
        validate_auxiliary_asset,
    )

    root = Path(dataset_root) if dataset_root else config.FASHIONIQ_ROOT
    canonical_root = Path(canonical_root) if canonical_root is not None else config.FASHIONIQ_ROOT
    source = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"] if model.get("source_dir") else None
    contract = preparation_contract_for_model(model)
    raw = preparation_paths(contract, root, "raw_inputs") if contract else fashioniq_required_paths(model, root)
    generated = preparation_paths(contract, root, "generated_artifacts", evaluator_root=source) if contract else ()
    paths = list(dict.fromkeys((*raw, *generated)))
    raw_missing, generated_missing = missing_paths(raw), missing_paths(generated)
    blocks: list[str] = []
    # Non-blocking statements the caller may surface: today only a deliberately staged caption alias,
    # which allows a run to proceed while keeping the variant visible.
    notes: list[str] = []
    layout = model.get("fashioniq_layout") or "FashionIQ"
    if contract:
        blocks.extend(f"raw benchmark input missing: {path}" for path in raw_missing)
        blocks.extend(f"required generated artifact missing: {path}" for path in generated_missing)
    else:
        blocks.extend(f"FashionIQ {layout} requirement missing: {path}" for path in raw_missing)
    image_problems = [problem for extension in model.get("fashioniq_image_extensions") or ()
                      for problem in image_extension_problems(root, extension, model.get("fashioniq_layout"))]
    blocks.extend(f"FashionIQ image extension requirement: {problem}" for problem in image_problems)

    # A checkout repaired by an approved patch is an approved variant, not official parity: the run
    # executed code the authors did not publish. Say so where a reader already looks for variant notes.
    if model.get("source_blocker"):
        repair = approved_patch_state(model, source)
        if repair is not None and repair["applied"]:
            applied = ", ".join(str(state["patch_file"]) for state in repair["patches"])
            note = (f"source repaired by the approved patch set [{applied}] "
                    f"(sha256={repair['patch_sha256']}); the published-source blocker no longer applies "
                    f"and the run is an approved compatibility variant, not official parity")
            notes.append(note)
        elif repair is not None:
            # A half-repaired checkout is neither the published source nor the approved variant, and
            # saying nothing here is how a partial repair came to look complete: the blocker text
            # alone does not say how much of the set is missing.
            missing = [state["path"] for state in repair["patches"] if not state["applied"]]
            notes.append(
                f"source repair incomplete: {len(missing)} of {len(repair['patches'])} required "
                f"patches are not applied ({', '.join(missing)}); {repair['detail']}")

    auxiliary = []
    for asset in auxiliary_assets_for_model(model["model_id"]):
        path = auxiliary_destination(asset, config.WORKBENCH_THIRD_PARTY_ROOT, canonical_root)
        paths.append(path)
        res = validate_auxiliary_asset(asset, path, fashioniq_root=canonical_root)
        missing = res.status == "MISSING"
        auxiliary.append({"name": asset["family"], "path": str(path), "missing": missing, "status": res.status, "issues": res.issues})
        prefix = "CSMCIR " if model["model_id"] == "csmcir" else ""
        if missing:
            blocks.append(f"{prefix}{asset['family']} missing: {path}")
            if asset["family"] == "COT_ours2 captions":
                blocks.append("required COT_ours2 captions have no verified automatic acquisition source")
        elif not res.ok:
            if res.status == "OPERATOR_SUPPLIED_UNVERIFIED":
                # An unverified asset that the workbench itself staged from a declared alias is a
                # deliberate, labelled variant, not a stray file: the run asked for it and the
                # manifest records which published bytes went in. It is accepted so the variant can
                # be evaluated, and reported as a variant so no result is mistaken for official
                # parity. Anything else at that path is still a blocker.
                alias = caption_alias_evidence(asset, path, config)
                if alias is not None:
                    auxiliary[-1]["caption_alias"] = alias
                    notes.append(f"{prefix}{asset['family']} staged as a source-compatible caption alias: "
                                 f"{alias['alias_source_path']}@{alias['source_revision']} "
                                 f"sha256={alias['staged_sha256']} — not official parity")
                else:
                    blocks.append(f"{prefix}{asset['family']} unverified: {path} (operator-supplied unverified provenance)")
            else:
                for issue in res.issues:
                    blocks.append(f"{prefix}{asset['family']} invalid: {issue}")
    link = model.get("canonical_dataset_link")
    link_evidence = None
    if link and source is not None:
        link_path = source / link
        actual = link_path.resolve() if link_path.is_symlink() else None
        expected = canonical_root.resolve()
        paths.append(link_path)
        link_evidence = {"path": str(link_path), "expected": str(expected),
                         "actual": str(actual) if actual is not None else None}
        if not link_path.is_symlink():
            blocks.append(f"canonical dataset link missing: {link_path} -> {expected}")
        elif actual != expected:
            blocks.append(f"canonical dataset link resolves to {actual}, expected {expected}")
        elif not actual.is_dir():
            blocks.append(f"canonical dataset link target missing: {actual}")
    if model["model_id"] == "encoder" and source is not None:
        paths.extend((source / "datasets1.py", source / "open_clip_pytorch_model.bin"))

    asset_statuses = external_asset_statuses(model)
    paths.extend(Path(item["path"]) for item in asset_statuses if item["path"])
    messages = {"UNRESOLVED_PROVENANCE": "provenance unresolved", "UNCONFIGURED_LOCATION": "local location unconfigured",
                "CONFIGURED_MISSING": "file missing", "INVALID_FILE_TYPE": "expected regular file, found non-file",
                "EMPTY_PLACEHOLDER": "empty placeholder file", "CHECKSUM_MISMATCH": "checksum mismatch",
                "EXISTS_UNVERIFIED_CHECKSUM": "file exists but no verified checksum is available"}
    blocks.extend(f"external runtime asset {messages[item['status']]}: {item['name']}" +
                  (f" ({item['path']})" if item["path"] else "")
                  for item in asset_statuses if item["status"] != "VERIFIED_LOCAL_ASSET")
    return {"paths": list(dict.fromkeys(paths)), "blocks": list(dict.fromkeys(blocks)),
            "warnings": list(dict.fromkeys(notes)),
            "evidence": {"raw_inputs_missing": [str(path) for path in raw_missing],
                         "generated_artifacts_missing": [str(path) for path in generated_missing],
                         "image_extension_problems": image_problems, "auxiliary_assets": auxiliary,
                         "dataset_link": link_evidence}}


def approved_patch_state(model: dict, source: Path | None) -> dict[str, Any] | None:
    """Whether EVERY declared approved patch is applied to the checkout at *source*.

    Returns `None` when the model declares no approved patch. Otherwise `applied` says whether the
    checkout carries the recorded post-patch digest for every required patch. `detail` explains a
    checkout that is neither the published original nor the full approved variant, and it names the
    patch that is missing rather than only the first one — a model may declare several required
    repairs (CSMCIR declares an import repair and a diagnostic-dump repair), and stopping at the first
    one reported a half-repaired checkout as fully approved.

    `patches` lists each declaration with its own state, in declaration order, so a consumer can see
    which member of the set is missing without re-hashing the tree.
    """
    patches = model.get("approved_source_patches") or []
    if not patches:
        return None
    if source is None or not source.is_dir():
        return {"applied": False, "patch_file": None, "patch_sha256": None, "patches": [],
                "detail": "the source checkout is not present, so no patch state can be observed"}

    from workbench.backend.registry import sha256_file

    states: list[dict[str, Any]] = []
    for patch in patches:
        state = {"path": patch["path"], "patch_file": patch.get("patch_file"),
                 "patch_sha256": patch.get("patch_sha256")}
        target = source / patch["path"]
        base, derived = patch.get("base_sha256"), patch.get("derived_sha256")
        if base is not None and base == derived:
            # A declaration whose two digests are equal states no delta at all: the published file
            # would be indistinguishable from the repaired one. That is a registry error, and
            # treating the base digest as "applied" would report every pristine checkout as repaired.
            state.update(applied=False, detail=(
                f"{patch['path']} declares the same base and derived digest "
                f"({str(base)[:16]}...), so the declaration proves nothing about this checkout; fix "
                "the registry entry before trusting the patch state"))
        elif target.is_symlink():
            # `sha256_file` refuses symlinks (they are how an artifact is smuggled past a digest
            # check), and a symlinked target is not the published file under any reading. Report it
            # rather than letting the ValueError escape a readiness report.
            state.update(applied=False, detail=(
                f"{patch['path']} is a symlink; an approved patch target must be the real file"))
        elif not target.is_file():
            state.update(applied=False, detail=f"the target file is missing: {target}")
        else:
            digest = sha256_file(target)
            if digest == derived:
                state.update(applied=True,
                             detail=f"{patch['path']} matches the recorded post-patch digest")
            elif digest == base:
                state.update(applied=False, detail=(
                    f"{patch['path']} is the unmodified published file; apply "
                    f"{patch.get('patch_file')} to a derived checkout of the pinned revision and "
                    f"register that checkout as the model source"))
            else:
                state.update(applied=False, detail=(
                    f"{patch['path']} matches neither the published file nor the approved variant "
                    f"(sha256={digest[:16]}...); an unexpected modification is not covered by the "
                    f"approved patch"))
        states.append(state)

    missing = [state for state in states if not state["applied"]]
    summary = {
        "applied": not missing,
        "patch_file": (missing[0]["patch_file"] if missing
                       else ", ".join(str(state["patch_file"]) for state in states)),
        "patch_sha256": (missing[0]["patch_sha256"] if missing
                         else ", ".join(str(state["patch_sha256"]) for state in states)),
        "patches": states,
        "detail": (missing[0]["detail"] if missing else "; ".join(
            f"{state['path']} matches the recorded post-patch digest" for state in states)),
    }
    if missing:
        summary["detail"] = (
            f"{len(missing)} of {len(states)} required patches are not applied. "
            f"{missing[0]['detail']}")
    return summary


def preflight_blockers(
    model: dict,
    config: Any,
    *,
    dataset_root: Path | None = None,
    canonical_root: Path | None = None,
) -> list[str]:
    """Single shared blocker list for model data and runtime prerequisites."""
    from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter
    from workbench.backend.runtime import environment_blockers, source_clean_and_pinned

    requirements = model_requirements(model, config, dataset_root=dataset_root, canonical_root=canonical_root)
    blockers = list(requirements["blocks"])
    model_id = model["model_id"]
    source = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"] if model.get("source_dir") else None
    if not model.get("source_available") or source is None or not source.is_dir():
        blockers.append("upstream source unavailable")
    else:
        clean, _, reason = source_clean_and_pinned(model, source)
        if not clean:
            blockers.append(reason or "source unavailable")

    adapter = ADAPTERS.get(model_id)
    command = getattr(adapter, "command", None) if adapter is not None else None
    audited = (command is not None and command is not OfficialScriptAdapter.command and
               model.get("command_status") in {None, "COMMAND_AUDITED", "REPLAY_COMMAND_IMPLEMENTED"})
    # A model may declare a defect in its upstream source that no amount of local preparation can
    # fix. Reporting it first matters: for CSMCIR the caption and checkpoint blockers are real but
    # downstream of a package that cannot even be imported, and an operator reading only the first
    # line would go looking for captions instead of at the source.
    #
    # The blocker is a statement about the *published* source, so it is reported only while the
    # checkout is in that state. When the model declares an approved patch and the checkout holds
    # exactly that delta, the blocker is satisfied: a repaired checkout must not be blocked by a claim
    # about its unrepaired original. `model_requirements` reports the repair as a warning so the
    # variant stays visible; this list carries blockers only.
    declared_source_blocker = model.get("source_blocker")
    if declared_source_blocker:
        repair = approved_patch_state(model, source)
        if repair is None:
            blockers.insert(0, f"source cannot run: {declared_source_blocker}")
        elif not repair["applied"]:
            blockers.insert(0, f"source cannot run: {declared_source_blocker}; {repair['detail']}")
    if not audited:
        blockers.append(f"command not runnable: {model.get('command_status') or 'COMMAND_UNAUDITED'}")
    elif source is not None and getattr(adapter, "script", None) and not (source / adapter.script).is_file():
        blockers.append(f"official evaluator missing: {source / adapter.script}")

    if model_id == "dcnet" and source is not None:
        # The official evaluator is launched as `test.py --resume <run_dir>` with `cwd` set by
        # `source_cwd` (`evaluate_models.py:214-215`), which is the checkout for every model except
        # csmcir -- and it takes its dataset from the config's own `data_loader.args.data_dir` rather
        # than from the dataset root the operator passed. Nothing else on this path compares the two.
        #
        # Both arguments matter, and leaving either implicit is how this check was absent from three
        # of the four entry points:
        #   * the evaluator's working directory is the checkout, not this process's cwd, so a
        #     correctly staged `dataset/fashioniq` symlink resolves to the dataset root;
        #   * a caller that passes no dataset root (`pc_readiness`, `doctor`, `prepare_model`) still
        #     gets the comparison, against the canonical root those callers treat as the target.
        from workbench.backend.registry import checkpoint_path
        from workbench.replay.dcnet import binding_mismatch

        requested = Path(dataset_root) if dataset_root is not None else Path(config.FASHIONIQ_ROOT)
        # Every declared run directory, not just the first: each carries its own config, and a
        # registry with two variants would otherwise have one of them unchecked.
        for variant in model.get("checkpoint_variants") or []:
            mismatch = binding_mismatch(
                checkpoint_path(model["model_id"], variant, config.WORKBENCH_CHECKPOINT_ROOT),
                requested, evaluator_cwd=source)
            if mismatch:
                blockers.append(f"{variant.get('checkpoint_id')}: {mismatch}")

    blockers.extend(environment_blockers(model))
    if model_id == "encoder" and source is not None:
        if not (source / "datasets1.py").is_file():
            blockers.append(f"ENCODER evaluator import missing: {source / 'datasets1.py'}")
        if not (source / "open_clip_pytorch_model.bin").is_file():
            blockers.append(f"ENCODER asset missing: {source / 'open_clip_pytorch_model.bin'}")
    return list(dict.fromkeys(blockers))

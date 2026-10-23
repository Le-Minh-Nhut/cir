from __future__ import annotations

from pathlib import Path

from workbench.backend.adapters.base import EvalRequest, ModelAdapter
from workbench.backend.errors import WorkbenchError
from workbench.backend.registry import checkpoint_path, checkpoint_member_root
from workbench.backend import runtime as _runtime

REPO_ROOT = Path(__file__).resolve().parents[3]


class OfficialScriptAdapter(ModelAdapter):
    script: str
    #: Set when the adapter launches a workbench replay wrapper instead of `script`.
    #:
    #: `script` stays the *upstream* file whose absence means the checkout is broken, so the
    #: integrity check keeps meaning what it says. But `script` is not always the program that
    #: runs: several pinned evaluators ship only `test(params, model, testset, category)` with no
    #: `__main__`, so the wrapper is the real entry point. Consumers that reason about *which
    #: programs run* -- `environment_planner._evaluator_imports` is the one that matters -- have to
    #: read that from a declaration rather than guess it from a command list, because guessing is
    #: how the dependency profile came to describe upstream training code instead of the wrapper's
    #: real imports.
    launcher: str = "script"
    #: Extra source roots, relative to the checkout, that the launcher resolves modules against.
    launcher_roots: tuple[str, ...] = ()

    def build_command(self, request: EvalRequest) -> list[str]:
        from workbench.backend.operator_config import resolve_config

        checkpoint = self.validate_request(request)
        config = resolve_config()
        return self.command(
            config.WORKBENCH_THIRD_PARTY_ROOT / self.model["source_dir"],
            checkpoint_path(self.model_id, checkpoint, config.WORKBENCH_CHECKPOINT_ROOT),
            request,
        )

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        raise NotImplementedError("upstream evaluation command requires audit before execution")


def _replay_command(adapter: OfficialScriptAdapter, checkpoint: Path, request: EvalRequest,
                    source: Path, *, variant: str | None = None,
                    checkpoint_id: bool = False) -> list[str]:
    """Build the command for a model whose only runnable entry point is a workbench replay wrapper.

    These evaluators have no CLI of their own: upstream ships `test(params, model, testset, category)`
    as a bare callable, so there is nothing to launch. The wrapper is the missing shell and calls the
    pinned evaluator unchanged, which is why the adapter points at it rather than at `test.py`.

    `--checkpoint-root` receives the directory the members occupy rather than the resolved file, so the
    shape is decided by the same rule every other consumer uses (`checkpoint_member_root`): a single
    artifact resolves to its own directory, a multi-member bundle to the directory holding its
    members. Hard-coding `.parent` here would break silently the moment a checkpoint became a bundle.
    """
    replayer = REPO_ROOT / "workbench" / "replay" / f"{adapter.model_id}.py"
    command = [
        _runtime.python_executable(adapter.model),
        str(replayer),
        "--checkpoint-root", str(checkpoint_member_root(checkpoint)),
        "--dataset-root", str(request.dataset_root),
        "--source-root", str(source),
    ]
    if variant is not None:
        command.extend(("--variant", variant))
    if checkpoint_id and request.checkpoint_id:
        command.extend(("--checkpoint-id", request.checkpoint_id))
    return command


class CSMCIRAdapter(OfficialScriptAdapter):
    model_id = "csmcir"
    script = "src/validate_blip_csmcir.py"
    launcher = "script"

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        return [_runtime.python_executable(self.model), str(source / self.script), "--dataset", "fashionIQ", "--blip-model-path", str(checkpoint)]


class EncoderAdapter(OfficialScriptAdapter):
    model_id = "encoder"
    script = "evaluate_model.py"
    launcher = "replay"
    launcher_roots = ()

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        # `evaluate_model.py` is importable but is not a script: it reads `args` at module scope and
        # its loader passes an arbitrary local file to `load_state_dict`, so what a run must actually
        # establish — which keys loaded and which did not — is only reported by the replay wrapper.
        # The official command is kept here, unmodified, as the fallback path.
        return _replay_command(self, checkpoint, request, source, checkpoint_id=True)


class HintAdapter(OfficialScriptAdapter):
    model_id = "hint"
    script = "test.py"
    launcher = "replay"
    launcher_roots = ("lavis",)
    # `workbench/replay/lavis_processors.py:52` builds `lavis.processors` as a skeleton whose
    # `__path__` is the real directory and inserts it before any submodule import, so the pinned
    # `processors/__init__.py` never runs -- that is verified by running the wrapper and reading
    # `sys.modules`. `lavis.models` is deliberately NOT stubbed: it is attempted, and once its
    # dependencies are present it really is executed, so its transitive imports are requirements.
    # Stubbing it looked plausible and was wrong -- it dropped `peft`, `diffusers`, `easydict`,
    # `matplotlib` and `nltk`, and an environment provisioned from that plan could not unpickle the
    # checkpoint classes those very re-exports install.
    stubbed_packages = ("lavis.processors",)
    # `__import__("datasets")` and `importlib.import_module("test")` resolve to the checkout's own
    # top-level modules, which the launcher puts on `sys.path`. Neither is a PyPI distribution.
    provided_modules = ("datasets", "test", "data_utils", "utils")
    # Names a text scan reaches but the evaluation does not load. The earlier version of this list
    # was taken from a census that called `importlib.import_module("lavis.models")` -- an import that
    # FAILS PARTWAY when a dependency is missing, so every module needed after the failure point was
    # never counted and five of these were classified from an incomplete import. `sklearn` and
    # `matplotlib` in particular are imported at module scope by native class files
    # (`HABIT.py:24-27`, `INTENT/.../blip2_cir.py:14,30`, `ConeSep.py:24`,
    # `AirKnow/.../blip2_cir.py:19`), so they are REQUIRED, not unreachable, and excluding them made
    # a clean install unable to import the checkpoint class at all.
    #
    # What remains excluded is genuinely out of reach on this path: `torchaudio` and `knn_cuda` sit
    # behind `if torch.cuda.is_available():` (`ulip_models/pointbert/dvae.py:6-7`) or in the audio
    # encoder only a video task imports (`beats/BEATs.py:14`); `termcolor` is reached through the
    # same unused PointBERT module; `nltk` and `h5py` through unused instruction/ULIP processors.
    unreachable_modules = ("torchaudio", "knn_cuda", "termcolor", "nltk", "h5py")

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        # The pinned evaluator has no `__main__`, no argument parser and no checkpoint loader, so
        # `test.py` cannot be launched at all. `workbench/replay/hint.py` is the missing shell and
        # calls the pinned `test()` and the pinned `datasets.FashionIQ` unchanged; the single
        # checkpoint is shared across the three categories (`category_model_policy: shared`).
        return [
            _runtime.python_executable(self.model),
            str(Path(__file__).resolve().parents[3] / "workbench" / "replay" / "hint.py"),
            # Shared member-root rule, applied to the already-resolved path: HINT's artifact is a
            # single file, so `checkpoint_member_root` returns its own directory and the wrapper
            # looks the member up by name. Hard-coding `.parent` would silently break if this
            # checkpoint ever became a directory artifact.
            "--checkpoint-root", str(checkpoint_member_root(checkpoint)),
            "--dataset-root", str(request.dataset_root),
            "--source-root", str(source),
        ]


class PairAdapter(OfficialScriptAdapter):
    model_id = "pair"
    script = "test.py"
    launcher = "replay"
    launcher_roots = ()

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        # `test.py:6` is `test(params, model, testset, category)` — a callable with no CLI, so the
        # wrapper supplies the missing shell and calls it unchanged.
        return _replay_command(self, checkpoint, request, source, checkpoint_id=True)


class AirKnowAdapter(OfficialScriptAdapter):
    model_id = "airknow"
    script = "test_BLIP2.py"
    launcher = "replay"
    launcher_roots = ("lavis",)

    # `workbench/replay/lavis_processors.py:52` builds `lavis.processors` as a skeleton whose
    # `__path__` is the real directory and inserts it before any submodule import, so the pinned
    # `processors/__init__.py` never runs -- that is verified by running the wrapper and reading
    # `sys.modules`. `lavis.models` is deliberately NOT stubbed: it is attempted, and once its
    # dependencies are present it really is executed, so its transitive imports are requirements.
    # Stubbing it looked plausible and was wrong -- it dropped `peft`, `diffusers`, `easydict`,
    # `matplotlib` and `nltk`, and an environment provisioned from that plan could not unpickle the
    # checkpoint classes those very re-exports install.
    stubbed_packages = ("lavis.processors",)
    # `__import__("datasets")` and `importlib.import_module("test")` resolve to the checkout's own
    # top-level modules, which the launcher puts on `sys.path`. Neither is a PyPI distribution.
    provided_modules = ("datasets", "test", "data_utils", "utils")
    # Names a text scan reaches but the evaluation does not load. The earlier version of this list
    # was taken from a census that called `importlib.import_module("lavis.models")` -- an import that
    # FAILS PARTWAY when a dependency is missing, so every module needed after the failure point was
    # never counted and five of these were classified from an incomplete import. `sklearn` and
    # `matplotlib` in particular are imported at module scope by native class files
    # (`HABIT.py:24-27`, `INTENT/.../blip2_cir.py:14,30`, `ConeSep.py:24`,
    # `AirKnow/.../blip2_cir.py:19`), so they are REQUIRED, not unreachable, and excluding them made
    # a clean install unable to import the checkpoint class at all.
    #
    # What remains excluded is genuinely out of reach on this path: `torchaudio` and `knn_cuda` sit
    # behind `if torch.cuda.is_available():` (`ulip_models/pointbert/dvae.py:6-7`) or in the audio
    # encoder only a video task imports (`beats/BEATs.py:14`); `termcolor` is reached through the
    # same unused PointBERT module; `nltk` and `h5py` through unused instruction/ULIP processors.
    unreachable_modules = ("torchaudio", "knn_cuda", "termcolor", "nltk", "h5py")

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        # `test_BLIP2.py` has no `__main__` and no checkpoint loader, and one of its branches calls
        # `load_obj("")` with an empty path, so it cannot be launched as published.
        #
        # The variant is passed as `--variant`, because that is the flag the wrapper actually
        # accepts (`airknow.py:add_argument("--variant")`). It used to be passed as
        # `--checkpoint-id`, which no wrapper argument matches: argparse exits 2 with
        # "unrecognized arguments" before any model is touched, so the model could not run at all.
        # The value is the request's selection, so the variant the caller chose is the variant
        # evaluated — not a default that happens to be first in the registry.
        return _replay_command(self, checkpoint, request, source,
                               variant=(request.checkpoint_id or None))


class ConeSepAdapter(OfficialScriptAdapter):
    model_id = "conesep"
    script = "test.py"
    launcher = "replay"
    launcher_roots = ("lavis",)

    # `workbench/replay/lavis_processors.py:52` builds `lavis.processors` as a skeleton whose
    # `__path__` is the real directory and inserts it before any submodule import, so the pinned
    # `processors/__init__.py` never runs -- that is verified by running the wrapper and reading
    # `sys.modules`. `lavis.models` is deliberately NOT stubbed: it is attempted, and once its
    # dependencies are present it really is executed, so its transitive imports are requirements.
    # Stubbing it looked plausible and was wrong -- it dropped `peft`, `diffusers`, `easydict`,
    # `matplotlib` and `nltk`, and an environment provisioned from that plan could not unpickle the
    # checkpoint classes those very re-exports install.
    stubbed_packages = ("lavis.processors",)
    # `__import__("datasets")` and `importlib.import_module("test")` resolve to the checkout's own
    # top-level modules, which the launcher puts on `sys.path`. Neither is a PyPI distribution.
    provided_modules = ("datasets", "test", "data_utils", "utils")
    # Names a text scan reaches but the evaluation does not load. The earlier version of this list
    # was taken from a census that called `importlib.import_module("lavis.models")` -- an import that
    # FAILS PARTWAY when a dependency is missing, so every module needed after the failure point was
    # never counted and five of these were classified from an incomplete import. `sklearn` and
    # `matplotlib` in particular are imported at module scope by native class files
    # (`HABIT.py:24-27`, `INTENT/.../blip2_cir.py:14,30`, `ConeSep.py:24`,
    # `AirKnow/.../blip2_cir.py:19`), so they are REQUIRED, not unreachable, and excluding them made
    # a clean install unable to import the checkpoint class at all.
    #
    # What remains excluded is genuinely out of reach on this path: `torchaudio` and `knn_cuda` sit
    # behind `if torch.cuda.is_available():` (`ulip_models/pointbert/dvae.py:6-7`) or in the audio
    # encoder only a video task imports (`beats/BEATs.py:14`); `termcolor` is reached through the
    # same unused PointBERT module; `nltk` and `h5py` through unused instruction/ULIP processors.
    unreachable_modules = ("torchaudio", "knn_cuda", "termcolor", "nltk", "h5py")

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        # Single checkpoint shared across the three categories, and a callable-only evaluator: the
        # `--variant` flag records WHICH published noise level this run evaluates, because the Hub
        # publishes 20/50/80 % variants and no clean one.
        return _replay_command(self, checkpoint, request, source,
                               variant=(request.checkpoint_id or None))


class HabitAdapter(OfficialScriptAdapter):
    model_id = "habit"
    script = "test.py"
    launcher = "replay"
    launcher_roots = ("lavis",)

    # `workbench/replay/lavis_processors.py:52` builds `lavis.processors` as a skeleton whose
    # `__path__` is the real directory and inserts it before any submodule import, so the pinned
    # `processors/__init__.py` never runs -- that is verified by running the wrapper and reading
    # `sys.modules`. `lavis.models` is deliberately NOT stubbed: it is attempted, and once its
    # dependencies are present it really is executed, so its transitive imports are requirements.
    # Stubbing it looked plausible and was wrong -- it dropped `peft`, `diffusers`, `easydict`,
    # `matplotlib` and `nltk`, and an environment provisioned from that plan could not unpickle the
    # checkpoint classes those very re-exports install.
    stubbed_packages = ("lavis.processors",)
    # `__import__("datasets")` and `importlib.import_module("test")` resolve to the checkout's own
    # top-level modules, which the launcher puts on `sys.path`. Neither is a PyPI distribution.
    provided_modules = ("datasets", "test", "data_utils", "utils")
    # Names a text scan reaches but the evaluation does not load. The earlier version of this list
    # was taken from a census that called `importlib.import_module("lavis.models")` -- an import that
    # FAILS PARTWAY when a dependency is missing, so every module needed after the failure point was
    # never counted and five of these were classified from an incomplete import. `sklearn` and
    # `matplotlib` in particular are imported at module scope by native class files
    # (`HABIT.py:24-27`, `INTENT/.../blip2_cir.py:14,30`, `ConeSep.py:24`,
    # `AirKnow/.../blip2_cir.py:19`), so they are REQUIRED, not unreachable, and excluding them made
    # a clean install unable to import the checkpoint class at all.
    #
    # What remains excluded is genuinely out of reach on this path: `torchaudio` and `knn_cuda` sit
    # behind `if torch.cuda.is_available():` (`ulip_models/pointbert/dvae.py:6-7`) or in the audio
    # encoder only a video task imports (`beats/BEATs.py:14`); `termcolor` is reached through the
    # same unused PointBERT module; `nltk` and `h5py` through unused instruction/ULIP processors.
    unreachable_modules = ("torchaudio", "knn_cuda", "termcolor", "nltk", "h5py")

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        # As with conesep: the selected variant is recorded so a noisy-trained checkpoint can never be
        # presented as a clean-trained one.
        return _replay_command(self, checkpoint, request, source,
                               variant=(request.checkpoint_id or None))


class IntentAdapter(OfficialScriptAdapter):
    model_id = "intent"
    script = "test.py"
    launcher = "replay"
    launcher_roots = ("lavis",)

    # `workbench/replay/lavis_processors.py:52` builds `lavis.processors` as a skeleton whose
    # `__path__` is the real directory and inserts it before any submodule import, so the pinned
    # `processors/__init__.py` never runs -- that is verified by running the wrapper and reading
    # `sys.modules`. `lavis.models` is deliberately NOT stubbed: it is attempted, and once its
    # dependencies are present it really is executed, so its transitive imports are requirements.
    # Stubbing it looked plausible and was wrong -- it dropped `peft`, `diffusers`, `easydict`,
    # `matplotlib` and `nltk`, and an environment provisioned from that plan could not unpickle the
    # checkpoint classes those very re-exports install.
    stubbed_packages = ("lavis.processors",)
    # `__import__("datasets")` and `importlib.import_module("test")` resolve to the checkout's own
    # top-level modules, which the launcher puts on `sys.path`. Neither is a PyPI distribution.
    provided_modules = ("datasets", "test", "data_utils", "utils")
    # Names a text scan reaches but the evaluation does not load. The earlier version of this list
    # was taken from a census that called `importlib.import_module("lavis.models")` -- an import that
    # FAILS PARTWAY when a dependency is missing, so every module needed after the failure point was
    # never counted and five of these were classified from an incomplete import. `sklearn` and
    # `matplotlib` in particular are imported at module scope by native class files
    # (`HABIT.py:24-27`, `INTENT/.../blip2_cir.py:14,30`, `ConeSep.py:24`,
    # `AirKnow/.../blip2_cir.py:19`), so they are REQUIRED, not unreachable, and excluding them made
    # a clean install unable to import the checkpoint class at all.
    #
    # What remains excluded is genuinely out of reach on this path: `torchaudio` and `knn_cuda` sit
    # behind `if torch.cuda.is_available():` (`ulip_models/pointbert/dvae.py:6-7`) or in the audio
    # encoder only a video task imports (`beats/BEATs.py:14`); `termcolor` is reached through the
    # same unused PointBERT module; `nltk` and `h5py` through unused instruction/ULIP processors.
    unreachable_modules = ("torchaudio", "knn_cuda", "termcolor", "nltk", "h5py")

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        return _replay_command(self, checkpoint, request, source,
                               variant=(request.checkpoint_id or None))


class CLVCNetAdapter(OfficialScriptAdapter):
    model_id = "clvc_net"
    script = "test.py"
    launcher = "replay"
    launcher_roots = ()

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        # `test.py:6` is `test(params, local_model, global_model, testset, dataname)`: this evaluator
        # needs TWO whole models per category (six files), which the wrapper loads by name.
        return _replay_command(self, checkpoint, request, source)


class DCNetAdapter(OfficialScriptAdapter):
    model_id = "dcnet"
    script = "test.py"
    launcher = "script"

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        return [_runtime.python_executable(self.model), str(source / self.script), "--resume", str(checkpoint)]

class CombinerNoftAdapter(OfficialScriptAdapter):
    model_id = "combiner_rn50x4_noft"
    script = "src/validate.py"
    launcher = "script"

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        return [_runtime.python_executable(self.model), str(source / self.script), "--dataset", "fashionIQ",
                "--combining-function", "combiner", "--combiner-path", str(checkpoint),
                "--projection-dim", "2560", "--hidden-dim", "5120",
                "--clip-model-name", "RN50x4", "--target-ratio", "1.25", "--transform", "targetpad"]


class CLIP4CirFullftAdapter(OfficialScriptAdapter):
    model_id = "clip4cir_rn50x4_fullft"
    script = "src/validate.py"
    launcher = "script"

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        # A bundle_placeholder path IS the bundle directory (registry.checkpoint_missing_paths
        # resolves members under it); the previous `checkpoint.parent` disagreed with that and
        # pointed the evaluator at sibling files that the integrity check never validates.
        combiner_path = checkpoint / "combiner_state.pt"
        clip_path = checkpoint / "clip_state.pt"
        return [_runtime.python_executable(self.model), str(source / self.script), "--dataset", "fashionIQ",
                "--combining-function", "combiner", "--combiner-path", str(combiner_path),
                "--clip-model-path", str(clip_path),
                "--projection-dim", "2560", "--hidden-dim", "5120",
                "--clip-model-name", "RN50x4", "--target-ratio", "1.25", "--transform", "targetpad"]


class TGCIRAdapter(OfficialScriptAdapter):
    model_id = "tgcir"
    script = "test.py"
    launcher = "replay"
    launcher_roots = ()

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        # `test.py:6` is `test(params, model, testset, category)` with no CLI and no checkpoint
        # loader, so the wrapper supplies both. `workbench/replay/tgcir.py` is the existing preflight
        # entry point; it refuses to run while the archive member mapping is unresolved rather than
        # launching an unrelated path.
        return _replay_command(self, checkpoint, request, source)


class SPRCAdapter(OfficialScriptAdapter):
    model_id = "sprc"
    script = "src/blip_validate.py"
    launcher = "script"

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        """The real evaluator, with the two flags its own defaults get wrong.

        `src/blip_validate.py:144` defaults `--blip-model-name` to `blip2_cir_rerank_learn`, which is
        **not registered anywhere in the pinned tree** — the class the FashionIQ path loads is
        `blip2_cir_align_prompt` (`blip2_qformer_cir_align_prompt.py:25`). Launching on the default
        therefore fails at `load_model_and_preprocess`, so the name is passed explicitly.

        `--backbone` selects the vision encoder (`:145`): `pretrain` is EVA-CLIP ViT-g, `pretrain_vitL`
        is ViT-L. The paper reports ViT-L while the code defaults to ViT-g (upstream issue #4), so the
        choice is a SCIENTIFIC VARIANT and is derived from the selected checkpoint rather than assumed:
        a `_vitl` checkpoint must not be run under the ViT-g backbone.
        """
        # The encoder comes from the DECLARED variant, not from the filename. `command()` receives
        # the resolved path, so the declaration is looked up by the request's checkpoint id: the
        # registry now states `backbone` per variant. Inferring from a substring was unsafe because
        # `blip_validate.py:49` loads with `strict=False`, so a rename could silently switch the
        # architecture from ViT-g to ViT-L and evaluate a partly-initialised model without raising.
        declared = next((variant.get("backbone")
                         for variant in self.model.get("checkpoint_variants") or []
                         if variant.get("checkpoint_id") == request.checkpoint_id), None)
        if declared is None:
            # No declared backbone for this selection: refuse to guess. A wrong encoder is a silently
            # different experiment, which is worse than a blocked command.
            raise WorkbenchError(
                "checkpoint_backbone_undeclared",
                "SPRC selects its vision encoder from --backbone and loads with strict=False; the "
                "selected checkpoint declares no backbone, so the encoder cannot be chosen safely.",
                {"model_id": self.model_id, "checkpoint_id": request.checkpoint_id})
        backbone = declared
        return [
            _runtime.python_executable(self.model),
            str(source / self.script),
            "--dataset", "fashionIQ",
            "--blip-model-name", "blip2_cir_align_prompt",
            "--backbone", backbone,
            "--model-path", str(checkpoint),
        ]



class LIMNAdapter(OfficialScriptAdapter):
    model_id = "limn"
    script = "LIMN/test.py"
    launcher = "replay"
    launcher_roots = ("LIMN",)

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        command = [
            _runtime.python_executable(self.model),
            str(Path(__file__).resolve().parents[3] / "workbench" / "replay" / "limn.py"),
            # The directory the bundle's members occupy. `checkpoint` is the path
            # `checkpoint_path` resolved -- for a declared bundle that IS the member directory
            # (the bundle's members are siblings inside it), while for a single-category selection
            # it is that member's own file and the members live beside it. `.parent` was correct
            # only for the second shape and named the whole checkpoint root for a bundle, so
            # `limn.py`'s `checkpoint_root / "<member>.pt"` looked one level too high.
            # The shared member-root rule: a multi-member artifact keeps its members as siblings
            # inside its resolved directory, a single artifact is itself the member. Delegated so
            # this expression cannot drift from the observer's.
            # The shared member-root rule, applied to an ALREADY-RESOLVED path: `checkpoint` comes
            # from `registry.checkpoint_path`, which returns the member directory for a bundle and
            # the member's own file for a single selection. So the shape is already decided, and the
            # rule is exactly "directory means the members are inside it". (Resolving from the
            # declaration instead would be wrong here: this receives a user-supplied path, which
            # cannot say whether it is a bundle before the artifact exists.)
            "--checkpoint-root", str(checkpoint_member_root(checkpoint)),
            "--dataset-root", str(request.dataset_root),
            "--source-root", str(source),
        ]
        if request.checkpoint_id != "base_iter0_all_categories":
            command.extend(("--category", request.checkpoint_id.removeprefix("base_iter0_")))
        return command
class TMEAdapter(OfficialScriptAdapter):
    model_id = "tme"
    script = "precompute_test.py"
    launcher = "replay"
    launcher_roots = ()

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        # The pinned `precompute_test.py` reads a local `--model-path` and its `main()` cannot serve
        # one category at a time or take a checkpoint root, so the audited entry point is the wrapper,
        # which calls the pinned evaluation unchanged and checks the load report the native
        # `strict=False` would have discarded.
        return _replay_command(self, checkpoint, request, source, checkpoint_id=True)


class PthaMtstAdapter(OfficialScriptAdapter):
    model_id = "ptha_mtst"
    script = "src/validate_blip.py"
    launcher = "replay"
    launcher_roots = ()

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        # `validate_blip.py` recommends via `clip_finetune_fiq` but its CLI dispatches only CIRR, and
        # the author publishes no FashionIQ fine-tune, so the wrapper is the audited entry point. The
        # transfer opt-in is passed ONLY for the variant registered as a cross-dataset transfer, which
        # keeps the flag's meaning tied to the checkpoint's recorded training dataset instead of to a
        # model id.
        command = _replay_command(self, checkpoint, request, source, checkpoint_id=True)
        if request.checkpoint_id == "cirr_transfer":
            command.append("--transfer-cirr-to-fashioniq")
        return command


ADAPTERS = {adapter.model_id: adapter for adapter in (CSMCIRAdapter, EncoderAdapter, HintAdapter, PairAdapter, AirKnowAdapter, ConeSepAdapter, HabitAdapter, IntentAdapter, CLVCNetAdapter, DCNetAdapter, CombinerNoftAdapter, CLIP4CirFullftAdapter, TGCIRAdapter, SPRCAdapter, LIMNAdapter, TMEAdapter, PthaMtstAdapter)}


def register_test_adapter(adapter_type: type[OfficialScriptAdapter]) -> None:
    """Register an adapter for synthetic test fixtures only.

    Synthetic adapters must never be reachable through normal registry-driven
    production operation. Tests that need one call this explicitly; production
    code paths never call it. See ``workbench/tests/test_only_synthetic.py``.
    """
    ADAPTERS[adapter_type.model_id] = adapter_type


def register_synthetic_adapters_from_env() -> None:
    """Register synthetic adapters ONLY when explicitly opted in by the harness.

    Subprocess-based tests cannot share an in-process registry, so they set
    ``WORKBENCH_ALLOW_SYNTHETIC_ADAPTERS=1`` for their own child process. Normal
    production invocations never set this variable, so synthetic fixtures remain
    unreachable. Synthetic output is additionally kept out of schema-v2 indexing
    by ``data_kind`` and per-query separation.
    """
    import os

    if os.environ.get("WORKBENCH_ALLOW_SYNTHETIC_ADAPTERS") != "1":
        return

    def _make(model_id: str, script: str) -> type[OfficialScriptAdapter]:
        class _SyntheticAdapter(OfficialScriptAdapter):
            def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
                return [_runtime.python_executable(self.model), str(source / self.script)]

        _SyntheticAdapter.model_id = model_id
        _SyntheticAdapter.script = script
        return _SyntheticAdapter

    # One adapter per model the registry declares as synthetic. A multi-model harness therefore
    # needs no code change here: declaring `synthetic_script` on a registry entry is enough, which
    # is what lets the autopilot end-to-end test drive several models through one command.
    try:
        from workbench.backend.registry import load_registry

        declared = [
            (model["model_id"], model.get("synthetic_script") or "src/evaluator.py")
            for model in load_registry()["models"]
            if model.get("synthetic_source")
        ]
    except Exception:
        declared = []
    if not declared:
        # A registry that predates the marker still gets the single synthetic model the older
        # harnesses rely on.
        declared = [("synthetic_model", "src/evaluator.py")]
    for model_id, script in declared:
        ADAPTERS[model_id] = _make(model_id, script)


register_synthetic_adapters_from_env()

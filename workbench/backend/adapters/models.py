from __future__ import annotations

from pathlib import Path

from workbench.backend.adapters.base import EvalRequest, ModelAdapter
from workbench.backend.registry import checkpoint_path
from workbench.backend import runtime as _runtime


class OfficialScriptAdapter(ModelAdapter):
    script: str

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


class CSMCIRAdapter(OfficialScriptAdapter):
    model_id = "csmcir"
    script = "src/validate_blip_csmcir.py"

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        return [_runtime.python_executable(self.model), str(source / self.script), "--dataset", "fashionIQ", "--blip-model-path", str(checkpoint)]


class EncoderAdapter(OfficialScriptAdapter):
    model_id = "encoder"
    script = "evaluate_model.py"

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        return [_runtime.python_executable(self.model), str(source / self.script), "--dataset", "fashioniq", "--fashioniq_split", "val-split", "--fashioniq_path", f"{request.dataset_root}/", "--ckpt_path", str(checkpoint)]


class HintAdapter(OfficialScriptAdapter):
    model_id = "hint"
    script = "test.py"


class PairAdapter(OfficialScriptAdapter):
    model_id = "pair"
    script = "test.py"


class AirKnowAdapter(OfficialScriptAdapter):
    model_id = "airknow"
    script = "test_BLIP2.py"


class ConeSepAdapter(OfficialScriptAdapter):
    model_id = "conesep"
    script = "test.py"


class HabitAdapter(OfficialScriptAdapter):
    model_id = "habit"
    script = "test.py"


class IntentAdapter(OfficialScriptAdapter):
    model_id = "intent"
    script = "test.py"


class CLVCNetAdapter(OfficialScriptAdapter):
    model_id = "clvc_net"
    script = "test.py"


class DCNetAdapter(OfficialScriptAdapter):
    model_id = "dcnet"
    script = "test.py"

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        return [_runtime.python_executable(self.model), str(source / self.script), "--resume", str(checkpoint)]

class CombinerNoftAdapter(OfficialScriptAdapter):
    model_id = "combiner_rn50x4_noft"
    script = "src/validate.py"

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        return [_runtime.python_executable(self.model), str(source / self.script), "--dataset", "fashionIQ",
                "--combining-function", "combiner", "--combiner-path", str(checkpoint),
                "--projection-dim", "2560", "--hidden-dim", "5120",
                "--clip-model-name", "RN50x4", "--target-ratio", "1.25", "--transform", "targetpad"]


class CLIP4CirFullftAdapter(OfficialScriptAdapter):
    model_id = "clip4cir_rn50x4_fullft"
    script = "src/validate.py"

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


class SPRCAdapter(OfficialScriptAdapter):
    model_id = "sprc"
    script = "src/blip_validate.py"



class LIMNAdapter(OfficialScriptAdapter):
    model_id = "limn"
    script = "LIMN/test.py"

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
            "--checkpoint-root", str(checkpoint if checkpoint.is_dir() else checkpoint.parent),
            "--dataset-root", str(request.dataset_root),
            "--source-root", str(source),
        ]
        if request.checkpoint_id != "base_iter0_all_categories":
            command.extend(("--category", request.checkpoint_id.removeprefix("base_iter0_")))
        return command
ADAPTERS = {adapter.model_id: adapter for adapter in (CSMCIRAdapter, EncoderAdapter, HintAdapter, PairAdapter, AirKnowAdapter, ConeSepAdapter, HabitAdapter, IntentAdapter, CLVCNetAdapter, DCNetAdapter, CombinerNoftAdapter, CLIP4CirFullftAdapter, TGCIRAdapter, SPRCAdapter, LIMNAdapter)}


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

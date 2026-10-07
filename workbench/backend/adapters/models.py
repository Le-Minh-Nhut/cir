from __future__ import annotations

from workbench.backend.adapters.base import EvalRequest, ModelAdapter
from workbench.backend.registry import checkpoint_path


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
        return ["python", str(source / self.script), "--dataset", "fashionIQ", "--blip-model-path", str(checkpoint)]


class EncoderAdapter(OfficialScriptAdapter):
    model_id = "encoder"
    script = "evaluate_model.py"

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        return ["python", str(source / self.script), "--dataset", "fashioniq", "--fashioniq_split", "val-split", "--fashioniq_path", f"{request.dataset_root}/", "--ckpt_path", str(checkpoint)]


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


class CombinerNoftAdapter(OfficialScriptAdapter):
    model_id = "combiner_rn50x4_noft"
    script = "src/validate.py"


class CLIP4CirFullftAdapter(OfficialScriptAdapter):
    model_id = "clip4cir_rn50x4_fullft"
    script = "src/validate.py"


class TGCIRAdapter(OfficialScriptAdapter):
    model_id = "tgcir"
    script = "test.py"


class SPRCAdapter(OfficialScriptAdapter):
    model_id = "sprc"
    script = "src/blip_validate.py"


class LIMNAdapter(OfficialScriptAdapter):
    model_id = "limn"
    script = "LIMN/test.py"


ADAPTERS = {adapter.model_id: adapter for adapter in (CSMCIRAdapter, EncoderAdapter, HintAdapter, PairAdapter, AirKnowAdapter, ConeSepAdapter, HabitAdapter, IntentAdapter, CLVCNetAdapter, DCNetAdapter, CombinerNoftAdapter, CLIP4CirFullftAdapter, TGCIRAdapter, SPRCAdapter, LIMNAdapter)}

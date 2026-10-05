from __future__ import annotations

from workbench.backend.adapters.base import EvalRequest, ModelAdapter
from workbench.backend.registry import ROOT, checkpoint_path


class OfficialScriptAdapter(ModelAdapter):
    script: str

    def build_command(self, request: EvalRequest) -> list[str]:
        checkpoint = self.validate_request(request)
        return self.command(ROOT / "third_party" / self.model["source_dir"], checkpoint, request)

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        raise NotImplementedError("upstream evaluation command requires audit before execution")


class CSMCIRAdapter(OfficialScriptAdapter):
    model_id = "csmcir"
    script = "src/validate_blip_csmcir.py"

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        return ["python", str(source / self.script), "--dataset", "fashionIQ", "--blip-model-path", str(checkpoint_path(self.model_id, checkpoint))]


class EncoderAdapter(OfficialScriptAdapter):
    model_id = "encoder"
    script = "evaluate_model.py"

    def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
        return ["python", str(source / self.script), "--dataset", "fashioniq", "--fashioniq_split", "val-split", "--fashioniq_path", str(request.dataset_root), "--ckpt_path", str(checkpoint_path(self.model_id, checkpoint))]


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


ADAPTERS = {adapter.model_id: adapter for adapter in (CSMCIRAdapter, EncoderAdapter, HintAdapter, PairAdapter, AirKnowAdapter, ConeSepAdapter, HabitAdapter, IntentAdapter)}

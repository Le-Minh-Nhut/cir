"""Registry-facing adapter for the single-file CLIP4Cir noft variant.

All machinery, the reproduced scoring expression, and both stated blockers live in
``workbench.composition.model_clip4cir``; this module exists because
``adapters.adapter_class`` requires ``module.ADAPTER.model_id`` to equal the registry row's
``model_id``, and one module cannot declare two ids.

The noft checkpoint is one file — ``{'epoch': 165, 'Combiner': OrderedDict(12)}`` with geometry
640->2560, 5120->5120, 5120->640 (see the registry's measurement) — so it supplies the Combiner
only; the CLIP encoder is the stock OpenAI RN50x4 asset, which this adapter does not stage or
verify.
"""
from __future__ import annotations

from workbench.composition.model_clip4cir import Clip4CirBase

class Adapter(Clip4CirBase):
    model_id = "combiner_rn50x4_noft"
    bundle = False

    def _verify_interpreter_once(self) -> None:
        """The shared environment gate, called from THIS module on purpose.

        `test_circus_env_gate_omp.py` reads each registered module's own source to prove the gate
        cannot be skipped. Delegating to the shared base would satisfy the runtime but hide the
        call from that check, so the call lives here where it is greppable.
        """
        if not self._interpreter_verified:
            self.require_verified_interpreter()
            self._interpreter_verified = True

ADAPTER = Adapter

"""Registry-facing adapter for the two-member CLIP4Cir full-FT bundle.

All machinery, the reproduced scoring expression, and both stated blockers live in
``workbench.composition.model_clip4cir``; this module exists because
``adapters.adapter_class`` requires ``module.ADAPTER.model_id`` to equal the registry row's
``model_id``, and one module cannot declare two ids.

The bundle's two members are resolved by DECLARED ROLE (``combiner`` / ``clip``) through
``backend.adapters.models._declared_member_paths``, never by filename: the registry's
``bundle_members`` is the one source of truth, and spelling a name here is the exact defect that
made a complete pair read as two missing paths.
"""
from __future__ import annotations

from workbench.composition.model_clip4cir import Clip4CirBase

class Adapter(Clip4CirBase):
    model_id = "clip4cir_rn50x4_fullft"
    bundle = True

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

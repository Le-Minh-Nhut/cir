# IAG-SRME for Composed Image Retrieval

This branch is a clean implementation of Intent-Anchored Grounded State-Recomputed Marginal
Execution:

```text
text -> four stable intents -> anchor grounding -> current-state reading
     -> grounded contexts -> four local same-parent counterfactuals
     -> target-free consequence scores -> execute one / STOP -> final retrieval query
```

There is no CIR model dependency outside the configured FG-CLIP backbone and no model-specific
precomputed representation requirement.

- [Implementation and run guide](docs/IAG_SRME_IMPLEMENTATION.md)
- [Clean-rewrite audit](docs/IAG_SRME_REWRITE_AUDIT.md)
- [Authoritative master specification](doc/CIR_TAPER_IAG_SRME_UNIFIED_MASTER_ARCHITECTURE_AND_LOSS_SPEC_V1_2026-08-29.md)

```bash
pytest -q
python src/smoke_iag_srme.py --diagnostics
python src/smoke_fgclip_integration.py --max-steps 3
```

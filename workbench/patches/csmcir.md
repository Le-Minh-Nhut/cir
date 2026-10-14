# CSMCIR — pinned source cannot register its own FashionIQ model

## Status

**BLOCKED_SOURCE_UNIMPORTABLE.** This is an upstream defect in `qzp2018/CSMCIR` at the pinned
commit `774f94e2076ff17ea91703a6239d2a08f0e1a44e`. It is not a workbench bug, and the workbench
does not apply this patch automatically.

## Evidence

The registered evaluation entrypoint is `src/validate_blip_csmcir.py`, which the model's README
documents at line 115 and which the adapter generates as:

```
python src/validate_blip_csmcir.py --dataset fashionIQ --blip-model-path <checkpoint>
```

That script does, at `src/validate_blip_csmcir.py:12`:

```python
from lavis.models import load_model_and_preprocess
```

`src/lavis/models/__init__.py` then executes 98 **unguarded** `from … import …` statements at
lines 44–151. Every one of those referenced modules is absent from the repository:

```
$ python3 - <<'PY'
import re; from pathlib import Path
init = Path('src/lavis/models/__init__.py'); base = init.parent
bad = [i for i, l in enumerate(init.read_text().splitlines(), 1)
       if (m := re.match(r'from (lavis\.models\.[\w.]+) import (\w+)', l.strip()))
       and not (base / (m.group(1).replace('lavis.models.','').replace('.','/') + '.py')).is_file()]
print(len(bad), min(bad), max(bad))   # -> 98 44 151
PY
98 44 151
```

Examples of the missing modules (all returning HTTP 404 from the pinned public revision):

- `src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt_caption_query.py`
- `src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt_daul_contrastive.py`
- `src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt_target_image_caption_shiyan{4..92}.py`

The class CSMCIR actually needs **is** present and **is** registered:

- `src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt_csmcir.py:27` —
  `@registry.register_model("blip2_cir_align_prompt_csmcir")`
- imported at `src/lavis/models/__init__.py:152`

But line 152 sits *after* the 98 failing imports, so the package never finishes loading and the
registration never happens. No `try`/`except` wraps any of lines 44–151, so the first missing module
raises `ModuleNotFoundError` and the entrypoint dies during import.

## Consequence

1. The official evaluator cannot start, so Recall@10/Recall@50 cannot be produced from the pinned
   source as published.
2. Because the registration never runs, `load_model_and_preprocess(name="blip2_cir_align_prompt_csmcir")`
   — the adapter's default model name — would raise `AttributeError: 'NoneType' object has no
   attribute 'from_pretrained'` even if the imports were bypassed.

The workbench reports this rather than hiding it: `csmcir` is listed with the pinned commit, and the
`COT_ours2` / `qwen_captions` blockers that appear alongside it are real but strictly downstream of
this one. **This defect, not the missing captions, is what blocks CSMCIR.**

## Minimal patch (for a human to apply and review)

Delete lines 44–151 of `src/lavis/models/__init__.py`, keeping line 152. That removes only
registrations for modules the repository does not contain, and leaves every shipped model
registered. Nothing else in the pinned tree needs to change for the import to succeed.

```diff
--- a/src/lavis/models/__init__.py
+++ b/src/lavis/models/__init__.py
@@
-from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_caption_query import Blip2QformerCirAlignPromptCaptionQuery
-... (96 further lines, 45-151, each naming an absent module) ...
 from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_csmcir import Blip2QformerCirAlignPromptCSMCIR
```

## Why the workbench does not apply it

Project rule: an upstream implementation may not be silently altered, and instrumentation is only
permitted after official reproduction is verified. Editing the model package would change which
classes LAVIS registers for the run, so it counts as modifying the official implementation. The
patch is left here for a human to apply deliberately, and `csmcir` is not presented as runnable.

## Verification after applying

```bash
cd workbench/third_party/CSMCIR/src
python3 -c "from lavis.models import load_model_and_preprocess; \
from lavis.common.registry import registry; \
print('blip2_cir_align_prompt_csmcir' in registry.list_models())"
# expected: import succeeds and prints True
```

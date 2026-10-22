# R16 — ENCODER checkpoint-key compatibility

## Verdict

**Allow narrowly**, not because of any score. Pinned `BindingDecoder.__init__` creates no `fc` attribute (source lines 89–116), `BindingDecoder` declares no `forward` method (lines 89–128), and the full pinned-checkout search found no reference to `binding_decoder.fc`. The two checkpoint keys do not correspond to module parameters; the actual load reports no missing keys, and the remaining 415 checkpoint tensors match model parameters in shape. Direct binding output comparison is exactly equal with vs. without perturbed `fc.0` tensors. This supports ignoring only these two known historical extras under identity gates.

## 1. Complete source path

Read `workbench/third_party/ENCODER/model_try2.py` lines 89–128 (`BindingDecoder`), 131–432 (`Backbone` methods/forward feature paths), and 464–485 (`Encoder` construction and feature path). Constructor arguments used by this model are established at `Backbone.__init__` line 207 and replay construction at `workbench/replay/encoder.py:171–180`.

- `BindingDecoder.__init__(dim, hidden_dim, N_p, N_v)` declares `learnable_relation`, `dim`, `layer_norm`, `sigmoid`, `softmax`, `mlp_p`, `mlp_v`, and `mlp`; it does **not** declare an attribute named `fc` (`model_try2.py:89–116`).
- Its `binding(V)` uses those declared members and returns `(entity, relation)` (`model_try2.py:118–128`). There is no `forward` method in this class; PyTorch's inherited forward is unimplemented. `Backbone` constructs it at line 207. Backbone calls `.binding(...)` at lines 375, 410, and 426; no forward call uses `fc`.
- `Backbone`'s own `self.fc = nn.Linear(768, 512)` at line 140 is separate from `self.binding_decoder.fc`. It is used for image features at lines 357 and 383. `Encoder` creates `Backbone` and `loss_T` at `model_try2.py:464–470`; its retrieval/compose paths delegate through `Backbone` at lines 476–497.

**Whole pinned-checkout search:** `functions.grep` over `workbench/third_party/ENCODER`, case-insensitive, pattern `binding_decoder\\.fc|binding_decoder\\[.fc.|binding_decoder.*fc`, returned `No matches found`. This search included the full checkout, not just `model_try2.py`. Therefore no pinned-tree forward path references `binding_decoder.fc`.

## 2. Actual load report

Exact command executed (working directory `workbench/third_party/ENCODER`; `TMPDIR=/tmp`):

```sh
TMPDIR=/tmp /home/heheboiz/data/cir/workbench/envs/encoder/bin/python -u -c 'import os,sys,torch; root=os.getcwd(); sys.path.insert(0,root); import model_try2; print("model imported",flush=True); model=model_try2.Encoder(hidden_dim=512,dropout=0.5,local_token_num=3,t=0.1,wc=2,N_p=2,weighted=True); print("constructed",flush=True); ckpt=torch.load("/home/heheboiz/data/cir/workbench/artifacts/checkpoints/encoder/fashioniq.pt",map_location="cpu"); report=model.load_state_dict(ckpt,strict=False); print("missing_keys:",repr(report.missing_keys)); print("unexpected_keys:",repr(report.unexpected_keys)); params=dict(model.named_parameters()); mismatches=[(k,tuple(v.shape),tuple(params[k].shape) if k in params else None) for k,v in ckpt.items() if k in params and tuple(v.shape)!=tuple(params[k].shape)]; consumed=[k for k in ckpt if k in params and tuple(ckpt[k].shape)==tuple(params[k].shape)]; unconsumed=[k for k in ckpt if k not in params]; print("shape_mismatches:",repr(mismatches)); print("consumed_count:",len(consumed)); print("unconsumed_count:",len(unconsumed)); print("unconsumed_keys:",repr(unconsumed)); print("checkpoint_tensor_count:",len(ckpt)); print("named_parameter_count:",len(params))'
```

Observed stdout (the leading `2` is the constructor's `print(N_p)` at `model_try2.py:91`):

```text
model imported
2
constructed
missing_keys: []
unexpected_keys: ['backbone.binding_decoder.fc.0.weight', 'backbone.binding_decoder.fc.0.bias']
shape_mismatches: []
consumed_count: 415
unconsumed_count: 2
unconsumed_keys: ['backbone.binding_decoder.fc.0.weight', 'backbone.binding_decoder.fc.0.bias']
checkpoint_tensor_count: 417
named_parameter_count: 415
```

`missing_keys: []` establishes every parameter declared by this model received a checkpoint value. The replay wrapper currently rejects these keys because it filters only `loss_T` from unexpected keys (`workbench/replay/encoder.py:195–203`).

## 3. Synthetic class-level proof

Exact command executed (working directory `workbench/third_party/ENCODER`; `TMPDIR=/tmp`):

```sh
TMPDIR=/tmp /home/heheboiz/data/cir/workbench/envs/encoder/bin/python -u -c 'import sys, torch
sys.path.insert(0, ".")
import model_try2
module = model_try2.BindingDecoder(512, 512, 2, 3).cuda().eval()
print("has_fc_attribute:", hasattr(module, "fc"))
print("forward_method:", module.forward.__qualname__)
V = torch.randn(1, 4, 512, device="cuda")
base = module.state_dict()
extra = {"fc.0.weight": torch.randn(512,512,device="cuda")*1000, "fc.0.bias": torch.randn(512,device="cuda")*1000}
with torch.no_grad(): out_absent = module.binding(V)
report = module.load_state_dict({**base, **extra}, strict=False)
with torch.no_grad(): out_present = module.binding(V)
print("synthetic_tensor_shapes:", {k:tuple(v.shape) for k,v in extra.items()})
print("load_report:", repr(report))
for label,a,b in zip(("entity","relation"), out_absent, out_present): print(label,"max_abs_diff:",float((a-b).abs().max()),"identical:",torch.equal(a,b))
print("state_keys_match_synthetic:", {k:k in module.state_dict() for k in extra})
try: module(V)
except Exception as e: print("forward_call:",type(e).__name__,str(e))'
```

The above is the exact multiline Python `-c` command executed. Observed stdout:

```text
2
has_fc_attribute: False
forward_method: _forward_unimplemented
synthetic_tensor_shapes: {'fc.0.weight': (512, 512), 'fc.0.bias': (512,)}
load_report: _IncompatibleKeys(missing_keys=[], unexpected_keys=['fc.0.weight', 'fc.0.bias'])
entity max_abs_diff: 0.0 identical: True
relation max_abs_diff: 0.0 identical: True
state_keys_match_synthetic: {'fc.0.weight': False, 'fc.0.bias': False}
forward_call: NotImplementedError Module [BindingDecoder] is missing the required "forward" function
The existing helper is named `sha256_file`, not `checkpoint_sha256`; `load_trusted_whole_model` accepts `expected_sha256` and verifies it before unpickling (`common.py:43–55`). I found no `checkpoint_sha256` function in `common.py`. ENCODER's current state-dict load in `replay/encoder.py:182–195` uses direct `torch.load` instead, after wrapper checkpoint preflight validates the expected hash (`encoder.py:47–71, 124–135`).

Exact command executed for the source-pin and checkpoint-key inspection (working directory `workbench/third_party/ENCODER`; `TMPDIR=/tmp`):

```sh
TMPDIR=/tmp /home/heheboiz/data/cir/workbench/envs/encoder/bin/python -u -c 'import sys, torch
from pathlib import Path
sys.path.insert(0, "/home/heheboiz/data/cir")
from workbench.replay.common import source_pin_status, sha256_file
source=Path("/home/heheboiz/data/cir/workbench/third_party/ENCODER")
print("source_pin_status:",repr(source_pin_status(source,"29a2a31d6a56f677bf450c3be7cdaef423fb7018")))
checkpoint=Path("/home/heheboiz/data/cir/workbench/artifacts/checkpoints/encoder/fashioniq.pt")
print("checkpoint_sha256:",sha256_file(checkpoint))
ckpt=torch.load(checkpoint,map_location="cpu")
for key in ("backbone.binding_decoder.fc.0.weight","backbone.binding_decoder.fc.0.bias"): print(key,tuple(ckpt[key].shape),ckpt[key].dtype)'
```

Observed stdout:

```text
source_pin_status: (False, 'undeclared files present in checkout: 1 (first: open_clip_pytorch_model.bin)')
checkpoint_sha256: 0f31c50150a7bda3495e3378df13b4cec7924b52efd23b5e5dd9731459da9173
backbone.binding_decoder.fc.0.weight (512, 512) torch.float32
backbone.binding_decoder.fc.0.bias (512,) torch.float32
```

Thus the source pin implementation is the correct binding check, but this direct call does **not** pass on the currently staged checkout because the auxiliary binary is undeclared; wrapper preflight uses that same source check. This report does not claim a successful runtime pin validation on this working tree.
## 4. Shape and coverage audit

The real-model load/audit above compared every checkpoint entry with its named model parameter and observed `shape_mismatches: []`. Exactly **415** of **417** checkpoint tensors are consumed by same-named parameters with matching shapes. Exactly **2** are unconsumed and have no parameter target:

- `backbone.binding_decoder.fc.0.weight`, checkpoint shape `(512, 512)`, `torch.float32`.
- `backbone.binding_decoder.fc.0.bias`, checkpoint shape `(512,)`, `torch.float32`.

No checkpoint tensor with a model parameter name had a shape mismatch. The direct checkpoint inspection printed those two shapes/dtypes; all other key/shape matches are summarized by the executed audit output above.

## 5. Identity binding using existing checks

The exception should be gated by the existing checkpoint identity and source-pin checks, not by a new framework:

- Checkpoint identity: `workbench/replay/common.py:43–55` has `load_trusted_whole_model(path, expected_sha256, torch_module)`, which refuses absent/mismatched trusted hashes before deserialization. `sha256_file` is implemented at `common.py:35–40`; the existing `checkpoint_sha256` field is the digest convention already used by reports. The real staged checkpoint digest observed here is `0f31c50150a7bda3495e3378df13b4cec7924b52efd23b5e5dd9731459da9173`, also declared as `expected_sha256` for ENCODER's `fashioniq` variant at `workbench/registry/models.yaml:543–557`.
- Source identity: `source_pin_status` delegates to `source_checkout_status` (`common.py:58–86`), which calls `source_clean_and_pinned`; ENCODER wrapper's `validate_source` passes the registered `MODEL["upstream_commit_sha"]` and model record (`workbench/replay/encoder.py:74–76`). Registry pin is `29a2a31d6a56f677bf450c3be7cdaef423fb7018` (`models.yaml:524–528`).
- The only permitted condition should therefore be the existing registered ENCODER checkpoint variant/digest (`checkpoint_id=fashioniq`) **and** the existing source pin/cleanliness check for that model. The rule should allow only the two complete key names listed above, and only if neither key exists in `encoder_model.state_dict()` (i.e. no corresponding model-owned target). Keep the current rejection of every other unexpected key and every missing key.

Observed call to the existing source-pin checker on this staged tree:

```text
source_pin_status: (False, 'undeclared files present in checkout: 1 (first: open_clip_pytorch_model.bin)')
checkpoint_sha256: 0f31c50150a7bda3495e3378df13b4cec7924b52efd23b5e5dd9731459da9173
backbone.binding_decoder.fc.0.weight (512, 512) torch.float32
backbone.binding_decoder.fc.0.bias (512,) torch.float32
```

Thus the source pin implementation is the correct binding check, but this direct call does **not** pass on the currently staged checkout because the auxiliary binary is undeclared; wrapper preflight uses that same source check. This report does not claim a successful runtime pin validation on this working tree.

## 6. Smallest proposed change (not applied)

Replace only the current unexpected-key filter in `workbench/replay/encoder.py:197` with an identity- and module-target-bound allowlist; retain the existing missing-key and unexpected-key failure gate:

```diff
-        unexpected = [key for key in getattr(report, "unexpected_keys", []) if key != "loss_T"]
+        decoder_fc = {"backbone.binding_decoder.fc.0.weight", "backbone.binding_decoder.fc.0.bias"}
+        permitted = {"loss_T"} | (decoder_fc if (
+            variant["checkpoint_id"] == "fashioniq"
+            and variant.get("expected_sha256") == "0f31c50150a7bda3495e3378df13b4cec7924b52efd23b5e5dd9731459da9173"
+            and source_pin_status(source_root, MODEL["upstream_commit_sha"], MODEL)[0]
+            and decoder_fc.isdisjoint(encoder_model.state_dict())
+        ) else set())
+        unexpected = [key for key in getattr(report, "unexpected_keys", []) if key not in permitted]
```

Use of `variant["expected_sha256"]` binds to registry identity; actual bytes are validated by `validate_checkpoint`/existing checkpoint hash machinery, and `source_pin_status` binds to the clean pinned source. The exception is exact-key-only, not prefix-wide. It must not mask any other unexpected key or missing parameter. Since the source check currently reports the staged auxiliary asset as undeclared, resolver/implementation must ensure the existing preflight's source state is valid before this exception can activate; no change to source checking is proposed here.

## Limits

No full ENCODER evaluation was run; no score was used. I did **not** establish from source history why the checkpoint contains these two unused tensors, whether they came from a prior model revision, or any author statement about them. The binding synthetic proof exercised `binding`, the actual class method called by `Backbone`, rather than a nonexistent `BindingDecoder.forward`. The staged checkout currently fails `source_pin_status` because of the undeclared auxiliary asset, so the identity gate is specified from existing code but was not observed passing in this working-tree state.

# 95 — external asset unblock: `clvc_net` ResNet-50 and `tgcir` CLIP ViT-B/16

Date: 2026-11-12
Repo: `/home/heheboiz/data/cir` (branch `research/cir-failure-workbench`)
Controller: `workbench/envs/ctl312h/bin/python`
Scope of this file's authority: the two refused models, the acquisition of one absent file, and one
added `auxiliary_assets.yaml` record. No adapter, no `models.yaml`, no test, no `workbench.env` was
touched; the live units on 8000/5173 were not restarted and the GPU/lock were not used.

## Verdict in one line

`clvc_net` is unblocked on the asset axis (its ResNet-50 file was genuinely absent and is now
present and prefix-verified); `tgcir`'s CLIP file was already present and is now **declared and
VERIFIED** in the auxiliary registry, but `tgcir` **still reports the same provenance blocker**,
because the blocker is read from the *preparation contract*, not from the auxiliary registry — see
`## Còn chặn`.

## Blocker 1 — `clvc_net`: the pinned ResNet-50 file was genuinely absent

Two different files, not a naming confusion. The declared asset is
`hub/checkpoints/resnet50-19c8e357.pth` under `TORCH_HOME`; the cache held
`resnet50-0676ba61.pth`, which is torchvision's newer URL and a different object (different size
too: 102530333 vs 102502400 bytes).

```
0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a  /home/heheboiz/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097  /home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
```

First 16 chars: `0676ba61b6795bbe` vs `19c8e3572231adff`. `0676ba61` != the declared prefix
`19c8e357`, so the miss is real. (The `19c8e357` line above is the file *after* acquisition; the
pre-acquisition state was `state=missing`.)

### Acquisition — repo tooling, not `curl`

The downloader has a model filter (`--model {airknow,clip4cir_rn50x4_fullft,clvc_net,...}`), so it
was scoped; nothing for another model was requested. `--dry-run` first showed exactly one planned
transfer.

```
$ workbench/envs/ctl312h/bin/python workbench/scripts/download_auxiliary_assets.py --model clvc_net --dry-run
torchvision ResNet-50 ImageNet weights: destination=/home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth state=missing required=true automatic_download_available=true source_unverified=false source=https://download.pytorch.org/models/resnet50-19c8e357.pth
DRY RUN: download https://download.pytorch.org/models/resnet50-19c8e357.pth -> /home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
DRY RUN: no network request and no filesystem mutation.

$ workbench/envs/ctl312h/bin/python workbench/scripts/download_auxiliary_assets.py --model clvc_net
torchvision ResNet-50 ImageNet weights: destination=/home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth state=missing required=true automatic_download_available=true source_unverified=false source=https://download.pytorch.org/models/resnet50-19c8e357.pth
DOWNLOADED /home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth: sha256=19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097 (official SHA256 prefix verified)
EXIT=0
```

### Verification of the acquired file

```
102502400 /home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097  /home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
```

The filename hash segment (`19c8e357`) equals the declared `expected_sha256_prefix`, and the
measured digest begins with it.

**What the bytes themselves prove, stated plainly:** only a 32-bit prefix. The registry declares
`expected_sha256: null` and `expected_sha256_prefix: 19c8e357`, so admission checks that the file's
SHA-256 *starts with* those 8 hex characters. There is no full published digest in this registry to
compare the remaining 56 hex characters against, and this workbench does not have one for this URL.
The evidence is therefore: URL-pinned acquisition + exact byte count + 8-hex-digit digest prefix.
It is **not** a full-digest verification, and this file does not claim one. (Note also that the
prefix matches the file's own name, so a truncated or substituted file of the same size whose hash
happened to share 8 leading hex digits would pass — the size pin is what bounds that risk here.)

## Blocker 2 — `tgcir`: the CLIP file exists and is now declared

### URL and digest, from the installed module

```
$ workbench/envs/tgcir39-la/bin/python -c "import clip.clip as c; print(c._MODELS['ViT-B/16'])"
https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt

$ stat -c '%s' ~/.cache/clip/ViT-B-16.pt
350837078
$ sha256sum ~/.cache/clip/ViT-B-16.pt
5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f  /home/heheboiz/.cache/clip/ViT-B-16.pt
```

### Cross-check: URL segment vs measured digest

| source | value |
|---|---|
| digest segment of the OpenAI URL | `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` |
| `sha256sum ~/.cache/clip/ViT-B-16.pt` | `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` |

**They match, byte for byte, all 64 hex characters.** So `provenance_status: VERIFIED` is justified
by a *published* digest (OpenAI's own digest-addressed URL, which upstream `clip.load` self-verifies
against), not by a local recording. `workbench/replay/tgcir.py:162-169` reads the same URL segment
from the pinned module, so the registry and the consumer agree on one value.

### The added asset block, verbatim

```yaml
- model_id: tgcir
  family: OpenAI CLIP ViT-B/16 weights
  required_for: official_evaluation
  source_type: direct_url
  # `direct_url` uses `source_repo` as the URL verbatim; this is the digest-addressed URL the
  # pinned `clip.clip._MODELS['ViT-B/16']` entry names (openai/CLIP clip/clip.py:37), read from
  # `workbench/envs/tgcir39-la/bin/python -c "import clip.clip as c; print(c._MODELS['ViT-B/16'])"`.
  source_repo: https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
  source_revision: null
  source_path: null
  destination_scope: user_cache
  destination_path: ViT-B-16.pt
  # OpenAI's CLIP does NOT read XDG_CACHE_HOME: `clip.load` resolves its cache as
  # `download_root or os.path.expanduser("~/.cache/clip")` (openai/CLIP clip/clip.py:120). The
  # registry must match the consumer, so this root is fixed rather than environment-driven.
  cache_root_fixed: ~/.cache/clip
  cache_root_env_var: XDG_CACHE_HOME
  automatic_download_available: true
  # Measured from the file already in the cache (`stat -c %s`) and `sha256sum`; the digest equals
  # the hash segment of the URL above, so this is a URL-published digest verified against the bytes
  # on disk, not a local-only recording. `workbench/replay/tgcir.py` checks the same value.
  expected_size_bytes: 350837078
  expected_sha256: 5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f
  provenance_status: VERIFIED
```

### The registry now resolves and validates that asset

```
$ workbench/envs/ctl312h/bin/python workbench/scripts/download_auxiliary_assets.py --model tgcir --verify-only
OpenAI CLIP ViT-B/16 weights: destination=/home/heheboiz/.cache/clip/ViT-B-16.pt state=present required=true automatic_download_available=true source_unverified=false source=https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
/home/heheboiz/.cache/clip/ViT-B-16.pt: 350837078 bytes sha256=5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f OFFICIAL SHA256 MATCH
EXIT=0
```

and through the shared validator, at the destination the registry resolves:

```
/home/heheboiz/.cache/clip/ViT-B-16.pt AuxiliaryValidationResult(ok=True, status='VERIFIED', issues=[])
```

## Before / after blocker lists

`preflight_blockers(model_by_id(m), resolve_config())`, controller interpreter, same shell, same
repo, before the edit and after it.

BEFORE:

```
clvc_net ['torchvision ResNet-50 ImageNet weights missing: /home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth', 'external runtime asset file missing: torchvision ResNet-50 ImageNet weights (/home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth)', 'command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT']
tgcir ['external runtime asset provenance unresolved: OpenAI CLIP ViT-B/16 weights/cache', 'command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT']
```

AFTER:

```
clvc_net ['command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT']
tgcir ['external runtime asset provenance unresolved: OpenAI CLIP ViT-B/16 weights/cache', 'command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT']
```

Delta: `clvc_net` lost both asset blockers (the file-missing line and the runtime-asset line) and
kept only the upstream-entrypoint statement. `tgcir` is **unchanged** — the asset is declared and
verified, but the blocker does not read that registry.

## POST codes and bodies

`POST /api/jobs`, live backend on 127.0.0.1:8000.

With the protocol named in the task (`circus_fashioniq_release_v1`, which is not a
`clvc_net` protocol — its only supported protocol is `fashioniq_val_split`, so that call also
reports `protocol incompatible`):

```
clvc_net 409
{"detail":{"error":"evaluation_blocked","message":"Official evaluation is not ready.","details":{"blockers":["command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT","protocol incompatible: circus_fashioniq_release_v1"]}}}

tgcir 409
{"detail":{"error":"evaluation_blocked","message":"Official evaluation is not ready.","details":{"blockers":["external runtime asset provenance unresolved: OpenAI CLIP ViT-B/16 weights/cache","command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT"]}}}
```

Repeated with each model's own native protocol (`fashioniq_val_split`), to isolate the asset question:

```
clvc_net 409
{"detail":{"error":"evaluation_blocked","message":"Official evaluation is not ready.","details":{"blockers":["command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT"]}}}

tgcir 409
{"detail":{"error":"evaluation_blocked","message":"Official evaluation is not ready.","details":{"blockers":["external runtime asset provenance unresolved: OpenAI CLIP ViT-B/16 weights/cache","command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT"]}}}
```

So both still return 409, and for `clvc_net` the **only** remaining reason is the expected
`command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT`. No asset reason remains for it.

## Did the API pick up the registry edit without a restart?

**Partly — and not in `/api/models`.** Two different caches, and only one of them was claimed to be
size+mtime keyed:

* `registry._load_yaml_cached` / `load_auxiliary_assets` are keyed on `(path, size, mtime_ns)`, so a
  fresh process or any handler that loads the registry sees the new entry immediately. Proven: the
  `--verify-only` run and the shared validator above resolved `tgcir`'s asset from the edited file
  with no restart, and `GET /api/models` was *not* used to establish that.
* `main._MODELS_CACHE` is a **module-level list** with no key at all (`main.py:170-176`), so
  `GET /api/models` serves the pre-edit document for the life of the process. It did **not** change:

```
clvc_net [('author_drive_unresolved', False, ['torchvision ResNet-50 ImageNet weights missing: /home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth', 'external runtime asset file missing: torchvision ResNet-50 ImageNet weights (/home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth)', 'command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT'])]
tgcir [('tgcir_archive_unresolved', False, ['external runtime asset provenance unresolved: OpenAI CLIP ViT-B/16 weights/cache', 'command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT'])]
```

That listing still reports `clvc_net`'s file as missing although the file is present and verified,
which is the cache, not the file. The `/api/jobs` POST path recomputes `preflight_blockers` per
request and **did** drop `clvc_net`'s asset blockers — that is the visible proof the registry edit
took effect without a restart. Restarting the unit was out of scope and was not done.

## Còn chặn

* **`command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT`** — still present for both models.
  Expected and correct: it is a true statement about the upstream entrypoint (TG-CIR's `train.py`
  declares `--restore_file` and never reads it; CLVCNet ships no CLI), not an asset fact. Not
  "fixed" here, and it must not be papered over.
* **`tgcir`: `external runtime asset provenance unresolved: OpenAI CLIP ViT-B/16 weights/cache`** —
  still present. This is the honest remaining gap and the reason this file does not claim `tgcir`
  unblocked. The cause is a **declaration gap outside this file's ownership**, observed directly:

  ```
  contract asset: {'name': 'OpenAI CLIP ViT-B/16 weights/cache', 'state': 'BLOCKED_EXTERNAL_ASSET_PROVENANCE', 'expected_local_location': None, 'location_env_var': 'WORKBENCH_TGCIR_VITB16_WEIGHTS', 'cache_path_suffix': None}
  statuses: [{'name': 'OpenAI CLIP ViT-B/16 weights/cache', 'status': 'UNRESOLVED_PROVENANCE', 'path': None, 'expected_sha256': None, 'expected_sha256_prefix': None, 'actual_sha256': None}]
  ```

  `external_asset_statuses` walks the **preparation contract**, whose `tgcir_fashioniq` record has
  `expected_local_location: null` and `cache_path_suffix` absent. With no location it returns
  `UNRESOLVED_PROVENANCE` purely from `state: BLOCKED_EXTERNAL_ASSET_PROVENANCE`
  (`registry.py:465-467`) and never looks at a file. There is no code path that joins a contract
  asset to an auxiliary asset of the same `family`; the only lever the contract offers is its
  `location_env_var`, and with `WORKBENCH_TGCIR_VITB16_WEIGHTS` unset the location is `None`.
  Measured, to be unambiguous: setting that variable to the existing file **does** clear it —

  ```
  $ WORKBENCH_TGCIR_VITB16_WEIGHTS=/home/heheboiz/.cache/clip/ViT-B-16.pt ... preflight_blockers(...)
  tgcir ['command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT']
  ```

  so the bytes and the declaration are fine and only the *location declaration* is missing. Closing
  that requires either `workbench/config/workbench.env` (explicitly forbidden here) or an edit to
  `workbench/registry/preparation_contracts.yaml` (`state` → a verified state, plus
  `expected_local_location: ~/.cache/clip/ViT-B-16.pt` and `expected_sha256`). Both are outside the
  files this task owns, so neither was done. Adding the auxiliary record — which is what this task
  asked for, and which is now `VERIFIED` and loadable — is necessary but **not sufficient** for
  `tgcir`; saying otherwise would be false.
* `clvc_net`'s `protocol incompatible: circus_fashioniq_release_v1` in the first POST is a
  protocol-name fact, not an asset fact: `clvc_net` supports only `fashioniq_val_split`. With its
  own protocol the reason disappears from the list.

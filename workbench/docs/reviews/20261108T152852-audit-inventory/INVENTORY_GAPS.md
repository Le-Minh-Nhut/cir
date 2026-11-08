# INVENTORY GAPS — what this audit could NOT establish

Every entry states the field, why it is unestablished, and the evidence that would settle it. Nothing in `REPOSITORY_INVENTORY.md` or `inventory.json` is guessed: where a value is unknown it is marked here.

## A. Blocking gaps (affect whether a model can run)

### A1. Interpreter for 8 of 17 models is unresolved

**Field:** `environment.interpreter` for `airknow`, `pair`, `clvc_net`, `dcnet`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `tgcir`, `sprc`.

**Status:** `UNRESOLVED`. The registry declares a `python_env_var` for each; the live `workbench/config/workbench.env` sets only 9 of the 17 variables. `workbench/backend/runtime.py:resolve_model_interpreter` falls back to `workbench/artifacts/environment_assignment.json`, and that file **does not exist** (verified by `ls`).

**Complication:** environments *do* exist for six of these eight — `envs/airknow-adapted`, `envs/pair-adapted`, `envs/clip4cir-adapted`, `envs/sprc-adapted`, `envs/dcnet-prepare`, `artifacts/environments/clip4cir-38`, `dcnet-377`, `sprc-39` — but none is reachable through the resolver, because being present on disk is not the same as being assigned. `envs/dcnet-prepare` additionally has no torch.

**Would settle it:** setting the 8 variables in `workbench/config/workbench.env` (operator-owned; this audit must not touch it), or provisioning `environment_assignment.json` with `state: RUNTIME_VERIFIED`. Either makes the interpreter a fact instead of a gap.

### A2. Three checkpoint artifacts have verified digests but unread internals

| artifact | what is known | what is missing |
|---|---|---|
| `dcnet/fashioniq_run_directory/trained_model.pth` (2,877,310,172 B) | sha256 `130f031e0f78…` matches declared; it is a **legacy non-zip pickle** | the class/key structure. A legacy pickle cannot be read without executing it; the stubbed reader returns nothing. |
| `clvc_net/author_drive_unresolved` (6 files, 176–218 MB each) | all six digests match `selected_bundle_digests.json` | whether each is a whole model, a state_dict, or a partial (the names `dress_local_model.pt`/`dress_global_model.pt` imply the CLVC local/global split) |
| `tgcir/tgcir_archive_unresolved/Fashioniq.pt` (607,928,637 B) | sha256 `c37768f3…` matches declared | internal structure |

**Would settle it:** run the stubbed-pickle reader (same method used for the other 17) over the CLVC and TG-CIR files; for DCNet, `torch.load(weights_only=False)` in a throwaway sandbox with no side effects, or adopt a legacy-pickle opcode walker.

### A3. Four declared variants have no digest and no bytes

`conesep/fiq_n05`, `conesep/fiq_n08`, `sprc/sprc_fiq_unresolved`, `tme/cirr_0` declare `expected_sha256: null` **and** are absent from the resolved root. So there is nothing to verify and nothing to evaluate; the registry cannot even state what artifact would satisfy them.

**Would settle it:** author download with a recorded digest, or an explicit `unsupported` declaration in the registry. (`conesep fiq_n08` also carries the filename inconsistency `ConeSep-FIQ_N0.8.pt` vs the `-N0.2`/`-N0.5` siblings.)

## B. Provenance gaps (affect how strong the pin is)

### B1. Twelve of sixteen repos have no `.git`, so the pinned commit is a record, not a measurement

`workbench/third_party/*` contains **no** nested `.git` — the vendor step stripped them and kept a `provenance.json` with `upstream_commit` plus a per-file `file_manifest` of sha256s. For those 12 repos I could verify the **content** (every manifest file re-hashed; 12 trees match 100%) but not that `upstream_commit` names the right commit in the upstream remote, because there is no object database to ask.

**Would settle it:** `git ls-remote <url>` for each URL, then one shallow fetch to confirm the commit exists and its tree hashes to the manifest. Or re-vendor with `.git` retained.

### B2. `teacher/repos/TME` is a different repository, and the pinned commit is absent from it

| field | value |
|---|---|
| registry / provenance pin | `https://github.com/li-shuxian/TME` @ `394aad79df2c104feb32a2c96132582012765062` |
| `teacher/repos/TME` remote | `https://github.com/He-Changhao/2025-CVPR-TME.git` |
| `teacher/repos/TME` HEAD | `7fe811992820d30828e24065eb0c7a7ba099dd3b` (2025-06-09) |
| `git cat-file -t 394aad79…` in that clone | `fatal: could not get object info` |

The vendored `workbench/third_party/TME` is internally consistent (349/351 manifest files match, 2 diffs = the 2 approved patches), so **execution is not affected** — but any statement of the form "the TME checkout is at the pinned commit" is false for `teacher/repos/TME`. The two trees also differ in `README.md` and `precompute.sh` (the fork points at FashionIQ; the pinned original at CIRR), which means the fork is not merely a re-clone at a different commit.

**Would settle it:** vendor `li-shuxian/TME@394aad79` into `teacher/repos/`, or delete that clone and record `third_party` as the only TME source.

### B3. `teacher/repos/CSMCIR` holds a third state of a patched file

| locus | sha256 |
|---|---|
| declared base | `9d8c914b551a7de6…` |
| declared derived | `0ec134431697af6f…` |
| `third_party/CSMCIR/src/lavis/models/__init__.py` | `0ec134431697af6f…` = **derived, correct** |
| `teacher/repos/CSMCIR/src/lavis/models/__init__.py` (working tree) | `8e9cd6a8303bc0c6…` = **neither** |
| `teacher/repos/CSMCIR` @ HEAD `git show` | `9d8c914b551a7de6…` = base |

So the teacher clone has 102 added / 102 removed lines of uncommitted edit in that file, and its `src/validate_blip_csmcir.py` is at **base** while the vendored copy is at **derived** — i.e. the teacher tree is patched in a different place than the approved set. The adapters launch `third_party`, so this does not affect evaluation today.

**Would settle it:** `git -C teacher/repos/CSMCIR checkout -- src/lavis/models/__init__.py`, or record the teacher variant as its own approved patch with base/derived digests.

### B4. Two non-model git checkouts exist outside every count

`teacher/repos/SPN4CIR` (`BUAADreamer/SPN4CIR` @ `84005f643ccaacf999982694ad5631df92cef098`, clean) and `data/cirr` (`Cuberick-Orion/CIRR` @ `2276fba2d2c53862c8ae79ea18e1f1ae03719f56`, clean) are real upstream clones with no registry entry. `SPN4CIR` is a CIR method with a README but no adapter, no provenance, no environment, and is referenced by exactly one file (`teacher/outputs/full_audit/tgcir/full_metrics.json`).

**Would settle it:** a decision — register SPN4CIR as an 18th variant, or delete the clone. This audit takes no position; it only records that the "16" is a *registered* count.

## C. Evidence gaps (affect how the numbers were derived)

### C1. No training-run metadata inside any checkpoint

For the nine **whole-model pickles** the producing class name is readable (`Blip2QformerCir`, `ConeSep`, `HABIT`, `HINT`, `LIMN`), and for five more an `epoch` + class-name wrapper is readable. But no checkpoint carries the upstream revision or command that produced it. So "this state_dict came from class X" is established; "this state_dict came from commit Y" is not, for **any** variant.

**Would settle it:** for the LAVIS family, compare the parameter-name sets against each candidate revision's class definition — a name-set match to the 861/857/628/669 counts is strong evidence and does not require running the model.

### C2. The registry on disk is uncommitted-dirty, so "the registry" needs a version

HEAD is `b2fca876` (2026-11-07 13:57:43 +0700); the working tree has 189 insertions / 46 deletions in `workbench/registry/models.yaml`. Counts are invariant (17 / 16 / 29 both sides, identical `checkpoint_id` lists), but 7 variants gained `expected_sha256`+`expected_size_bytes`, 5 gained `checkpoint_mapping_status: VERIFIED_METADATA`, and 5 `filename` placeholders were resolved. Both hashes are recorded in `inventory.json` (`sources.registry.sha256` and `head_sha256`).

**Would settle it:** a commit. Not this audit's call — it is read-only by instruction.

### C3. GPU capability is not independently re-probed

`workbench/artifacts/gpu_capabilities/` holds 547 probe records; a histogram of `compute_capability` is 342× `"8.9"`, 199× `null`, 6× `"12.0"`. All 6 `12.0` records (the sm_120-capable ones) are `capability_verified` and name `torch 2.7.1+cu128`; the 342 `8.9` records name a `"Synthetic GPU"` with `"error": "CUDA out of memory"` and a `pytest-of-…` interpreter, i.e. they are **test fixtures, not host measurements**. This audit started no GPU work (an evaluator is running and owns the device), so the host's real compute capability is taken from those 6 records, not measured here.

**Would settle it:** one `torch.cuda.get_device_capability()` call at a moment when no evaluator holds the GPU.

## D. Deliberately not investigated

- Whether any run **succeeded** — that is `reproduction_status`, not inventory. This audit reports wiring, not outcomes.
- `workbench/artifacts/results/**` and `logs/**` contents, beyond noting they exist.
- Float/numeral equivalence between adapters and native evaluators — the "live paths" column says a path is *wired*, never that the two agree numerically.
- The 547 `gpu_capabilities` JSONs individually; only the aggregate and the 6 verified records were read.

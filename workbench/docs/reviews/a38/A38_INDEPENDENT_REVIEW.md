## R1 (attack pairs + digests; framing ambiguity verdict; invalidation path)

**Claim:** per-command ASCII byte-length + `:` + `shlex.join(cmd)` distinguishes command boundaries.

**Attacks / raw digest evidence** (SHA-256 over updates, reimplemented in scratch):

- Old collision: stage `x`, commands `[ ["ab"] ]` vs `[ ["a"], ["b"] ]`: old digest both `8906bb0efdf319cfb033b78e340740dd50a8df05aca21cc1ee67af2eed186d98`. New: `0b7c0866caeb7bb98763aa9f2d025dc834bb8485b1153c93a7d6600ca5306630` vs `e76ae0c590b57827cf0d8a91edc1ab398a16a11cfb187f7c72e4a08c2ac9a112`.
- Stage boundary survives command framing attack: stage `x`, command `y` vs stage `xy`, no commands: new digest `5c5f9769614e2bdfefc5997daebf40379d12b5f922d42600e5a473c124c6006e` vs `769a4e6d0003189c7e96c5d9b7e810a0d11c3a12832527ec94b0f86d277f51ca`.
- Prefix attack: stage `s`, `model_id=abc`, no command vs stage `s`, no model, command `MODEL:abc`: `c2c651410469202590f62b704e1f1ddf1b309e357d97e23cae04e872efbce425` vs `d042b785732f3d61c0597e911ad3e1b3ae0fb90d9e293b25c8213592cce57023` (different).
- **Remaining collision:** stage `sMODEL:abc`, no model/command vs stage `s`, `model_id=abc`, no command both hash exactly `c2c651410469202590f62b704e1f1ddf1b309e357d97e23cae04e872efbce425`. `stage.name` is fed unframed before `MODEL:`. Similar ambiguity exists between unframed stage name and later prefix fields. The per-command length prefix fixes command-to-command concatenation only; it does not make the whole serialization injective.

**Invalidation trace:** pipeline computes current fingerprint, only sets `can_skip` when prior status is `COMPLETE`, fingerprint equals, and outputs validate (`pipeline.py:2272-2287`). A changed fingerprint therefore leaves `can_skip=False`; runner writes a new `RUNNING` record (`:2307-2325`) and executes commands. Old fingerprints mismatch safely: rerun, not skip or crash.

**Verdict: DEFECTIVE.** Concrete counterexample: `sMODEL:abc` stage/no model collides with `s` + model `abc`; identifiers are unframed. (The command-boundary fix itself works.)

## R2 (validator bypass list; lock scope/finally/self-deadlock/no-op verdict)

**Validator attacks** (`runner.py:67-73`): empty ranks dict rejected when `conditions` nonempty (`set(ranks) != set(conditions)`); condition value `{}` rejected when `qids` nonempty; `True` rejected (`type(True) is not int`); `1.0` rejected; huge integer rejected above gallery count; non-string qid rejected by exact key-set comparison to query qids; extra qid rejected. No tested bypass reaches metrics. Valid `None` rank remains accepted intentionally. An empty qid set plus empty per-condition mappings is accepted; runner rejects `not queries` earlier (`:587-591`), so not a live cache case.

**Lock:** opened at `runner.py:685-686`, within the per-model `try` begun at `:630`; normal release before metrics at `:872-874`; `finally` release at `:1009-1010` covers exceptions thereafter. Lock pathname is passed through `assert_composition_destination`, which resolves and requires it inside `artifacts/composition` (`:46-53`). Search found one acquisition call site and no nested acquire of same lock in-process; no self-deadlock path found. Rare exception from `flock` after successful `os.open` can leak descriptor: assignment to `model_lock` does not complete, so `finally` sees `None`.

No detection of ineffective `flock`: code directly calls `fcntl.flock` and assumes exclusion (`:76-79`). On a filesystem with no-op locking, simultaneous workers may compute/write the same rank cache and race atomic result replacement / attempt bookkeeping. That is residual risk, not a local bypass.

**Verdict: SOUND_WITH_CAVEAT.** Residual risk: network filesystem no-op locking is undetected; additionally, `flock` raising after open leaks descriptor.

## R3 (legitimate-checkpoint risk verdict + error path)

**Claim:** any missing or unexpected model-state key fails closed.

Vendored training `utils_csmcir.py:478-482` saves `{'epoch': cur_epoch, model_name: model_to_save.state_dict()}`; training call uses `save_model('tuned_clip_best', epoch, blip_model, ...)` (`blip_fine_tune_2_csmcir.py:297`); vendored validator loads `checkpoint[blip_model.__class__.__name__]` with `strict=False` (`validate_blip_csmcir.py`, matching grep evidence; also `cirr_test_submission_csmcir.py:215-217`). Worker likewise selects `state[model.__class__.__name__]` (`model_csmcir.py:218-219`), so wrapper/epoch and class-name handling match. No DataParallel use or intentional frozen-key omission found in this training path. Model has registered parameters/modules and inherited buffers; ordinary `state_dict()` includes persistent buffers and excludes non-persistent ones on both save/load.

Registry records CSMCIR checkpoint `checkpoint_mapping_status: VERIFIED_METADATA`, expected hash/size, but this is metadata verification, not an observed mapping test (`models.yaml:110-125`). No downloaded 2.72-GB checkpoint or conversion script available here to compare the actual state keys. Therefore cannot establish actual public checkpoint's exact keys or absence of DataParallel prefix solely from source. `strict=False` upstream suggests historical permissiveness but does not prove missing/unexpected keys exist.

**Error path:** `_load_csmcir_checkpoint` calls `load_state_dict(..., strict=False)` then `_check_checkpoint_load`, raising `ValueError` for either list (`model_csmcir.py:185-197`). Worker catches model-load exceptions and calls `reply(False, error_type=..., error=...)`, then returns (`:215-222`). Adapter sees nonzero subprocess status and raises `AdapterWorkerError` (`:150-155`). Runner catches model errors, writes failed results (`runner.py:976-984`), and `finally` deletes per-model work root (`:987-1004`) and releases lock (`:1009-1010`); no worker crash escapes through this path.

**Verdict: SOUND_WITH_CAVEAT.** Residual risk: actual author checkpoint tensor keys were not inspected; a legitimate checkpoint with upstream-tolerated omissions would now be rejected (safe against garbage, potentially unusable).

## R4 (_shell_code counterexamples; per-test comment-only FAIL proof; brittleness)

**Scanner probes, raw returned code:**

- `echo "#keep" #cut` → `echo "#keep" `; `echo '#keep' #cut` → `echo '#keep' `: expected.
- Escaped quote / continuation / heredoc / ANSI / shebang / substitution: `echo \\"#keep` → `echo \\"`; `echo foo \\\n#continued` → `echo foo \\`; `cat <<EOF\n#payload\nEOF` → `cat <<EOF\nEOF`; `echo $'\#not-comment' #cut` → `echo $'\#not-comment' `; `#!/bin/bash` → empty; `echo $(echo ok #comment\n echo next)` → `echo $(echo ok \n echo next)`.
- Definite mis-strip: `echo "foo\\\n#bar"` (backslash continuation inside a still-open double quote) is scanned line-by-line; quote state resets at newline and `#bar"` is removed although shell treats it as quoted content. Thus a required literal/token on that continued quoted line can be falsely absent. Conversely heredoc body tokens remain counted when they are data, not executed shell; implementation documents this knowingly, so comment-only heredoc payload can satisfy presence assertions if a future invariant token is checked only by `_shell_code`.
- `$'...'` handling is not a shell parser (e.g. escape sequences that contain quote syntax), though the probes do not show a specific current invariant bypass. Shebang loss is harmless for current token checks.

**Scratch mutation evidence:** rewrote only scratch copies of the relevant files to contain every tested token on a `#` comment line; invoking each test function produced `AssertionError`:
`test_i11_an_invalid_mount_cannot_reach_evaluation FAIL: AssertionError`; `test_i13_no_unverified_gpu_record_produces_gpu_runtime_pass FAIL: AssertionError`; `test_i14_no_synthetic_inference_produces_model_inference_pass FAIL: AssertionError`; `test_i17_an_api_error_never_becomes_an_empty_success FAIL: AssertionError: a ledger that exists but does not verify must be distinguishable from an empty workspace`; `test_i19_a_harness_does_not_pass_without_its_infrastructure FAIL: AssertionError`; `test_i20_a_verdict_never_claims_more_than_its_evidence FAIL: AssertionError`.

**Real-file evidence:** invoked all six corresponding test functions against repository files; each returned `PASS` (I11, I13, I14, I17, I19, I20).

**Brittleness:** several assertions demand exact Python source substrings, including I11 `splitlines()`/`{{.Source}}=>{{.Destination}}{{"\\n"}}`, I19 full status expressions and I20 full verdict/exit expressions. Harmless formatting, quote-style, or equivalent refactoring breaks tests despite unchanged behavior. I11 also asserts source text `assert len(ro) >= 2`, coupling to implementation spelling.

**Verdict: SOUND_WITH_CAVEAT.** Residual risk: scanner erases real quoted shell content after line continuation; documented heredoc handling can let heredoc data satisfy future checks. Comment-only tests do fail on scratch mutation and all six real checks pass.

## verdicts table

| Fix | Verdict |
|---|---|
| R1 pipeline command fingerprint | DEFECTIVE |
| R2 rank-cache validation / lock | SOUND_WITH_CAVEAT |
| R3 CSMCIR checkpoint strictness | SOUND_WITH_CAVEAT |
| R4 release invariant comment stripping | SOUND_WITH_CAVEAT |

## anything the four fixes BREAK that the authors did not mention

- R1 still aliases distinct stage metadata due to unframed `stage.name` + `MODEL:`/`CKPT:`/`PROTO:` fields; demonstrated collision above. Ordinary generated names may not currently exploit it, but the serializer contract is not unambiguous.
- R2's per-model critical section increases contention as documented; no second acquisition exists in current process path. The uncovered filesystem semantics and exceptional-open descriptor leak are caveats rather than observed production failures.
- R3 turns any formerly tolerated upstream checkpoint key mismatch into worker failure. Existing checkpoint metadata and vendored save source do not settle actual downloaded mapping; no claim of observed breakage.
- R4 exact-source assertions create maintenance failures on behavior-preserving formatting changes; continuation scan false-negative is concrete.

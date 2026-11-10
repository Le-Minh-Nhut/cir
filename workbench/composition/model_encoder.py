"""ENCODER FashionIQ adapter and model-interpreter worker.

Runner sets ``work_root`` after construction, before encoding; encode methods refuse with
``AdapterWorkRootError`` when unset. Each instance creates a unique
``<work_root>/encoder/worker-*/`` directory, and ``close()``/``restart()`` remove only that
directory, never the run root. Model-runtime imports stay in the child process. Construction
mirrors ``workbench/replay/encoder.py:253-295`` because replay's loader is not reusable without a dataset.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import (
    checkpoint_by_id,
    checkpoint_is_present,
    checkpoint_path,
    model_by_id,
)
from workbench.backend.model_settings import setting
from workbench.backend.runtime import resolve_model_interpreter
from workbench.composition.adapters import (
    AdapterStatus,
    AdapterWorkerError,
    CompositionAdapter,
    limit_torch_threads,
    prefetch_rows,
    resolved_decode_workers,
    worker_environment,
)
from workbench.composition.inputs import CONDITIONS

#: Seed pinned immediately BEFORE `model_try2.Encoder(...)` in the worker.
#:
#: `BindingDecoder.learnable_relation` is `nn.Parameter(torch.randn(N_p, dim)).cuda()`
#: (`third_party/ENCODER/model_try2.py:93`). `.cuda()` on a Parameter returns a plain Tensor
#: subclass instance, so the attribute is not registered, is absent from `state_dict()` and from the
#: 417-key released checkpoint, and no seeded construction covered it: without this pin two clean
#: processes draw different relations (measured max |Δ| 4.11787748336792 / 4.744387149810791 between
#: two runs of `raw/m01/m01_unseeded_construct.py`, and 4.923079013824463 in the earlier probe at
#: `20261110-e2e-repair/raw/r4_determinism_p1.json`) and the relation enters BOTH the query
#: (`extract_retrieval_compose`) and the gallery (`extract_retrieval_target`) representations.
#:
#: Pinned rather than "reproducible by luck": with this seed the same tensor is drawn in every clean
#: process (digest 20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92 for
#: shape [2, 512] under torch 2.7.1+cu128), and the drawn bytes plus their digest are persisted with
#: the run so a reader never has to reproduce a draw to know what was used.
RELATION_SEED = 20261111

#: The digest of the tensor that seed draws under the pinned interpreter, recorded so a torch RNG
#: change is a visible difference rather than a silent one. Not an assertion: a future interpreter
#: that draws different bytes is reported (`relation_digest_matches_pin: false`) and must mint a new
#: run, never be treated as the same measurement.
RELATION_EXPECTED_DIGEST = "20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92"


class Adapter(CompositionAdapter):
    """Run ENCODER in its registered Python environment; no reference-image score mask."""

    model_id = "encoder"
    category_scopes = (None,)
    reference_eligible = True
    # score() is a plain normalized dot product over 512-d features.
    score_dimension = 512
    #: The transform policy this adapter ACTUALLY applies, declared so it travels with the run.
    #: Native ENCODER feeds the query through `transform[0]` = preprocess_train and the gallery
    #: through `transform[1]` = preprocess_val (`third_party/ENCODER/datasets.py:177` vs `:195,202`).
    #: CIRCUS composition uses preprocess_val for BOTH sides -- a deliberate adaptation, not an
    #: oversight, and not something to "fix" silently: switching it would move every published
    #: number. Measured sensitivity is in `20261111-master-audit/raw/m01/transform_control.txt`.
    transform_policy = {
        "query_transform": "preprocess_val",
        "gallery_transform": "preprocess_val",
        "native_query_transform": "preprocess_train",
        "native_gallery_transform": "preprocess_val",
        "kind": "declared adaptation",
        "note": ("CIRCUS composition uses preprocess_val for both query and gallery; native ENCODER "
                 "uses preprocess_train (RandomResizedCrop) for the query and preprocess_val for the "
                 "gallery. Keeping val for both is deliberate: the asymmetry is a property of the "
                 "native release, and honouring it here would change every published number."),
    }

    def __init__(self, config: Any, checkpoint_id: str | None = None) -> None:
        super().__init__(config, checkpoint_id)
        self._model_record = model_by_id(self.model_id)
        self._checkpoint_id = checkpoint_id or self._model_record.get("per_query_checkpoint", "fashioniq")
        self._checkpoint = checkpoint_by_id(self._model_record, self._checkpoint_id)
        self._checkpoint_path = checkpoint_path(
            self.model_id, self._checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
        variable = self._model_record["environment"]["python_env_var"]
        self.interpreter = resolve_model_interpreter(self._model_record, config) or setting(variable)
        self.source_root = Path(config.WORKBENCH_THIRD_PARTY_ROOT) / self._model_record["source_dir"]
        self.dataset_root = Path(config.FASHIONIQ_ROOT)
        self._counter = 0
        #: What the worker actually drew for `learnable_relation` (None until the first reply).
        self.relation_state: dict[str, Any] | None = None
        self._job_dir: Path | None = None
        self._worker: subprocess.Popen[str] | None = None
        self._worker_log: Any | None = None
        self._worker_failed: str | None = None
        self._interpreter_verified = False

    @classmethod
    def status(cls, config: Any | None = None) -> AdapterStatus:
        """Check artifact, registered interpreter, interpreter CUDA, and mapping without model loading."""
        config = config or resolve_config()
        model = model_by_id(cls.model_id)
        checkpoint_id = model.get("per_query_checkpoint") or model["checkpoint_variants"][0]["checkpoint_id"]
        checkpoint = checkpoint_by_id(model, checkpoint_id)
        path = checkpoint_path(cls.model_id, checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
        checkpoint_present = checkpoint_is_present(path, checkpoint, model_id=cls.model_id, config=config)
        variable = model["environment"]["python_env_var"]
        setting_value = setting(variable)
        interpreter = resolve_model_interpreter(model, config) or setting_value
        interpreter_present = bool(interpreter and Path(interpreter).is_file())
        cuda_present = False
        if interpreter_present:
            try:
                probe = subprocess.run(
                    [str(interpreter), "-c",
                     "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"],
                    capture_output=True, text=True, check=True)
                cuda_present = probe.stdout.strip().split()[-1] == "True"
            except (OSError, subprocess.SubprocessError, IndexError):
                pass
        mapping = checkpoint.get("checkpoint_mapping_status")
        checks = (f"checkpoint={'passed' if checkpoint_present else 'failed'}; "
                  f"interpreter={'passed' if interpreter_present else 'failed'} "
                  f"({variable}={'set' if setting_value else 'unset'}); "
                  f"GPU={'passed' if cuda_present else 'failed'}; "
                  f"mapping={'passed' if mapping == 'VERIFIED_METADATA' else 'failed'} ({mapping})")
        if not checkpoint_present:
            reason = f"checkpoint missing or integrity check failed: {path}"
        elif not interpreter_present:
            reason = f"model interpreter unavailable: {variable} is {'set but missing' if setting_value else 'unset'}"
        elif not cuda_present:
            reason = "CUDA unavailable in ENCODER model interpreter; native forward requires CUDA"
        elif mapping != "VERIFIED_METADATA":
            reason = f"checkpoint mapping unresolved: {mapping}"
        else:
            source = Path(config.WORKBENCH_THIRD_PARTY_ROOT) / model["source_dir"]
            auxiliary = source / "open_clip_pytorch_model.bin"
            reason = None if auxiliary.is_file() else f"ENCODER auxiliary asset missing: {auxiliary}"
        if reason is None:
            # F1(c): every presence check above passes on an interpreter the execution gate still
            # refuses (measured: pair-adapted is Python 3.11.15 against a declared 3.10). Ask the
            # gate so `--plan` cannot print READY for a run that dies at launch.
            from workbench.composition.adapters import _environment_gate_reason

            reason = _environment_gate_reason(model, config)
        return AdapterStatus(
            model_id=cls.model_id, ready=reason is None, reason=reason,
            checkpoint_ids=(checkpoint_id,),
            per_mode={condition: (True, None) for condition in CONDITIONS}, notes=checks)

    def construction_state_root(self) -> Path:
        """Where the drawn `learnable_relation` is written: the RUN's own directory, not `work/`.

        Measured on the pinned rerun: the runner's keep-features cleanup removes every child of
        `work/<model>/` except `features/` (`runner.py:2263-2271`), so a state record placed under
        `work/` is deleted at the end of the run that produced it -- exactly the artifact a reader
        needs afterwards. `runs/<run_id>/` outlives the run, so it lives here, beside `ranks/`.
        """
        # `work_root` is `<run>/work/<model>`; the run directory is two levels up.
        return Path(self.require_work_root()).parents[1] / "construction_state"

    def _image_paths(self, image_paths: Sequence[str | Path]) -> list[str]:
        paths = [str(Path(path).expanduser().resolve()) for path in image_paths]
        missing = [path for path in paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"ENCODER input image missing: {missing[0]}")
        return paths

    def _run_job(self, phase: str, image_paths: Sequence[str | Path],
                 texts: Sequence[str] = (), categories: Sequence[str | None] = ()) -> np.ndarray:
        work_root = self.require_work_root()
        if not self.interpreter or not Path(self.interpreter).is_file():
            variable = self._model_record["environment"]["python_env_var"]
            raise RuntimeError(f"model interpreter unavailable: {variable} is unset or missing")
        if not self._interpreter_verified:
            # Same gate, same code path, as every other adapter: see
            # CompositionAdapter.require_verified_interpreter.
            self.require_verified_interpreter()
            self._interpreter_verified = True
        if self._worker_failed is not None:
            raise AdapterWorkerError(self._worker_failed)
        adapter_root = work_root / self.model_id
        adapter_root.mkdir(parents=True, exist_ok=True)
        if self._job_dir is None:
            self._job_dir = adapter_root / f"worker-{uuid.uuid4().hex}"
            self._job_dir.mkdir()
        self._counter += 1
        job_path = self._job_dir / f"job-{self._counter}.json"
        output_path = self._job_dir / f"output-{self._counter}.npy"
        scope = ",".join(sorted({str(value) for value in categories})) or "global"
        payload = {
            "phase": phase, "image_paths": self._image_paths(image_paths),
            # RUN-scoped, OUTSIDE `work/`: the runner's keep-features cleanup deletes every child of
            # `work/<model>/` except `features/`, so a record inside it is deleted at the end of the
            # very run it describes. The drawn tensor is the only record of state `state_dict()`
            # cannot carry, so it belongs beside `ranks/` and `candidates/`, not under `work/`.
            "relation_root": str(self.construction_state_root()),
            "texts": list(texts), "categories": list(categories),
            "output_path": str(output_path), "checkpoint_path": str(self._checkpoint_path),
            "expected_sha256": self._checkpoint.get("expected_sha256"),
            "checkpoint_mapping_status": self._checkpoint.get("checkpoint_mapping_status"),
            "source_root": str(self.source_root), "dataset_root": str(self.dataset_root),
            "batch_size": self.batch_size,
            # The one unseeded draw in this model is pinned by the WORKER, immediately before the
            # model is constructed; the seed travels with the job so it is never a module default
            # that a different caller could inherit.
            "relation_seed": RELATION_SEED,
            "relation_expected_digest": RELATION_EXPECTED_DIGEST,
            "transform_policy": self.transform_policy,
        }
        job_path.write_text(json.dumps(payload), encoding="utf-8")
        if self._worker is not None and self._worker.poll() is not None:
            status = self._worker.returncode
            self._worker_failed = (
                f"adapter_worker_exited: model=encoder scope={scope} phase={phase} exit_status={status}")
            self._stop_worker()
            raise AdapterWorkerError(self._worker_failed)
        if self._worker is None:
            self._worker_error = self._job_dir / "worker.stderr.log"
            self._worker_log = self._worker_error.open("w", encoding="utf-8")
            self._worker = subprocess.Popen(
                [self.interpreter, "-m", "workbench.composition.model_encoder",
                 "--worker-job", str(job_path)], cwd=self.config.CIR_REPO_ROOT,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._worker_log,
                text=True, bufsize=1, env=worker_environment())
        else:
            assert self._worker.stdin is not None
            self._worker.stdin.write(json.dumps({"job_path": str(job_path)}) + "\n")
            self._worker.stdin.flush()
        assert self._worker.stdout is not None
        line = self._worker.stdout.readline()
        while line and not line.startswith("CIRCUS_WORKER:"):
            line = self._worker.stdout.readline()
        if not line:
            worker = self._worker
            self._stop_worker()
            status = worker.returncode if worker is not None else None
            self._worker_failed = (
                f"adapter_worker_exited: model=encoder scope={scope} phase={phase} exit_status={status}")
            raise AdapterWorkerError(self._worker_failed)
        response = json.loads(line[len("CIRCUS_WORKER:"):])
        if not response.get("ok"):
            job_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            self._worker_failed = (
                f"adapter_worker_job_failed: model=encoder scope={scope} phase={phase} "
                f"error={response.get('error_type')}: {response.get('error')}")
            self._stop_worker()
            raise AdapterWorkerError(self._worker_failed)
        # Recorded, not asserted: a future interpreter that draws different bytes is a new
        # measurement, and the flag says so instead of pretending the pin did not move.
        if response.get("relation_digest"):
            self.relation_state = {
                "seed": response.get("relation_seed"),
                "sha256": response["relation_digest"],
                "matches_pinned_digest": response["relation_digest"] == RELATION_EXPECTED_DIGEST,
                "expected_sha256": RELATION_EXPECTED_DIGEST,
                "in_state_dict": False,
                "transform_policy": self.transform_policy,
                "path": str(self.construction_state_root()),
            }
        result = np.load(output_path, allow_pickle=False).astype(np.float32, copy=False)
        if result.ndim != 2 or result.shape != (len(image_paths), 512):
            raise AdapterWorkerError(
                f"adapter_worker_invalid_matrix: model=encoder scope={scope} phase={phase} "
                f"shape={result.shape} expected=({len(image_paths)}, 512)")
        job_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        return result

    def encode_queries(self, image_paths: Sequence[str | Path], texts: Sequence[str],
                       categories: Sequence[str | None]) -> np.ndarray:
        self.require_work_root()
        if len(image_paths) != len(texts) or len(image_paths) != len(categories):
            raise ValueError("ENCODER query image, text, and category counts must match")
        return self._run_job("queries", image_paths, texts, categories)

    def encode_targets(self, image_paths: Sequence[str | Path],
                       categories: Sequence[str | None]) -> np.ndarray:
        self.require_work_root()
        if len(image_paths) != len(categories):
            raise ValueError("ENCODER target image and category counts must match")
        if any(category is not None for category in categories):
            raise ValueError("ENCODER target scope must be None")
        return self._run_job("targets", image_paths, categories=categories)

    def score(self, queries: Any, targets: Any,
              categories: Sequence[str | None] | None = None) -> np.ndarray:
        """Normalized-dot scores, unmasked; native replay `test.py:67-68` mask is not inherited."""
        queries = np.asarray(queries, dtype=np.float32)
        targets = np.asarray(targets)
        query_norm = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / np.maximum(query_norm, np.finfo(np.float32).tiny)
        scores = np.empty((len(queries), len(targets)), dtype=np.float32)
        for start in range(0, len(targets), 8192):
            stop = min(start + 8192, len(targets))
            gallery = np.asarray(targets[start:stop], dtype=np.float32)
            norms = np.linalg.norm(gallery, axis=1)
            scores[:, start:stop] = (queries @ gallery.T) / np.maximum(
                norms, np.finfo(np.float32).tiny)[None, :]
        return scores

    def _stop_worker(self) -> None:
        worker = self._worker
        if worker is not None:
            if worker.stdin is not None:
                try:
                    worker.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            if worker.poll() is None:
                worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
            self._worker = None
        if self._worker_log is not None:
            self._worker_log.close()
            self._worker_log = None

    def close(self) -> None:
        if self._worker is not None:
            if self._worker.stdin is not None:
                try:
                    self._worker.stdin.write(json.dumps({"stop": True}) + "\n")
                    self._worker.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            try:
                self._worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._worker.terminate()
                try:
                    self._worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._worker.kill()
                    self._worker.wait()
            self._worker = None
        if self._worker_log is not None:
            self._worker_log.close()
            self._worker_log = None
        if self._job_dir is not None:
            # A latched failure leaves the job directory for the runner's preservation copy of
            # worker.stderr.log; the runner's rmtree removes it. Only a clean close deletes here.
            if self._worker_failed is None:
                shutil.rmtree(self._job_dir, ignore_errors=True)
            self._job_dir = None

    def restart(self) -> None:
        """Kill worker, discard its private job directory, and reload on next encode call."""
        self.close()

def _worker_main(job_path: Path) -> None:
    """Load verified ENCODER weights once, process jobs, and keep stdout for replies."""
    import torch
    import open_clip
    from PIL import Image

    from workbench.replay.common import load_trusted_whole_model
    from workbench.replay.encoder import _assert_checkpoint_loaded, _load_replay_modules
    def reply(ok: bool, **values: Any) -> None:
        print("CIRCUS_WORKER:" + json.dumps({"ok": ok, **values}), flush=True)
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        source_root = Path(job["source_root"])
        # The mapping status is one of the four gates `_unused_checkpoint_keys` applies before the
        # two provably-unread `binding_decoder.fc.0.*` tensors may be skipped. Asserting
        # `VERIFIED_METADATA` here meant a registry downgrade could never tighten the loader; take
        # the registry's value instead, so a downgrade reaches this decision by construction.
        checkpoint = {"checkpoint_id": "fashioniq", "expected_sha256": job["expected_sha256"],
                      "checkpoint_mapping_status": job.get("checkpoint_mapping_status")}
        torch_module, open_clip_module, _datasets, _test, model_try2 = _load_replay_modules(source_root)
        # The draw that no checkpoint can restore happens inside `Encoder(...)`: pin it HERE, before
        # construction, and before anything else in the process can consume the RNG.
        relation_seed = int(job["relation_seed"])
        torch.manual_seed(relation_seed)
        previous_cwd = Path.cwd()
        try:
            os.chdir(source_root)
            model = model_try2.Encoder(
                hidden_dim=512, dropout=0.5, local_token_num=3, t=0.1,
                wc=2, N_p=2, weighted=True)
        finally:
            os.chdir(previous_cwd)
        # Persist the tensor that was actually drawn, not a claim about it: `learnable_relation` is
        # not in `state_dict()`, so this file plus its digest is the only record of the state this
        # run's query and gallery features were both computed with.
        relation = model.backbone.binding_decoder.learnable_relation.detach().float().cpu().contiguous()
        import hashlib
        relation_digest = hashlib.sha256(relation.numpy().tobytes()).hexdigest()
        # Written under the RUN's work root, not the worker's job directory: `close()` deletes the
        # latter, and this record must outlive the worker that drew it.
        relation_root = Path(job["relation_root"])
        relation_root.mkdir(parents=True, exist_ok=True)
        relation_path = relation_root / f"relation-seed{relation_seed}.npy"
        np.save(relation_path, relation.numpy(), allow_pickle=False)
        Path(str(relation_path) + ".json").write_text(json.dumps({
            "tensor": "backbone.binding_decoder.learnable_relation",
            "shape": list(relation.shape), "dtype": str(relation.dtype),
            "seed": relation_seed, "sha256": relation_digest,
            "seed_placement": "torch.manual_seed(seed) immediately before Encoder(...) construction",
            "in_state_dict": False,
            "transform_policy": job.get("transform_policy"),
            "source": "workbench/third_party/ENCODER/model_try2.py:93 (torch.randn(N_p, dim))",
            "permutation_scope": "whole model construction (CLIP tower, FactorNet, decoder heads and "
                                 "this tensor all draw from the seeded generator); the pin is what "
                                 "makes the run reproducible, not a claim that the released relation "
                                 "was recoverable",
        }, indent=1) + "\n", encoding="utf-8")
        state = load_trusted_whole_model(
            Path(job["checkpoint_path"]), job["expected_sha256"], torch_module)
        if isinstance(state, dict):
            report = model.load_state_dict(state, strict=False)
            _assert_checkpoint_loaded(model, report, checkpoint, source_root)
        else:
            model = state
        model = model.cuda().eval()
        _, _, preprocess_val = open_clip_module.create_model_and_transforms(
            "ViT-B-32", pretrained=str(source_root / "open_clip_pytorch_model.bin"))
        limit_torch_threads()
        workers = resolved_decode_workers()
    except Exception as error:
        reply(False, error_type=type(error).__name__, error=str(error))
        return

    def run_one(current: dict[str, Any]) -> None:
        paths, texts = current["image_paths"], current["texts"]
        features = []
        batch_size = max(1, int(current.get("batch_size") or 32))
        batches = [paths[start:start + batch_size]
                   for start in range(0, len(paths), batch_size)]

        def decode(chunk: Sequence[str]) -> Any:
            images = []
            for path in chunk:
                with Image.open(path) as image:
                    images.append(preprocess_val(image.copy()))
            return torch.stack(images)

        try:
            # Same decode pool as the rest of the family. This model's open_clip transform
            # (`preprocess_val`) is used unchanged, and so is the forward.
            for index, batch in enumerate(prefetch_rows(decode, batches, workers)):
                start = index * batch_size
                stop = start + len(batch)
                with torch.no_grad():
                    batch = batch.cuda()
                    if current["phase"] == "queries":
                        encoded = model.extract_retrieval_compose(batch, texts[start:stop])
                    else:
                        encoded = model.extract_retrieval_target(batch)
                features.append(encoded.detach().float().cpu().numpy())
            result = np.concatenate(features, axis=0) if features else np.empty((0, 512), dtype=np.float32)
            np.save(current["output_path"], result.astype(np.float32, copy=False), allow_pickle=False)
            reply(True, relation_digest=relation_digest, relation_seed=relation_seed)
        except Exception as error:
            reply(False, error_type=type(error).__name__, error=str(error))

    run_one(job)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("stop"):
            break
        try:
            run_one(json.loads(Path(request["job_path"]).read_text(encoding="utf-8")))
        except Exception as error:
            reply(False, error_type=type(error).__name__, error=str(error))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker-job":
        raise SystemExit("usage: python -m workbench.composition.model_encoder --worker-job JOB.json")
    _worker_main(Path(sys.argv[2]))


ADAPTER = Adapter

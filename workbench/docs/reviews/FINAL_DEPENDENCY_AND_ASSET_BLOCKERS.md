# Final Dependency and Asset Blocker Inventory

Complete dependency and non-Python asset blocker inventory across all 14 pinned source trees (covering 16 registered models) in the CIR Failure Analysis Workbench (`/home/heheboiz/data/cir`).

Every candidate top-level import was extracted mechanically via AST analysis of every `.py` file under each tree, filtered against stdlib (`sys.stdlib_module_names` in Python 3.13) and the workbench virtual environment (`.venv/bin/python`), and cross-referenced with upstream manifests (`requirements.txt`, `environment.yml`, `setup.py`) and evaluation callgraphs.

---

## 1. Dependency Inventory by Taxonomy (D1–D8)

### D1: Installable Package Missing
Packages available on PyPI or public git repositories that are reachable along the model's evaluation path (or required by imported modules) but absent from the workbench venv.

| Model | Missing Item | Item Type | Exact Source Location | Public Availability | Compatible Version | Why Needed | Fixability | Proposed Fix | Difficulty | Verification Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `csmcir` | `spacy` | Python package | `workbench/third_party/CSMCIR/src/lavis/models/blip2_models/blip2.py:205` | PUBLIC_AVAILABLE | `spacy>=3.5.0` (declared in `requirements.txt`) | Optional lemmatizer property in `Blip2Base.lemmatizer` | FULLY_FIXABLE | Install `spacy` via pip into worker venv | LOW | STATIC_AUDIT |
| `airknow` | `scikit-learn` (`sklearn`) | Python package | `workbench/third_party/AirKnow/lavis/models/blip2_models/blip2_cir.py:19` | PUBLIC_AVAILABLE | `scikit-learn==1.3.2` | K-Means clustering in BLIP2 CIR feature processing | FULLY_FIXABLE | Install `scikit-learn==1.3.2` into model venv | LOW | STATIC_AUDIT |
| `airknow` | `clip` | Python package | `workbench/third_party/AirKnow/cirr_sub_BLIP2.py:6` | PUBLIC_AVAILABLE | `git+https://github.com/openai/CLIP.git` | CLIP feature extraction in CIRR submission script | FULLY_FIXABLE | Install `openai-clip` via git URL | LOW | STATIC_AUDIT |
| `conesep` | `scikit-learn` (`sklearn`) | Python package | `workbench/third_party/ConeSep/lavis/models/blip2_models/ConeSep.py:24` | PUBLIC_AVAILABLE | `scikit-learn==1.3.2` | K-Means clustering and noise separation in ConeSep model | FULLY_FIXABLE | Install `scikit-learn==1.3.2` into model venv | LOW | STATIC_AUDIT |
| `conesep` | `clip` | Python package | `workbench/third_party/ConeSep/cirr_sub.py:6` | PUBLIC_AVAILABLE | `git+https://github.com/openai/CLIP.git` | CLIP feature extraction in CIRR submission script | FULLY_FIXABLE | Install `openai-clip` via git URL | LOW | STATIC_AUDIT |
| `habit` | `scikit-learn` (`sklearn`) | Python package | `workbench/third_party/HABIT/lavis/models/blip2_models/HABIT.py:24` | PUBLIC_AVAILABLE | `scikit-learn==1.3.2` | K-Means clustering, mutual information score, KBinsDiscretizer | FULLY_FIXABLE | Install `scikit-learn==1.3.2` into model venv | LOW | STATIC_AUDIT |
| `habit` | `open_clip` | Python package | `workbench/third_party/HABIT/lavis/models/blip2_models/HABIT.py:27` | PUBLIC_AVAILABLE | `open-clip-torch==2.24.0` | OpenCLIP backbone feature extraction | FULLY_FIXABLE | Install `open-clip-torch==2.24.0` | LOW | STATIC_AUDIT |
| `intent` | `scikit-learn` (`sklearn`) | Python package | `workbench/third_party/INTENT/lavis/models/blip2_models/blip2_cir.py:30` | PUBLIC_AVAILABLE | `scikit-learn==1.3.2` | K-Means clustering for intent clustering | FULLY_FIXABLE | Install `scikit-learn==1.3.2` into model venv | LOW | STATIC_AUDIT |
| `intent` | `matplotlib` | Python package | `workbench/third_party/INTENT/lavis/models/blip2_models/blip2_cir.py:14` | PUBLIC_AVAILABLE | `matplotlib>=3.5.0` | Unconditional import of `matplotlib.pyplot` at top level of model file | FULLY_FIXABLE | Install `matplotlib` into model venv | LOW | STATIC_AUDIT |
| `hint` | `torchaudio` | Python package | `workbench/third_party/HINT/lavis/processors/audio_processors.py:9` | PUBLIC_AVAILABLE | `torchaudio>=0.13.0` | Unconditionally imported by `lavis/processors/__init__.py:29` during `load_model_and_preprocess` | FULLY_FIXABLE | Install `torchaudio` or guard import in `audio_processors.py` | LOW | STATIC_AUDIT |
| `hint` | `moviepy` | Python package | `workbench/third_party/HINT/lavis/processors/audio_processors.py:11` | PUBLIC_AVAILABLE | `moviepy>=1.0.3` | Unconditionally imported by `lavis/processors/__init__.py:29` | FULLY_FIXABLE | Install `moviepy` or guard import in `audio_processors.py` | LOW | STATIC_AUDIT |
| `hint` | `open_clip` | Python package | `workbench/third_party/HINT/lavis/models/blip2_models/HINT.py:15` | PUBLIC_AVAILABLE | `open-clip-torch==2.24.0` | OpenCLIP vision transformer backbone | FULLY_FIXABLE | Install `open-clip-torch==2.24.0` | LOW | STATIC_AUDIT |
| `encoder` | `open_clip` | Python package | `workbench/third_party/ENCODER/evaluate_model.py:6` | PUBLIC_AVAILABLE | `open-clip-torch==2.24.0` | Vision-language feature extraction in `evaluate_model.py` and `model_try2.py:9` | FULLY_FIXABLE | Install `open-clip-torch==2.24.0` | LOW | STATIC_AUDIT |
| `pair` | `open_clip` | Python package | `workbench/third_party/PAIR/models/model.py:8` | PUBLIC_AVAILABLE | `open-clip-torch==2.24.0` | Vision-language backbone in `models/model.py` | FULLY_FIXABLE | Install `open-clip-torch==2.24.0` | LOW | STATIC_AUDIT |
| `combiner_rn50x4_noft` | `clip` | Python package | `workbench/third_party/CLIP4Cir/src/validate.py:8` | PUBLIC_AVAILABLE | `git+https://github.com/openai/CLIP.git` | CLIP RN50x4 vision and text feature extraction in `src/validate.py` | FULLY_FIXABLE | Install `openai-clip` via git URL | LOW | STATIC_AUDIT |
| `clip4cir_rn50x4_fullft` | `clip` | Python package | `workbench/third_party/CLIP4Cir/src/validate.py:8` | PUBLIC_AVAILABLE | `git+https://github.com/openai/CLIP.git` | CLIP RN50x4 vision and text feature extraction in `src/validate.py` | FULLY_FIXABLE | Install `openai-clip` via git URL | LOW | STATIC_AUDIT |
| `sprc` | `clip` | Python package | `workbench/third_party/SPRC/src/blip_validate.py:8` | PUBLIC_AVAILABLE | `git+https://github.com/openai/CLIP.git` | Optional CLIP reranking utility functions in `src/blip_validate.py` | FULLY_FIXABLE | Install `openai-clip` via git URL | LOW | STATIC_AUDIT |
| `limn` | `open_clip` | Python package | `workbench/third_party/LIMN/LIMN/model.py:5` | PUBLIC_AVAILABLE | `open-clip-torch==2.20.0` | ViT-B/32 backbone extraction in `model.py` and `workbench/replay/limn.py:168` | FULLY_FIXABLE | Install `open-clip-torch==2.20.0` | LOW | STATIC_AUDIT |

---

### D2: Wrong Package / Version Conflict
Cases where the pinned or declared package name maps to an incompatible or incorrect package on PyPI.

| Model | Missing Item | Item Type | Exact Source Location | Public Availability | Compatible Version | Why Needed | Fixability | Proposed Fix | Difficulty | Verification Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `tgcir` | `clip` (`clip==0.2.0`) | Package name conflict | `workbench/third_party/TGCIR/model.py:5` | WRONG_PYPI_TARGET | `git+https://github.com/openai/CLIP.git` | Upstream README and `models.yaml` declare `clip==0.2.0`, which on PyPI is an unrelated "Clipboard library". Code imports `clip.load()` from OpenAI CLIP | FULLY_FIXABLE | Replace `clip==0.2.0` with `git+https://github.com/openai/CLIP.git` in environment specification | LOW | STATIC_AUDIT |

---

### D3: Local Import Typo
Source bugs where an import statement fails because of a typo in the local module name, while the intended file is shipped in the repository.

| Model | Missing Item | Item Type | Exact Source Location | Public Availability | Compatible Version | Why Needed | Fixability | Proposed Fix | Difficulty | Verification Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `encoder` | `datasets1` | Local import typo | `workbench/third_party/ENCODER/evaluate_model.py:7` | NOT_FOUND (ships `datasets.py`) | n/a (first-party code) | Dataset loader `FashionIQDataset`. Pinned evaluator crashes at import with `ModuleNotFoundError: No module named 'datasets1'`. Shipped file is `datasets.py` | FULLY_FIXABLE | Approved patch to change `import datasets1` to `import datasets as datasets1` | LOW | RUNTIME_VERIFIED |
| `csmcir` | `validate_blip` | Local import typo (unreachable) | `workbench/third_party/CSMCIR/src/blip_validate.py:23` | NOT_FOUND (ships `validate_blip_csmcir.py`) | n/a (first-party code) | Unused script `blip_validate.py` attempts `from validate_blip import compute_fiq_val_metrics`. Shipped file is `validate_blip_csmcir.py`. Not reachable by official entrypoint `validate_blip_csmcir.py` | FULLY_FIXABLE | Edit import in `src/blip_validate.py` or ignore as unreachable | LOW | STATIC_AUDIT |

---

### D4: Unpublished Source Module
Code references a proprietary or local module that was never published or committed to the upstream repository. Decisive test: tree does NOT ship a file of that name.

| Model | Missing Item | Item Type | Exact Source Location | Public Availability | Compatible Version | Why Needed | Fixability | Proposed Fix | Difficulty | Verification Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `pair` | `MHTransformer_rev` | Unpublished source module | `workbench/third_party/PAIR/models/model.py:13` | UNPUBLISHED_UPSTREAM | n/a (unreleased code) | `self.transformer_b = MHTransformer_rev.Transformer(...)` in `models/model.py:47`. Tree ships only `models/MHTransformer.py`. Model cannot instantiate | UPSTREAM_COOPERATION_REQUIRED | Request authors publish `MHTransformer_rev.py` or inspect whether `MHTransformer.py` can be adapted | HIGH | STATIC_AUDIT |
| `sprc` | `combiner` | Unpublished source module | `workbench/third_party/SPRC/src/validate_blip_rerank.py:17` | UNPUBLISHED_UPSTREAM | n/a (unreleased code) | `from combiner import Combiner`. Tree ships only leftover bytecode `src/__pycache__/combiner.cpython-39.pyc`; no `combiner.py` exists in source. Re-rank evaluation path crashes | UPSTREAM_COOPERATION_REQUIRED | Port `combiner.py` from `CLIP4Cir` checkout or decompile bytecode | MEDIUM | STATIC_AUDIT |
| `ptha_mtst` | Whole repository checkout | Pinned source tree | `workbench/registry/models.yaml:228-232` | UNPINNED_UPSTREAM | `zyinan99/MTST_PTHA` | Model registered with `source_available: false` and `source_dir: null`. Author repository exists on GitHub (`zyinan99/MTST_PTHA`) but was never synced or audited into workbench | UPSTREAM_COOPERATION_REQUIRED | Clone and pin `zyinan99/MTST_PTHA` commit into `workbench/third_party/PTHA_MTST` and audit | HIGH | STATIC_AUDIT |

---

### D5: Legacy Runtime Incompatibility
Code relies on deprecated, removed, or incompatible APIs from obsolete Python/PyTorch runtimes that cannot run on modern interpreters.

| Model | Missing Item | Item Type | Exact Source Location | Public Availability | Compatible Version | Why Needed | Fixability | Proposed Fix | Difficulty | Verification Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `clvc_net` | `torchvision.models.utils` | Removed PyTorch internal API | `workbench/third_party/CLVCNet/resnet.py:3` | REMOVED_UPSTREAM | `torchvision<=0.13.1` (Python 3.7 stack) | `from torchvision.models.utils import load_state_dict_from_url`. Module `torchvision.models.utils` was deleted in modern torchvision. Raises `ModuleNotFoundError` on modern torch | FULLY_FIXABLE | Approved patch to change import to `from torch.hub import load_state_dict_from_url` | LOW | RUNTIME_VERIFIED |

---

### D6: Optional / Training Imports Never Reached by Evaluation
Modules imported by offline preprocessing, training loops, or unused submodules within vendored libraries (such as LAVIS) that are never executed during evaluation.

| Model | Missing Item | Item Type | Exact Source Location | Public Availability | Compatible Version | Why Needed | Fixability | Proposed Fix | Difficulty | Verification Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `csmcir` | `apex` | GPU acceleration lib | `workbench/third_party/CSMCIR/src/lavis/models/blip2_models/modeling_t5.py:281` | PUBLIC_AVAILABLE | `apex` | FusedRMSNorm optimization in T5. Wrapped in `try...except ImportError` | NOT_NEEDED_FOR_EVAL | Keep guarded fallback to standard PyTorch LayerNorm | LOW | STATIC_AUDIT |
| `csmcir` | `horovod` | Distributed training | `workbench/third_party/CSMCIR/src/lavis/models/clip_models/loss.py:15` | PUBLIC_AVAILABLE | `horovod` | Multi-node distributed all-gather. Wrapped in `try...except ImportError` | NOT_NEEDED_FOR_EVAL | Unused in single-node evaluation | LOW | STATIC_AUDIT |
| `csmcir` | `tensorflow` | Deep learning framework | `workbench/third_party/CSMCIR/src/lavis/models/blip2_models/modeling_t5.py:85` | PUBLIC_AVAILABLE | `tensorflow>=2.0` | TF checkpoint conversion utility in modeling_t5. Unreached during PyTorch evaluation | NOT_NEEDED_FOR_EVAL | None required | LOW | STATIC_AUDIT |
| `csmcir` | `magic` | MIME type detector | `workbench/third_party/CSMCIR/src/lavis/datasets/download_scripts/.../download_data_cc3m.py:20` | PUBLIC_AVAILABLE | `python-magic` | CC3M dataset download helper script. Never imported during FashionIQ evaluation | NOT_NEEDED_FOR_EVAL | None required | LOW | STATIC_AUDIT |
| `csmcir` | `opendatasets` | Data downloader | `workbench/third_party/CSMCIR/src/lavis/datasets/download_scripts/download_flickr.py:19` | PUBLIC_AVAILABLE | `opendatasets` | Flickr downloader helper. Never imported during FashionIQ evaluation | NOT_NEEDED_FOR_EVAL | None required | LOW | STATIC_AUDIT |
| `csmcir` | `pycocoevalcap` | Metrics library | `workbench/third_party/CSMCIR/src/lavis/tasks/captioning.py:104` | PUBLIC_AVAILABLE | `pycocoevalcap==1.2` | COCO caption evaluation task. Never imported during FashionIQ retrieval | NOT_NEEDED_FOR_EVAL | None required | LOW | STATIC_AUDIT |
| `csmcir` | `pycocotools` | Dataset utility | `workbench/third_party/CSMCIR/src/lavis/tasks/captioning.py:105` | PUBLIC_AVAILABLE | `pycocotools==2.0.7` | COCO caption dataset evaluation. Never imported during FashionIQ retrieval | NOT_NEEDED_FOR_EVAL | None required | LOW | STATIC_AUDIT |
| `csmcir` | `scipy` | Scientific computing | `workbench/third_party/CSMCIR/src/lavis/common/gradcam.py:3` | PUBLIC_AVAILABLE | `scipy>=1.7.0` | GradCAM visualization utility. Never imported during FashionIQ retrieval | NOT_NEEDED_FOR_EVAL | None required | LOW | STATIC_AUDIT |
| `csmcir` | `skimage` | Image processing | `workbench/third_party/CSMCIR/src/lavis/common/gradcam.py:4` | PUBLIC_AVAILABLE | `scikit-image>=0.19.0` | GradCAM visualization utility. Never imported during FashionIQ retrieval | NOT_NEEDED_FOR_EVAL | None required | LOW | STATIC_AUDIT |
| `csmcir` | `matplotlib` | Plotting library | `workbench/third_party/CSMCIR/src/lavis/common/gradcam.py:2` | PUBLIC_AVAILABLE | `matplotlib>=3.5.0` | GradCAM visualization plotting. Never imported during FashionIQ retrieval | NOT_NEEDED_FOR_EVAL | None required | LOW | STATIC_AUDIT |
| `airknow` | `torchaudio`, `moviepy`, `nltk`, `scipy`, `skimage`, `matplotlib`, `parrots`, `addict`, `yapf`, `tifffile`, `turbojpeg`, `peft`, `openai`, `setproctitle`, `thop`, `tensorboardX` | Auxiliary libraries (22 roots) | Various files in `workbench/third_party/AirKnow/lavis/...` | PUBLIC_AVAILABLE | Various PyPI | Unused tasks (audio, video, dialogue, segmentation, online LLM, training hooks) in vendored LAVIS | NOT_NEEDED_FOR_EVAL | None required for evaluation path | LOW | STATIC_AUDIT |
| `conesep` | `apex`, `horovod`, `tensorflow`, `magic`, `opendatasets`, `pycocoevalcap`, `pycocotools`, `scipy`, `skimage`, `matplotlib`, `parrots`, `addict`, `yapf`, `tifffile`, `turbojpeg`, `peft`, `setproctitle`, `thop`, `tensorboardX`, `torchaudio`, `moviepy` | Auxiliary libraries (20 roots) | Various files in `workbench/third_party/ConeSep/lavis/...` | PUBLIC_AVAILABLE | Various PyPI | Unused tasks and training hooks in vendored LAVIS | NOT_NEEDED_FOR_EVAL | None required for evaluation path | LOW | STATIC_AUDIT |
| `habit` | `clip`, `apex`, `horovod`, `tensorflow`, `magic`, `opendatasets`, `pycocoevalcap`, `pycocotools`, `scipy`, `skimage`, `matplotlib`, `parrots`, `addict`, `yapf`, `tifffile`, `turbojpeg`, `peft`, `setproctitle`, `tensorboardX`, `torchaudio`, `moviepy` | Auxiliary libraries (21 roots) | Various files in `workbench/third_party/HABIT/lavis/...` | PUBLIC_AVAILABLE | Various PyPI | Unused tasks and training hooks in vendored LAVIS | NOT_NEEDED_FOR_EVAL | None required for evaluation path | LOW | STATIC_AUDIT |
| `intent` | `clip`, `apex`, `horovod`, `tensorflow`, `magic`, `opendatasets`, `pycocoevalcap`, `pycocotools`, `scipy`, `skimage`, `parrots`, `addict`, `yapf`, `tifffile`, `turbojpeg`, `peft`, `setproctitle`, `tensorboardX`, `torchaudio`, `moviepy` | Auxiliary libraries (20 roots) | Various files in `workbench/third_party/INTENT/lavis/...` | PUBLIC_AVAILABLE | Various PyPI | Unused tasks and training hooks in vendored LAVIS | NOT_NEEDED_FOR_EVAL | None required for evaluation path | LOW | STATIC_AUDIT |
| `hint` | `setproctitle`, `clip`, `apex`, `horovod`, `tensorflow`, `magic`, `opendatasets`, `pycocoevalcap`, `pycocotools`, `scipy`, `skimage`, `matplotlib`, `parrots`, `addict`, `yapf`, `tifffile`, `turbojpeg`, `peft`, `tensorboardX` | Auxiliary libraries (18 roots) | Various files in `workbench/third_party/HINT/lavis/...` and `train.py:17` | PUBLIC_AVAILABLE | Various PyPI | Unused tasks in vendored LAVIS and training entrypoints | NOT_NEEDED_FOR_EVAL | None required for evaluation path | LOW | STATIC_AUDIT |
| `encoder` | `model` | Internal training import | `workbench/third_party/ENCODER/train.py:17` | NOT_FOUND (ships `model_try2.py`) | n/a | Training script imports non-existent `model.py` (`model_try2.py` in repo). Evaluation uses `evaluate_model.py:8` which imports `from model_try2 import Encoder` | NOT_NEEDED_FOR_EVAL | Unreachable from `evaluate_model.py` | LOW | STATIC_AUDIT |
| `dcnet` | `ipdb` | Debugger | `workbench/third_party/DCNet/model/metric.py:64,100` | PUBLIC_AVAILABLE | `ipdb` | Triggered only inside `if torch.isnan(sims).sum().item():` exception handler | NOT_NEEDED_FOR_EVAL | Not executed unless NaNs occur in similarity matrix | LOW | STATIC_AUDIT |
| `dcnet` | `joblib`, `nltk`, `spacy` | Offline preprocessing | `workbench/third_party/DCNet/preprocess/resize_img.py:6`, `process_cap.py:1,7` | PUBLIC_AVAILABLE | Various PyPI | Used strictly during offline data preparation, never during evaluation entrypoint `test.py` | NOT_NEEDED_FOR_EVAL | Preprocessed artifacts loaded as `.pkl` by evaluator | LOW | STATIC_AUDIT |
| `combiner_rn50x4_noft` | `comet_ml` | Experiment tracker | `workbench/third_party/CLIP4Cir/src/clip_fine_tune.py:1` | PUBLIC_AVAILABLE | `comet-ml==3.21.0` | Training experiment tracking. Never reached by evaluation entrypoint `src/validate.py` | NOT_NEEDED_FOR_EVAL | None required for evaluation | LOW | STATIC_AUDIT |
| `clip4cir_rn50x4_fullft` | `comet_ml` | Experiment tracker | `workbench/third_party/CLIP4Cir/src/clip_fine_tune.py:1` | PUBLIC_AVAILABLE | `comet-ml==3.21.0` | Training experiment tracking. Never reached by evaluation entrypoint `src/validate.py` | NOT_NEEDED_FOR_EVAL | None required for evaluation | LOW | STATIC_AUDIT |
| `sprc` | `apex`, `horovod`, `tensorflow`, `magic`, `opendatasets`, `pycocoevalcap`, `pycocotools`, `scipy`, `skimage`, `matplotlib`, `spacy` | Auxiliary libraries (11 roots) | Various files in `workbench/third_party/SPRC/src/lavis/...` | PUBLIC_AVAILABLE | Various PyPI | Unused tasks and optional utilities in vendored LAVIS | NOT_NEEDED_FOR_EVAL | None required for evaluation path | LOW | STATIC_AUDIT |

---

### D7: Broken Upstream Package Init
Upstream package `__init__.py` files that crash unconditionally during package import because of unguarded imports of modules absent from the repository.

| Model | Missing Item | Item Type | Exact Source Location | Public Availability | Compatible Version | Why Needed | Fixability | Proposed Fix | Difficulty | Verification Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `csmcir` | 98 absent LAVIS modules | Broken package init | `workbench/third_party/CSMCIR/src/lavis/models/__init__.py:44-141` | UNPUBLISHED_UPSTREAM | n/a (author omitted files) | `lavis.models.__init__.py` imports 98 non-existent modules before reaching line 142 where `Blip2QformerCirAlignPromptCSMCIR` is registered. Importing `lavis.models` crashes immediately | FULLY_FIXABLE | Apply declared patch `workbench/patches/csmcir_models_init_compat.patch` to wrap absent imports in `try...except ImportError` | LOW | RUNTIME_VERIFIED |
| `airknow` | `blip_cir`, `blip_cir_caption` | Broken package init | `workbench/third_party/AirKnow/lavis/models/__init__.py:33-34` | UNPUBLISHED_UPSTREAM | n/a (ships only bytecode/checkpoints) | `lavis.models.__init__.py` imports `from lavis.models.blip_models.blip_cir import BlipCir`. Source `.py` files are absent (only `.pyc` in checkpoints). Package init crashes | FULLY_FIXABLE | Wrap imports in `try...except ImportError` or restore source files | LOW | STATIC_AUDIT |
| `sprc` | `blip2_cir_rerank_learn` model registration | Missing model registration in package registry | `workbench/third_party/SPRC/src/blip_validate.py:144` | UNPUBLISHED_UPSTREAM | n/a | Evaluator CLI defaults to `--blip-model-name blip2_cir_rerank_learn`. LAVIS model registry does not contain this name (only `blip2_cir_rerank`, `blip2_cir_cls`, `blip2_cir_rel`). `load_model_and_preprocess` raises KeyError | FULLY_FIXABLE | Specify `--blip-model-name blip2_cir_rerank` or register missing model class | MEDIUM | STATIC_AUDIT |

---

### D8: GPU-Specific / CUDA Acceleration Hardcodes
Evaluators that contain hard-coded `.cuda()` calls or CUDA-only functions that crash unconditionally on CPU runtimes.

| Model | Missing Item | Item Type | Exact Source Location | Public Availability | Compatible Version | Why Needed | Fixability | Proposed Fix | Difficulty | Verification Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `limn` | NVIDIA CUDA Runtime & GPU | Hardware acceleration | `workbench/third_party/LIMN/LIMN/test.py:30,48`, `workbench/replay/limn.py:125` | HARDWARE_DEPENDENCY | CUDA >= 11.7, Compute Capability >= 7.0 | `torch.autograd.Variable(imgs).cuda()`, `all_queries.cuda()` hardcoded. Replay preflight checks `torch.cuda.is_available()` | HARDWARE_REQUIRED | Run on host equipped with NVIDIA GPU and CUDA-enabled PyTorch build | LOW | RUNTIME_VERIFIED |
| `csmcir` | NVIDIA CUDA Runtime & GPU | Hardware acceleration | `workbench/third_party/CSMCIR/src/utils_csmcir.py:34-37`, `src/validate_blip_csmcir.py:128` | HARDWARE_DEPENDENCY | CUDA >= 11.7 | `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` and `torch.cuda.empty_cache()` executed unconditionally | HARDWARE_REQUIRED | Run on CUDA GPU | LOW | STATIC_AUDIT |
| `airknow` | NVIDIA CUDA Runtime & GPU | Hardware acceleration | `workbench/third_party/AirKnow/test_BLIP2.py:30` | HARDWARE_DEPENDENCY | CUDA >= 11.7 | `imgs.cuda()`, `torch.cuda.amp.autocast()` called unconditionally | HARDWARE_REQUIRED | Run on CUDA GPU | LOW | STATIC_AUDIT |
| `conesep` | NVIDIA CUDA Runtime & GPU | Hardware acceleration | `workbench/third_party/ConeSep/test.py:30` | HARDWARE_DEPENDENCY | CUDA >= 11.7 | `imgs.cuda()`, `torch.cuda.amp.autocast()` called unconditionally | HARDWARE_REQUIRED | Run on CUDA GPU | LOW | STATIC_AUDIT |
| `habit` | NVIDIA CUDA Runtime & GPU | Hardware acceleration | `workbench/third_party/HABIT/test.py:30` | HARDWARE_DEPENDENCY | CUDA >= 11.7 | `imgs = torch.autograd.Variable(imgs).cuda()` called unconditionally | HARDWARE_REQUIRED | Run on CUDA GPU | LOW | STATIC_AUDIT |
| `intent` | NVIDIA CUDA Runtime & GPU | Hardware acceleration | `workbench/third_party/INTENT/test.py:30` | HARDWARE_DEPENDENCY | CUDA >= 11.7 | `imgs = torch.autograd.Variable(imgs).cuda()` called unconditionally | HARDWARE_REQUIRED | Run on CUDA GPU | LOW | STATIC_AUDIT |
| `hint` | NVIDIA CUDA Runtime & GPU | Hardware acceleration | `workbench/third_party/HINT/test.py:30` | HARDWARE_DEPENDENCY | CUDA >= 11.7 | `imgs = torch.autograd.Variable(imgs).cuda()` called unconditionally | HARDWARE_REQUIRED | Run on CUDA GPU | LOW | STATIC_AUDIT |
| `encoder` | NVIDIA CUDA Runtime & GPU | Hardware acceleration | `workbench/third_party/ENCODER/evaluate_model.py:48` | HARDWARE_DEPENDENCY | CUDA >= 11.7 | `Encoder_model.cuda()` called unconditionally | HARDWARE_REQUIRED | Run on CUDA GPU | LOW | STATIC_AUDIT |
| `pair` | NVIDIA CUDA Runtime & GPU | Hardware acceleration | `workbench/third_party/PAIR/test.py:30` | HARDWARE_DEPENDENCY | CUDA >= 11.7 | `imgs.cuda()` called unconditionally | HARDWARE_REQUIRED | Run on CUDA GPU | LOW | STATIC_AUDIT |
| `clvc_net` | NVIDIA CUDA Runtime & GPU | Hardware acceleration | `workbench/third_party/CLVCNet/test.py:30` | HARDWARE_DEPENDENCY | CUDA >= 10.2 / 11.x | `torch.cuda.FloatTensor`, `.cuda()` called unconditionally | HARDWARE_REQUIRED | Run on CUDA GPU | LOW | STATIC_AUDIT |
| `dcnet` | NVIDIA CUDA Runtime & GPU | Hardware acceleration | `workbench/third_party/DCNet/test.py:45` | HARDWARE_DEPENDENCY | CUDA >= 10.2 / 11.x | `model.to(device)` with device forced to cuda | HARDWARE_REQUIRED | Run on CUDA GPU | LOW | STATIC_AUDIT |
| `tgcir` | NVIDIA CUDA Runtime & GPU | Hardware acceleration | `workbench/third_party/TGCIR/test.py:30` | HARDWARE_DEPENDENCY | CUDA >= 11.7 | `imgs.cuda()` called unconditionally | HARDWARE_REQUIRED | Run on CUDA GPU | LOW | STATIC_AUDIT |
| `sprc` | NVIDIA CUDA Runtime & GPU | Hardware acceleration | `workbench/third_party/SPRC/src/blip_validate.py:45` | HARDWARE_DEPENDENCY | CUDA >= 11.7 | `blip_model.to(device)` with device forced to cuda | HARDWARE_REQUIRED | Run on CUDA GPU | LOW | STATIC_AUDIT |

---

## 2. Summary by Model

| Model ID | Pinned Tree | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | Engineering Effort vs Upstream Cooperation Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| `csmcir` | `CSMCIR` | 1 | 0 | 1 | 0 | 0 | 10 | 1 | 1 | Engineering effort: local approved patch repairs D7 package init; published Qwen CoT captions alias unpublished COT_ours2; runs with GPU. |
| `airknow` | `AirKnow` | 2 | 0 | 0 | 0 | 0 | 22 | 1 | 1 | Upstream cooperation required: unpublished correction dicts, no clean 0% checkpoint, test script evaluates in train mode with no CLI. |
| `conesep` | `ConeSep` | 2 | 0 | 0 | 0 | 0 | 20 | 0 | 1 | Upstream cooperation required: unpublished correction dicts, no clean 0% checkpoint, in-memory function only without CLI. |
| `ptha_mtst` | *(None)* | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | Upstream cooperation required: tree not synced/pinned, no fine-tuned FashionIQ checkpoint or MTST corpus released. |
| `habit` | `HABIT` | 2 | 0 | 0 | 0 | 0 | 21 | 0 | 1 | Upstream cooperation required: unpublished correction dicts, no clean 0% checkpoint, in-memory function only without CLI. |
| `intent` | `INTENT` | 2 | 0 | 0 | 0 | 0 | 20 | 0 | 1 | Upstream cooperation required: unpublished correction dicts, no clean 0% checkpoint, in-memory function only without CLI. |
| `hint` | `HINT` | 3 | 0 | 0 | 0 | 0 | 18 | 0 | 1 | Upstream cooperation required: unpublished correction dicts, no FashionIQ checkpoint loader in repo, in-memory function only. |
| `encoder` | `ENCODER` | 1 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | Engineering effort: local patch repairs `datasets1` import typo; needs unversioned local weight file `./open_clip_pytorch_model.bin`. |
| `pair` | `PAIR` | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | Upstream cooperation required: missing `MHTransformer_rev.py` prevents model construction; test function lacks CLI. |
| `clvc_net` | `CLVCNet` | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | Engineering effort / Legacy: trivial patch repairs `torchvision.models.utils`; upstream weights locked in unmapped archive. |
| `dcnet` | `DCNet` | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 1 | Upstream cooperation required: unseeded NumPy normal RNG in OOV generation prevents reproducible deterministic replication. |
| `combiner_rn50x4_noft` | `CLIP4Cir` | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | Engineering effort: install `openai-clip`; official entrypoint `src/validate.py` has complete CLI and matches paper arguments. |
| `clip4cir_rn50x4_fullft` | `CLIP4Cir` | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | Engineering effort: install `openai-clip`; official entrypoint `src/validate.py` matches paper arguments; needs checkpoint bundle. |
| `tgcir` | `TGCIR` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | Engineering effort: fix PyPI `clip` specification to `openai-clip`; test function lacks CLI; model is unauthenticated whole-model pickle. |
| `sprc` | `SPRC` | 1 | 0 | 0 | 1 | 0 | 11 | 1 | 1 | Upstream cooperation required: missing `combiner.py` and unregistered `blip2_cir_rerank_learn` model name break evaluation CLI. |
| `limn` | `LIMN` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | Engineering effort: official replay harness (`workbench/replay/limn.py`) implements complete CLI, validation, and per-query export on GPU. |

---

## 3. Non-Python Asset Inventory

Non-Python assets required by the models for preprocessing, inference, tokenization, or backbone feature extraction.

### Classification Categories
- **`PUBLIC_AVAILABLE`**: Hosted on public official repositories (HuggingFace Hub, PyTorch/OpenAI/Salesforce CDNs) with stable URLs.
- **`GENERATABLE_DETERMINISTICALLY`**: Can be generated offline by deterministic preprocessing scripts using fixed RNG seeds.
- **`GENERATABLE_BUT_NOT_EQUIVALENT`**: Generated by offline scripts, but unseeded random number generators or non-deterministic heuristics produce stochastic outputs that change evaluation semantics.
- **`UNPUBLISHED_REQUIRED`**: Mandatory asset required for evaluation that was never released or hosted publicly by the authors.
- **`OPTIONAL`**: Asset used only for supplementary visualization or non-evaluation tasks.
- **`UNKNOWN`**: Provenance or generation process cannot be established from upstream sources.

---

### Non-Python Asset Ledger

| Asset Name | Model(s) | What Creates It | Public? | Downloadable? | Generated by Preprocessing? | Deterministic? | Substitute Changes Semantics? | Classification | Exact Evidence & Details |
|---|---|---|---|---|---|---|---|---|---|
| `correction_dict_{cat}.json` (LIMN) | `limn` | Shipped in repository under `correct_files/` | YES | YES (in git) | NO (shipped) | YES | NO (canonical) | `PUBLIC_AVAILABLE` | `workbench/third_party/LIMN/correct_files/correction_dict_{dress,shirt,shoes,toptee}.json` tracked in git checkout (`7d7bc9b`). |
| `correction_dict_{cat}.json` (TGCIR) | `tgcir` | Shipped in repository under `correction_dict/` | YES | YES (in git) | NO (shipped) | YES | NO (canonical) | `PUBLIC_AVAILABLE` | `workbench/third_party/TGCIR/correction_dict/correction_dict_{dress,shirt,shoes,toptee}.json` tracked in git checkout (`65fa78e`). |
| `correction_dict_{cat}.json` (iLearn Lab) | `airknow`, `conesep`, `habit`, `intent`, `hint` | Author preprocessing script (uncommitted) | NO | NO | YES | UNKNOWN | YES | `UNPUBLISHED_REQUIRED` | Dataset loader `FashionIQDataset.__init__` in `datasets.py:137` unconditionally loads `captions/correction_dict_{category}.json`. None of the 5 trees commit these files. |
| `cap.{cat}.glove.val.pkl` & `glove_vecs.pkl` | `dcnet` | `preprocess/process_cap.py` | NO | NO | YES | NO | YES | `GENERATABLE_BUT_NOT_EQUIVALENT` | `preprocess/process_cap.py:29-31` assigns random 300-d vectors `np.random.normal(0, 0.3, (300,))` to OOV words without initializing RNG seed. Regenerating produces non-identical embeddings. |
| spaCy `en_vectors_web_lg` / GloVe | `dcnet` | spaCy model download | YES | YES | NO | YES | NO | `PUBLIC_AVAILABLE` | Downloadable via `python -m spacy download en_vectors_web_lg` or direct spaCy release wheel. |
| `qwen_captions/{cat}_cot_val.json` | `csmcir` | Author Qwen caption generation pipeline | YES | YES | YES | YES | NO | `PUBLIC_AVAILABLE` | Hosted on author Hugging Face Hub `peng12138/CSMCIR` revision `cf0c19bb346266c295b4b5772ebf03bd1c4f0467`. Verified SHA-256 in `auxiliary_assets.yaml`. |
| `COT_ours2/fashioniq/{cat}_cot_val.json` | `csmcir` | Author internal CoT captioning pipeline | NO | NO | YES | UNKNOWN | YES (slight deviation) | `UNPUBLISHED_REQUIRED` | Hardcoded in `src/data_utils_csmcir.py:141`. Author never published `COT_ours2`. Workbench allows approved aliasing from `qwen_captions` under explicit variant flag. |
| `bert-base-uncased` tokenizer / config | `csmcir`, `sprc`, `airknow`, `conesep`, `habit`, `intent` | Hugging Face Hub release | YES | YES | NO | YES | NO | `PUBLIC_AVAILABLE` | Standard Hugging Face model repository. Note: `AirKnow`, `ConeSep`, `HABIT`, and `INTENT` hardcode `/root/bert-base-uncased` path in `datasets.py:136`. |
| `open_clip_pytorch_model.bin` | `encoder` | Pretrained OpenCLIP ViT-B-32 weights | YES | YES | NO | YES | NO | `PUBLIC_AVAILABLE` | Required at local path `./open_clip_pytorch_model.bin` or `/root/CLIP-ViT-B-32-laion2B-s34B-b79K` by `model_try2.py:134-135`. Corresponds to LAION-2B ViT-B-32 weights. |
| OpenAI CLIP `RN50x4.pt` | `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft` | OpenAI official CLIP model release | YES | YES | NO | YES | NO | `PUBLIC_AVAILABLE` | Downloadable from `https://openaipublic.azureedge.net/clip/models/.../RN50x4.pt`. Verified SHA-256 `7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd`. |
| OpenAI CLIP `ViT-B-16.pt` | `tgcir` | OpenAI official CLIP model release | YES | YES | NO | YES | NO | `PUBLIC_AVAILABLE` | Downloadable from OpenAI public Azure CDN via `clip.load('ViT-B/16')`. |
| `resnet50-19c8e357.pth` | `clvc_net`, `dcnet` | torchvision ImageNet pretrained weights | YES | YES | NO | YES | NO | `PUBLIC_AVAILABLE` | Hosted at `https://download.pytorch.org/models/resnet50-19c8e357.pth`. Expected prefix `19c8e357`. |
| `blip2_pretrained.pth` | `csmcir`, `sprc`, `airknow`, `conesep`, `habit`, `intent`, `hint` | Salesforce LAVIS research release | YES | YES | NO | YES | NO | `PUBLIC_AVAILABLE` | Hosted at `https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth` (747 MB). |
| `eva_vit_g.pth` | `csmcir`, `sprc`, `airknow`, `conesep`, `habit`, `intent`, `hint` | Salesforce LAVIS research release | YES | YES | NO | YES | NO | `PUBLIC_AVAILABLE` | Hosted at `https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth` (2.02 GB). |
| Resized FashionIQ images (224x224 / 256x256) | `airknow`, `conesep`, `habit`, `intent`, `hint` | Author PIL/OpenCV resizing scripts | NO | NO | YES | NO | YES | `GENERATABLE_BUT_NOT_EQUIVALENT` | Evaluators read pre-resized images from `resized_image/{category}/`. Resizing algorithms (bicubic vs bilinear, PIL vs OpenCV, antialiasing filters) differ across authors and shift feature activations. |
| MTST Modification Text Corpus | `ptha_mtst` | Multi-turn modification text generation pipeline | NO | NO | YES | UNKNOWN | YES | `UNPUBLISHED_REQUIRED` | Scale-up training and evaluation modification texts generated via proprietary MLLM pipeline; no FashionIQ split corpus verified. |

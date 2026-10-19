# R10 LAVIS-Family Dependency Profiles Audit

## 1. Runtime Entry Point Imports (Question 1)

The adapter mapping in `workbench/backend/adapters/models.py:ADAPTERS` dispatches the six models:

### 1.1 Adapter Entrypoint Architecture
- **HINT**: `HintAdapter` (`workbench/backend/adapters/models.py:80-97`) executes `workbench/replay/hint.py` directly with arguments `--checkpoint-root`, `--dataset-root`, `--source-root`.
- **Air-Know**: `AirKnowAdapter` (`workbench/backend/adapters/models.py:112-128`) uses `_replay_command` (`workbench/backend/adapters/models.py:31-63`) to execute `workbench/replay/airknow.py` with `--variant`.
- **ConeSep**: `ConeSepAdapter` (`workbench/backend/adapters/models.py:130-140`) uses `_replay_command` to execute `workbench/replay/conesep.py` with `--variant`.
- **HABIT**: `HabitAdapter` (`workbench/backend/adapters/models.py:142-151`) uses `_replay_command` to execute `workbench/replay/habit.py` with `--variant`.
- **INTENT**: `IntentAdapter` (`workbench/backend/adapters/models.py:153-162`) uses `_replay_command` to execute `workbench/replay/intent.py` with `--variant`.
- **SPRC**: `SPRCAdapter` (`workbench/backend/adapters/models.py:219-247`) defines `script = "src/blip_validate.py"` and returns:
  `[_runtime.python_executable(self.model), str(source / self.script), "--dataset", "fashionIQ", "--blip-model-name", "blip2_cir_align_prompt", "--backbone", backbone, "--model-path", str(checkpoint)]`.
  **CRITICAL FINDING**: `workbench/replay/sprc.py` DOES NOT EXIST on disk (and has never existed in git history). `SPRCAdapter` launches the vendored script `workbench/third_party/SPRC/src/blip_validate.py` directly without a workbench wrapper.

---

### 1.2 Five LAVIS Wrappers Runtime Call Graph
Each of the five wrappers (`workbench/replay/{airknow,conesep,habit,intent,hint}.py`) executes:
1. **Replay Wrapper Scope**:
   - `workbench/replay/<model>.py:12-42`: imports `json`, `sys`, `argparse.ArgumentParser`, `dataclasses.asdict`, `datetime.datetime`, `datetime.timezone`, `pathlib.Path`, `types.SimpleNamespace`, `typing.Any`.
   - Local workbench modules: `from workbench.backend.fashioniq_layout import CATEGORIES`, `from workbench.backend.registry import model_by_id`, `from workbench.replay.common import ReplayResult, isolated_cache_root, load_trusted_whole_model, macro_aggregate, sha256_file, source_pin_status, constructor_input_blockers`.
2. **Replay Module Loader (`_load_replay_modules`)**:
   - `import torch`
   - Inserts `source_root` and `source_root / "lavis"" into `sys.path`.
   - `datasets = __import__("datasets")` (`<source_root>/datasets.py`)
   - `data_utils = __import__("data_utils")` (`<source_root>/data_utils.py`)
   - `test_module = __import__("test")` (or `test_BLIP2` in Air-Know)
   - `from workbench.replay.lavis_processors import text_processors`
3. **LAVIS Text Processor Construction (`workbench/replay/lavis_processors.py`)**:
   - `from omegaconf import OmegaConf` (`lavis_processors.py:107`)
   - Installs stub `sys.modules["lavis"]`, imports `lavis.common.registry.registry` (`<source_root>/lavis/common/registry.py`).
   - Installs stub `sys.modules["lavis.processors"]`, loads `lavis.processors.base_processor.BaseProcessor` and `lavis.processors.blip_processors.BlipCaptionProcessor` (`<source_root>/lavis/processors/blip_processors.py`).
   - Compiles AST for `load_preprocess` from `<source_root>/lavis/models/__init__.py`.
   - `BlipCaptionProcessor.from_config` calls `RandomAugment` and regex (`re`).
4. **Vendored Dataset and Evaluation Execution**:
   - `datasets.FashionIQ` (`datasets.py:24` in HINT, `:28` in AirKnow, `:25` in ConeSep/INTENT, `:27` in HABIT):
     - `import os, numpy, PIL, torch, json, torch.utils.data, string, glob, torchvision, random, pickle, logging`
     - `from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation` (`datasets.py:13`)
     - `from tqdm import trange` (`datasets.py:14`)
   - `data_utils.targetpad_transform` (`data_utils.py:92`):
     - `import json, pathlib.Path, typing_extensions.Literal, typing.Union, typing.List, typing.Dict`
     - `import PIL, PIL.Image, torchvision.transforms.functional as F, torch.utils.data.Dataset`
     - `from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, torch`
   - `test_module.test` (`test.py:9` in ConeSep, HABIT, INTENT; `test.py:11` in HINT; `test_BLIP2.py:9` in AirKnow):
     - `import numpy as np, torch, tqdm.tqdm as tqdm, torch.nn.functional as F, pickle`
5. **Whole Model Deserialization via `load_trusted_whole_model` (`torch.load`)**:
   - Checkpoints are saved via `torch.save(model, ...)` (whole serialized PyTorch Module instances).
   - PyTorch unpickler reconstructs model class:
     - HINT: `lavis.models.blip2_models.HINT.HINT`
     - AirKnow: `lavis.models.blip2_models.blip2_cir.Blip2CIR`
     - ConeSep: `lavis.models.blip2_models.ConeSep.ConeSep`
     - HABIT: `lavis.models.blip2_models.HABIT.HABIT`
     - INTENT: `lavis.models.blip2_models.blip2_cir.Blip2CIR`
   - Model module imports triggered at unpickle time:
     - `from lavis.common.registry import registry`
     - `from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures`
     - `from lavis.models.blip2_models.blip2 import Blip2Base, compute_sim_matrix, disabled_train`
     - In AirKnow, ConeSep, HABIT, INTENT: `from sklearn.cluster import KMeans` (`blip2_cir.py:19`, `ConeSep.py:24`, `HABIT.py:24`, `blip2_cir.py:30`)
     - In HABIT: `from sklearn.metrics import mutual_info_score`, `from sklearn.preprocessing import KBinsDiscretizer`, `from sklearn.cluster import DBSCAN` (`HABIT.py:25-27`)
   - Base model `blip2.py` imports:
     - `from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel` (`blip2.py:23`)
     - `from lavis.models.eva_vit import create_eva_vit_g` (`blip2.py:24`)
     - `from lavis.models.clip_vit import create_clip_vit_L` (`blip2.py:25`)
     - `from transformers import BertTokenizer` (`blip2.py:26`)
     - `import lavis.common.dist_utils as dist_utils` (`blip2.py:18`)
     - `from lavis.common.utils import is_url` (`blip2.py:20`)
     - `from lavis.common.logger import MetricLogger` (`blip2.py:21`)
     - `from lavis.models.base_model import BaseModel` (`blip2.py:22`)
   - Vision & Q-Former imports:
     - `Qformer.py:24-46`: `from transformers.activations import ACT2FN`, `from transformers.file_utils import ModelOutput`, `from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, ...`, `from transformers.modeling_utils import PreTrainedModel, apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer`, `from transformers.utils import logging`, `from transformers.models.bert.configuration_bert import BertConfig`.
     - `eva_vit.py:15-16`: `from timm.models.layers import drop_path, to_2tuple, trunc_normal_`, `from timm.models.registry import register_model`.
     - `clip_vit.py:10`: `from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper`.
     - `dist_utils.py:14`: `import timm.models.hub as timm_hub`.
     - `utils.py:25-26`: `from iopath.common.download import download`, `from iopath.common.file_io import file_lock, g_pathmgr`, `import pandas as pd`, `import yaml`.

---

### 1.3 SPRC Runtime Call Graph
Entrypoint: `workbench/third_party/SPRC/src/blip_validate.py`.
1. **Direct Top-level Imports in `src/blip_validate.py`**:
   - Stdlib: `json` (`:2`), `argparse.ArgumentParser` (`:3`), `datetime.datetime` (`:4`), `pathlib.Path` (`:5`), `statistics.mean, geometric_mean, harmonic_mean` (`:6`), `typing.List` (`:7`).
   - External: `import clip` (`:8`), `import numpy as np` (`:9`), `import pandas as pd` (`:10`), `import torch` (`:11`), `import torch.nn.functional as F` (`:12`), `from torch import optim, nn` (`:13`), `from torch.utils.data import DataLoader` (`:14`), `from tqdm import tqdm` (`:15`), `from torch.optim.lr_scheduler import OneCycleLR` (`:17`).
   - Vendored / Local:
     - `from lavis.models import load_model_and_preprocess` (`:16`)
     - `from data_utils import base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset` (`:20`)
     - `from utils import collate_fn, update_train_running_results, set_train_bar_description, extract_index_blip_features, save_model, generate_randomized_fiq_caption, element_wise_sum, device` (`:21`)
     - `from validate_blip import compute_cirr_val_metrics, compute_fiq_val_metrics` (`:23`)
2. **Cascading Imports triggered by `from lavis.models import load_model_and_preprocess`**:
   Unlike the five workbench wrappers, SPRC does not stub `lavis`. `from lavis.models import ...` executes `SPRC/src/lavis/__init__.py`:
   - `lavis/__init__.py:15`: `from lavis.datasets.builders import *` -> loads `base_dataset_builder.py` -> imports `lavis.datasets.data_utils` -> imports `decord` and `webdataset as wds`.
   - `lavis/__init__.py:16`: `from lavis.models import *` -> loads `SPRC/src/lavis/models/__init__.py`, which imports:
     - `AlbefClassification` (`:15`) -> imports `lavis.models.med` -> imports `from transformers.modeling_utils import apply_chunking_to_forward, ...` (`med.py:40`).
     - `Blip2QformerCirCat` (`:42`) -> imports `from skimage import transform as skimage_transform` (`blip2_qformer_cir_cat.py:23`).
     - `Img2PromptVQA` (`:59`) -> imports `import spacy` (`img2prompt_vqa.py:15`).
     - `CLIP` (`:62`) -> imports `from lavis.tasks.multimodal_classification import ...` -> `lavis.tasks.captioning` -> `from pycocoevalcap.eval import COCOEvalCap` (`captioning.py:104`) and `from pycocotools.coco import COCO` (`captioning.py:105`).
3. **Execution of `clip_finetune_fiq` (`blip_validate.py:26`)**:
   - Calls `load_model_and_preprocess(name="blip2_cir_align_prompt", model_type=backbone, is_eval=False, device=device)`
   - Calls `torch.load(checkpoint_path)` (loads state dict dictionary, not whole model).
   - Constructs `FashionIQDataset` (`data_utils.py:108`), reading PIL images, applying `targetpad_transform`.
   - Calls `compute_fiq_val_metrics` (`validate_blip.py:24`).

---

## 2. Dependency Taxonomy Table (Question 2)

| Module / Import Name | Category | Providing Package | Declared in models.yaml? | Status & Notes |
|:---|:---|:---|:---|:---|
| `__future__` | (a) stdlib | Python stdlib | N/A | Language future imports |
| `argparse` | (a) stdlib | Python stdlib | N/A | CLI arguments parsing |
| `ast` | (a) stdlib | Python stdlib | N/A | Used in `lavis_processors.py` |
| `base64` | (a) stdlib | Python stdlib | N/A | Used in AirKnow `datasets.py:16` |
| `collections` | (a) stdlib | Python stdlib | N/A | Deque, OrderedDict, defaultdict |
| `contextlib` | (a) stdlib | Python stdlib | N/A | Context managers |
| `dataclasses` | (a) stdlib | Python stdlib | N/A | Data structures in wrappers/lavis |
| `datetime` | (a) stdlib | Python stdlib | N/A | Timestamps |
| `functools` | (a) stdlib | Python stdlib | N/A | Partial functions |
| `glob` | (a) stdlib | Python stdlib | N/A | File pattern matching |
| `importlib` | (a) stdlib | Python stdlib | N/A | Dynamic module loading |
| `io` | (a) stdlib | Python stdlib | N/A | Streams |
| `itertools` | (a) stdlib | Python stdlib | N/A | Iteration utilities |
| `json` | (a) stdlib | Python stdlib | N/A | Serialization |
| `logging` | (a) stdlib | Python stdlib | N/A | Logging |
| `math` | (a) stdlib | Python stdlib | N/A | Math operations |
| `multiprocessing`| (a) stdlib | Python stdlib | N/A | Worker process handling |
| `operator` | (a) stdlib | Python stdlib | N/A | Itemgetter |
| `os` | (a) stdlib | Python stdlib | N/A | OS filesystem calls |
| `pathlib` | (a) stdlib | Python stdlib | N/A | Filesystem path objects |
| `pickle` | (a) stdlib | Python stdlib | N/A | Checkpoint / cache deserialization |
| `random` | (a) stdlib | Python stdlib | N/A | Random numbers |
| `re` | (a) stdlib | Python stdlib | N/A | Regular expressions |
| `shutil` | (a) stdlib | Python stdlib | N/A | File copying |
| `statistics` | (a) stdlib | Python stdlib | N/A | Metric means in SPRC |
| `string` | (a) stdlib | Python stdlib | N/A | String constants in datasets.py |
| `sys` | (a) stdlib | Python stdlib | N/A | System path and interpreter state |
| `tarfile` | (a) stdlib | Python stdlib | N/A | Archive extraction in utils.py |
| `time` | (a) stdlib | Python stdlib | N/A | Sleep, timing |
| `types` | (a) stdlib | Python stdlib | N/A | ModuleType stubbing |
| `typing` | (a) stdlib | Python stdlib | N/A | Type annotations |
| `urllib` | (a) stdlib | Python stdlib | N/A | URL parsing/downloading |
| `warnings` | (a) stdlib | Python stdlib | N/A | Warning controls |
| `torch` | (b) declared | `torch` | YES (`pytorch: 2.1.0` / `2.0.1`) | Core tensor runtime |
| `torchvision` | (b) declared | `torchvision` | YES (`torchvision: 0.16.0` / `0.15.2`) | Image transforms |
| `transformers` | (b) declared | `transformers` | YES (`4.25.0` / `4.36.2`) | Tokenizer & Q-Former layers |
| `timm` | (b) declared | `timm` | YES (`0.9.16` / `0.9.12`) | Backbone & hub downloads |
| `sklearn` | (b) declared | `scikit-learn` | YES (`1.3.2`, 5 models) | Unused at eval; needed for unpickle |
| `spacy` | (b) declared | `spacy` | YES (`3.7.2` in sprc only) | Unused at eval; needed for SPRC imports |
| `open_clip` | (b) declared | `open-clip-torch`| YES (`2.24.0` in habit/intent/hint) | **SPURIOUS: NEVER IMPORTED** |
| `salesforce-lavis`|(b) declared | `salesforce-lavis`| YES (`1.0.2` in all 6) | **PHANTOM: COMPLETELY SHADOWED** |
| `clip` | (c) undeclared | `clip` (OpenAI) | **NO** | **BLOCKER**: Imported in SPRC `blip_validate.py:8` |
| `numpy` | (c) undeclared | `numpy` | **NO** | Pinned transitively by torch/timm |
| `PIL` | (c) undeclared | `pillow` | **NO** | Pinned transitively by torchvision |
| `tqdm` | (c) undeclared | `tqdm` | **NO** | Progress bars in eval/datasets |
| `omegaconf` | (c) undeclared | `omegaconf` | **NO** | Loaded in `lavis_processors.py` & lavis |
| `iopath` | (c) undeclared | `iopath` | **NO** | Imported in `lavis.common.utils` |
| `fairscale` | (c) undeclared | `fairscale` | **NO** | Imported in `lavis.models.clip_vit` |
| `pandas` | (c) undeclared | `pandas` | **NO** | Imported in `lavis.common.utils` & SPRC |
| `yaml` | (c) undeclared | `pyyaml` | **NO** | Imported in `lavis.common.utils` |
| `requests` | (c) undeclared | `requests` | **NO** | Imported in `lavis.common.utils` |
| `decord` | (c) undeclared | `decord` | **NO** | Imported in SPRC `lavis/__init__.py` |
| `webdataset` | (c) undeclared | `webdataset` | **NO** | Imported in SPRC `lavis/__init__.py` |
| `skimage` | (c) undeclared | `scikit-image` | **NO** | Imported in SPRC `lavis/__init__.py` |
| `pycocoevalcap` | (c) undeclared | `pycocoevalcap` | **NO** | Imported in SPRC `lavis/__init__.py` |
| `pycocotools` | (c) undeclared | `pycocotools` | **NO** | Imported in SPRC `lavis/__init__.py` |
| `cv2` | (c) undeclared | `opencv-python` | **NO** | Imported in SPRC `validate_blip.py` |
| `typing_extensions`|(c) undeclared | `typing-extensions`| **NO** | Imported in `data_utils.py` |
| `workbench.*` | (d) local repo | local repo tree | N/A | Replay harnesses & layout definitions |
| `datasets` | (d) local repo | vendored checkout | N/A | `<checkout>/datasets.py` |
| `data_utils` | (d) local repo | vendored checkout | N/A | `<checkout>/data_utils.py` or `src/` |
| `test`/`test_BLIP2`|(d) local repo | vendored checkout | N/A | `<checkout>/test.py` or `test_BLIP2.py` |
| `lavis` | (d) local repo | vendored checkout | N/A | `<checkout>/lavis` or `src/lavis` |
| `utils` | (d) local repo | vendored checkout | N/A | `workbench/third_party/SPRC/src/utils.py` |
| `validate_blip` | (d) local repo | vendored checkout | N/A | `workbench/third_party/SPRC/src/validate_blip.py`|

---

## 3. salesforce-lavis==1.0.2 Published Requires-Dist and Wheel Digest (Question 3)

### 3.1 Published Requires-Dist
Fetched directly from PyPI API (`https://pypi.org/pypi/salesforce-lavis/1.0.2/json`):
Exact 28 entries in `info.requires_dist`:
```text
contexttimer
decord
einops (>=0.4.1)
fairscale (==0.4.4)
ftfy
iopath
ipython
omegaconf
opencv-python-headless (==4.5.5.64)
opendatasets
packaging
pandas
plotly
pre-commit
pycocoevalcap
pycocotools
python-magic
scikit-image
sentencepiece
spacy
streamlit
timm (==0.4.12)
torch (>=1.10.0)
torchvision
tqdm
transformers (<4.27,>=4.25.0)
webdataset
wheel
```

### 3.2 Wheel SHA-256 Digest Verification
- URL: `https://files.pythonhosted.org/packages/77/1b/112981873d44de113ecaab8084da82606871ad1e914e00c6ceb68f96b394/salesforce_lavis-1.0.2-py3-none-any.whl`
- Target location: `/tmp/salesforce_lavis-1.0.2-py3-none-any.whl`
- File size: 1,812,319 bytes
- Computed SHA-256: `489782efc2d98d8e878112b2758f82ac2e6c10dfcfa087db6656465c7784d49f`
- Verification status: **EXACT MATCH** with expected hash.

---

## 4. Vendored Code vs Installed lavis Distribution (Question 4)

### 4.1 Quoted Evidence of sys.path Insertion and Module Stubbing
In the five wrappers (`workbench/replay/{airknow,conesep,habit,intent,hint}.py:133-136`):
```python
    for entry in (str(source_root), str(source_root / "lavis")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
```
In `workbench/replay/lavis_processors.py:48-78`:
```python
def _processors_stub(lavis_root: Path) -> types.ModuleType:
    module = types.ModuleType("lavis.processors")
    module.__path__ = [str(lavis_root / "processors")]
    sys.modules["lavis.processors"] = module
    module.BaseProcessor = importlib.import_module("lavis.processors.base_processor").BaseProcessor
    module.BlipCaptionProcessor = importlib.import_module(
        "lavis.processors.blip_processors").BlipCaptionProcessor
    return module


def _pinned_registry(lavis_root: Path) -> Any:
    if "lavis" not in sys.modules:
        package = types.ModuleType("lavis")
        package.__path__ = [str(lavis_root)]
        sys.modules["lavis"] = package
    registry = importlib.import_module("lavis.common.registry").registry
    if registry.get_path("library_root") is None:
        registry.register_path("library_root", str(lavis_root))
    return registry
```
In SPRC (`workbench/backend/adapters/models.py:220-246`):
`SPRCAdapter` launches `python <source_root>/src/blip_validate.py`. In Python, `sys.path[0]` is automatically prepended with `<source_root>/src`, which contains directory `<source_root>/src/lavis`.

### 4.2 Shadowing and Runtime Necessity
1. **Does the code under `workbench/third_party/<model>/lavis/` import the installed `lavis` distribution at runtime?**
   **NO.** In all six models, `sys.path.insert(0, ...)` or Python script directory resolution prepends the local vendored directory. Every `import lavis` resolves directly to the vendored copy. The installed `salesforce-lavis` distribution is 100% shadowed.
2. **Which parts of `salesforce-lavis` (if any) are still needed at runtime?**
   **NONE.** No python code, weight files, or assets from the installed `salesforce-lavis` distribution are imported or executed.
3. **The Phantom Dependency Problem**:
   Declaring `salesforce-lavis==1.0.2` in `workbench/registry/models.yaml` forces package planners and solvers to reconcile its strict, legacy dependency constraints:
   - `timm (==0.4.12)`
   - `transformers (<4.27,>=4.25.0)`
   - `opencv-python-headless (==4.5.5.64)`
   - `fairscale (==0.4.4)`
   Installing `salesforce-lavis` provides zero runtime code, but its declared pins break the environment solve against the authors' actual working pins (`timm==0.9.16`, `transformers==4.36.2`).

---

## 5. timm Import and Version Bounds (Question 5)

### 5.1 Quoted Import and Call Sites
1. **Import in `lavis/models/eva_vit.py:15-16`**:
   ```python
   from timm.models.layers import drop_path, to_2tuple, trunc_normal_
   from timm.models.registry import register_model
   ```
2. **Import in `lavis/common/dist_utils.py:14`**:
   ```python
   import timm.models.hub as timm_hub
   ```
3. **Call sites in `lavis/common/dist_utils.py:127,132`**:
   ```python
   cached_file = os.path.join(timm_hub.get_cache_dir(), filename)
   ...
   timm_hub.download_cached_file(url, check_hash, progress)
   ```
4. **Call sites in `lavis/models/eva_vit.py:84,103,138`**:
   Calls `to_2tuple`, `trunc_normal_`, and `drop_path` during EVA-ViT-g patch embedding and block construction.

### 5.2 Version Bounds: Distribution vs Vendored Code Needs
- **LAVIS Distribution Pin**: `salesforce-lavis==1.0.2` pins `timm (==0.4.12)`.
- **Vendored Code Needs**:
  - The vendored code contains no version checks.
  - In `timm < 1.0.0` (e.g. `0.4.12` through `0.9.16`), `timm.models.layers`, `timm.models.registry`, and `timm.models.hub` exist natively.
  - In `timm >= 1.0.0`, these paths emit `FutureWarning` deprecation warnings (`Importing from timm.models.layers is deprecated, please import via timm.layers`).
- **Author Declarations**:
  - Air-Know, ConeSep, HABIT, INTENT, HINT READMEs pin: `timm==0.9.16`
  - SPRC `requirements.txt` pins: `timm==0.9.12`
- **Resolution Conflict**:
  `salesforce-lavis==1.0.2` pins `timm==0.4.12` with strict equality. A standard pip solve fails immediately if both `salesforce-lavis==1.0.2` and `timm==0.9.16` are requested:
  `"Because salesforce-lavis==1.0.2 depends on timm==0.4.12 and you require timm==0.9.16, your requirements are unsatisfiable."`

---

## 6. transformers Imports and Version Bounds (Question 6)

### 6.1 Quoted Import Sites and Symbols
1. **In `datasets.py:13`** (Air-Know, ConeSep, HABIT, INTENT, HINT):
   ```python
   from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
   ```
2. **In `lavis/models/blip2_models/blip2.py:26`** (All 6 models):
   ```python
   from transformers import BertTokenizer
   ```
3. **In `lavis/models/blip2_models/Qformer.py:24-46`** (All 6 models):
   ```python
   from transformers.activations import ACT2FN
   from transformers.file_utils import ModelOutput
   from transformers.modeling_outputs import (
       BaseModelOutputWithPastAndCrossAttentions,
       BaseModelOutputWithPoolingAndCrossAttentions,
       CausalLMOutputWithCrossAttentions,
       MaskedLMOutput,
       MultipleChoiceModelOutput,
       NextSentencePredictorOutput,
       QuestionAnsweringModelOutput,
       SequenceClassifierOutput,
       TokenClassifierOutput,
   )
   from transformers.modeling_utils import (
       PreTrainedModel,
       apply_chunking_to_forward,
       find_pruneable_heads_and_indices,
       prune_linear_layer,
   )
   from transformers.utils import logging
   from transformers.models.bert.configuration_bert import BertConfig
   ```
4. **In `lavis/models/med.py:40-48`** (Imported via SPRC `src/lavis/__init__.py`):
   Imports the identical set from `transformers.modeling_utils`.

### 6.2 Minimum Version Providing All Symbols
- `CLIPSegProcessor` and `CLIPSegForImageSegmentation` were introduced in **`transformers==4.25.0`** (verified: absent in `4.24.0`, present in `4.25.0` wheel).
- Therefore, the minimum version providing every symbol used is **`transformers>=4.25.0`**.

### 6.3 Maximum Version / Breaking Boundary
- `apply_chunking_to_forward`, `find_pruneable_heads_and_indices`, and `prune_linear_layer` remained in `transformers.modeling_utils` through `v4.55.4`.
- In **`transformers==4.56.0`**, these symbols were removed from `transformers.modeling_utils` (moved to `transformers.pytorch_utils`).
- Verified failure on `transformers==4.57.6`:
  `ImportError: cannot import name 'apply_chunking_to_forward' from 'transformers.modeling_utils'`
- Usable version window for the vendored tree: **`4.25.0 <= transformers < 4.56.0`**.

---

## 7. SPRC Consistency: requirements.txt vs Vendored src/lavis (Question 7)

SPRC upstream ships `requirements.txt` with:
```text
pandas==2.2.0
tqdm==4.66.1
omegaconf==2.3.0
iopath==0.1.10
timm==0.9.12
decord==0.6.0
webdataset==0.2.86
opencv-python
transformers==4.36.2
fairscale==0.4.13
einops==0.7.0
scikit-image==0.22.0
spacy==3.7.2
pycocoevalcap==1.2
```

### 7.1 Conflicts with Registry Declaration (`models.yaml`)
`models.yaml:923` declares:
`packages: [transformers==4.36.2, timm==0.9.12, spacy==3.7.2, salesforce-lavis==1.0.2]`
1. **Contradiction with salesforce-lavis==1.0.2**:
   - `salesforce-lavis==1.0.2` pins `transformers (<4.27,>=4.25.0)` vs SPRC's `transformers==4.36.2`.
   - `salesforce-lavis==1.0.2` pins `timm (==0.4.12)` vs SPRC's `timm==0.9.12`.
   - `salesforce-lavis==1.0.2` pins `fairscale (==0.4.4)` vs SPRC's `fairscale==0.4.13`.
   - `salesforce-lavis==1.0.2` pins `opencv-python-headless (==4.5.5.64)` vs SPRC's `opencv-python`.
   These requirements are mathematically unsatisfiable by any PEP 517/518 package manager.

### 7.2 Missing Dependency in SPRC requirements.txt: OpenAI clip
- In `workbench/third_party/SPRC/src/blip_validate.py:8`:
  `import clip`
- `clip` is **NOT** in `SPRC/requirements.txt` and **NOT** in `models.yaml`.
- Observed failure when running `python src/blip_validate.py`:
  `ModuleNotFoundError: No module named 'clip'`

### 7.3 Missing Dependencies in models.yaml Caused by Unstubbed Imports
Because `blip_validate.py` does `from lavis.models import load_model_and_preprocess` without stubbing, `SPRC/src/lavis/__init__.py` imports `lavis.models.*` and `lavis.tasks.*`:
- `skimage` (`scikit-image`): imported in `blip2_qformer_cir_cat.py:23`. Present in `requirements.txt`, missing from `models.yaml`.
- `pycocotools`: imported in `lavis/tasks/captioning.py:105` (`from pycocotools.coco import COCO`). Missing from **BOTH** `requirements.txt` and `models.yaml`.

---

## 8. Training-Only vs Runtime Packages (Question 8)

The following declared packages are needed only for training code paths or are completely unused at runtime:

1. **`scikit-learn` (`sklearn`)**:
   - **HINT**: Completely unused. 0 references in the entire repository.
   - **SPRC**: Completely unused. 0 references in the entire repository.
   - **Air-Know / ConeSep / INTENT**: Contains a dead top-level import (`from sklearn.cluster import KMeans` in `blip2_cir.py` / `ConeSep.py`), but `KMeans` is never called anywhere in the file or repository.
   - **HABIT**: Calls `DBSCAN(eps=0.01, min_samples=10, n_jobs=-1)` at `HABIT.py:342,344` inside `def forward()`, which computes training loss (`robust_infoNCE` soft labels). The evaluation wrapper calls only `extract_retrieval_compose` and `extract_retrieval_target`. `forward()` is never executed at runtime.
   - *Verdict*: Needed at runtime solely because PyTorch unpickler executes the module-level import when deserializing checkpoints. The actual algorithms are 100% training-only.
2. **`open-clip-torch`**:
   - Declared for `habit`, `intent`, and `hint`.
   - Grep verification across all six checkouts: **ZERO** occurrences of `import open_clip`.
   - *Verdict*: Completely spurious. Unused in both training and inference.
3. **`salesforce-lavis`**:
   - Declared for all six models.
   - *Verdict*: Completely unused at runtime. The vendored `lavis` directory shadows the pip distribution entirely.
4. **`spacy`**:
   - Declared for `sprc`.
   - Inside SPRC, `spacy` is only referenced in `blip2.py:170` (the `lemmatizer` property for VQA answer normalization) and `img2prompt_vqa.py:15`.
   - *Verdict*: Completely unused during FashionIQ evaluation. It is only imported because `src/lavis/models/__init__.py` imports `Img2PromptVQA`.

---

## 9. Per-Model Dependency and Runtime Tables (Acceptance Criteria)

### 9.1 Air-Know
| Package | Provided by | Declared in registry? | Needed at runtime? | Evidence |
|:---|:---|:---|:---|:---|
| `torch` | PyTorch wheel | YES (`pytorch: 2.1.0`) | YES | `workbench/replay/airknow.py:132`, `datasets.py:5`, `test_BLIP2.py:2` |
| `torchvision` | PyTorch wheel | YES (`torchvision: 0.16.0`) | YES | `datasets.py:10`, `data_utils.py:8` |
| `transformers` | PyPI | YES (`transformers==4.25.0`) | YES | `datasets.py:13` (`CLIPSegProcessor`), `Qformer.py:24-46` |
| `timm` | PyPI | YES (`timm==0.9.16`) | YES | `eva_vit.py:15-16`, `dist_utils.py:14` |
| `scikit-learn` | PyPI | YES (`scikit-learn==1.3.2`) | YES (import only) | `blip2_cir.py:19` (module import executed during unpickling; `KMeans` never called) |
| `salesforce-lavis` | PyPI | YES (`salesforce-lavis==1.0.2`) | **NO** | Vendored `AirKnow/lavis` shadows distribution; provides 0 runtime symbols |
| `omegaconf` | PyPI | NO (undeclared) | YES | `workbench/replay/lavis_processors.py:107` |
| `pillow` | PyPI | NO (undeclared) | YES | `data_utils.py:6` (imported as `PIL`) |
| `tqdm` | PyPI | NO (undeclared) | YES | `test_BLIP2.py:3` |
| `iopath` | PyPI | NO (undeclared) | YES | `lavis/common/utils.py:25` |
| `fairscale` | PyPI | NO (undeclared) | YES | `lavis/models/clip_vit.py:10` |

### 9.2 ConeSep
| Package | Provided by | Declared in registry? | Needed at runtime? | Evidence |
|:---|:---|:---|:---|:---|
| `torch` | PyTorch wheel | YES (`pytorch: 2.1.0`) | YES | `workbench/replay/conesep.py:129`, `datasets.py:5`, `test.py:2` |
| `torchvision` | PyTorch wheel | YES (`torchvision: 0.16.0`) | YES | `datasets.py:10`, `data_utils.py:8` |
| `transformers` | PyPI | YES (`transformers==4.25.0`) | YES | `datasets.py:13` (`CLIPSegProcessor`), `Qformer.py:24-46` |
| `timm` | PyPI | YES (`timm==0.9.16`) | YES | `eva_vit.py:15-16`, `dist_utils.py:14` |
| `scikit-learn` | PyPI | YES (`scikit-learn==1.3.2`) | YES (import only) | `ConeSep.py:24` (module import executed during unpickling; `KMeans` never called) |
| `salesforce-lavis` | PyPI | YES (`salesforce-lavis==1.0.2`) | **NO** | Vendored `ConeSep/lavis` shadows distribution; provides 0 runtime symbols |
| `omegaconf` | PyPI | NO (undeclared) | YES | `workbench/replay/lavis_processors.py:107` |
| `pillow` | PyPI | NO (undeclared) | YES | `data_utils.py:6` (imported as `PIL`) |
| `tqdm` | PyPI | NO (undeclared) | YES | `test.py:3` |
| `iopath` | PyPI | NO (undeclared) | YES | `lavis/common/utils.py:25` |
| `fairscale` | PyPI | NO (undeclared) | YES | `lavis/models/clip_vit.py:10` |

### 9.3 HABIT
| Package | Provided by | Declared in registry? | Needed at runtime? | Evidence |
|:---|:---|:---|:---|:---|
| `torch` | PyTorch wheel | YES (`pytorch: 2.1.0`) | YES | `workbench/replay/habit.py:117`, `datasets.py:5`, `test.py:2` |
| `torchvision` | PyTorch wheel | YES (`torchvision: 0.16.0`) | YES | `datasets.py:10`, `data_utils.py:8` |
| `transformers` | PyPI | YES (`transformers==4.25.0`) | YES | `datasets.py:13` (`CLIPSegProcessor`), `Qformer.py:24-46` |
| `timm` | PyPI | YES (`timm==0.9.16`) | YES | `eva_vit.py:15-16`, `dist_utils.py:14` |
| `scikit-learn` | PyPI | YES (`scikit-learn==1.3.2`) | YES (import only) | `HABIT.py:24-27` (module import during unpickling; DBSCAN called only in training forward) |
| `open-clip-torch`| PyPI | YES (`open-clip-torch==2.24.0`)| **NO** | Spurious declaration. Zero imports in `HABIT` checkout |
| `salesforce-lavis` | PyPI | YES (`salesforce-lavis==1.0.2`) | **NO** | Vendored `HABIT/lavis` shadows distribution; provides 0 runtime symbols |
| `omegaconf` | PyPI | NO (undeclared) | YES | `workbench/replay/lavis_processors.py:107` |
| `pillow` | PyPI | NO (undeclared) | YES | `data_utils.py:6` (imported as `PIL`) |
| `tqdm` | PyPI | NO (undeclared) | YES | `test.py:3` |
| `iopath` | PyPI | NO (undeclared) | YES | `lavis/common/utils.py:25` |
| `fairscale` | PyPI | NO (undeclared) | YES | `lavis/models/clip_vit.py:10` |

### 9.4 INTENT
| Package | Provided by | Declared in registry? | Needed at runtime? | Evidence |
|:---|:---|:---|:---|:---|
| `torch` | PyTorch wheel | YES (`pytorch: 2.1.0`) | YES | `workbench/replay/intent.py:117`, `datasets.py:5`, `test.py:2` |
| `torchvision` | PyTorch wheel | YES (`torchvision: 0.16.0`) | YES | `datasets.py:10`, `data_utils.py:8` |
| `transformers` | PyPI | YES (`transformers==4.25.0`) | YES | `datasets.py:13` (`CLIPSegProcessor`), `Qformer.py:24-46` |
| `timm` | PyPI | YES (`timm==0.9.16`) | YES | `eva_vit.py:15-16`, `dist_utils.py:14` |
| `scikit-learn` | PyPI | YES (`scikit-learn==1.3.2`) | YES (import only) | `blip2_cir.py:30` (module import during unpickling; `KMeans` never called) |
| `open-clip-torch`| PyPI | YES (`open-clip-torch==2.24.0`)| **NO** | Spurious declaration. Zero imports in `INTENT` checkout |
| `salesforce-lavis` | PyPI | YES (`salesforce-lavis==1.0.2`) | **NO** | Vendored `INTENT/lavis` shadows distribution; provides 0 runtime symbols |
| `omegaconf` | PyPI | NO (undeclared) | YES | `workbench/replay/lavis_processors.py:107` |
| `pillow` | PyPI | NO (undeclared) | YES | `data_utils.py:6` (imported as `PIL`) |
| `tqdm` | PyPI | NO (undeclared) | YES | `test.py:3` |
| `iopath` | PyPI | NO (undeclared) | YES | `lavis/common/utils.py:25` |
| `fairscale` | PyPI | NO (undeclared) | YES | `lavis/models/clip_vit.py:10` |

### 9.5 HINT
| Package | Provided by | Declared in registry? | Needed at runtime? | Evidence |
|:---|:---|:---|:---|:---|
| `torch` | PyTorch wheel | YES (`pytorch: 2.1.0`) | YES | `workbench/replay/hint.py:146`, `datasets.py:5`, `test.py:4` |
| `torchvision` | PyTorch wheel | YES (`torchvision: 0.16.0`) | YES | `datasets.py:10`, `data_utils.py:8` |
| `transformers` | PyPI | YES (`transformers==4.25.0`) | YES | `datasets.py:13` (`CLIPSegProcessor`), `Qformer.py:24-46` |
| `timm` | PyPI | YES (`timm==0.9.16`) | YES | `eva_vit.py:15-16`, `dist_utils.py:14` |
| `scikit-learn` | PyPI | YES (`scikit-learn==1.3.2`) | **NO** | Completely unused. Zero imports in `HINT` checkout |
| `open-clip-torch`| PyPI | YES (`open-clip-torch==2.24.0`)| **NO** | Spurious declaration. Zero imports in `HINT` checkout |
| `salesforce-lavis` | PyPI | YES (`salesforce-lavis==1.0.2`) | **NO** | Vendored `HINT/lavis` shadows distribution; provides 0 runtime symbols |
| `omegaconf` | PyPI | NO (undeclared) | YES | `workbench/replay/lavis_processors.py:107` |
| `pillow` | PyPI | NO (undeclared) | YES | `data_utils.py:6` (imported as `PIL`) |
| `tqdm` | PyPI | NO (undeclared) | YES | `test.py:5` |
| `iopath` | PyPI | NO (undeclared) | YES | `lavis/common/utils.py:25` |
| `fairscale` | PyPI | NO (undeclared) | YES | `lavis/models/clip_vit.py:10` |

### 9.6 SPRC
| Package | Provided by | Declared in registry? | Needed at runtime? | Evidence |
|:---|:---|:---|:---|:---|
| `torch` | PyTorch wheel | YES (`pytorch: 2.0.1`) | YES | `src/blip_validate.py:11`, `src/data_utils.py:10` |
| `torchvision` | PyTorch wheel | YES (`torchvision: 0.15.2`) | YES | `src/data_utils.py:7-9` |
| `transformers` | PyPI | YES (`transformers==4.36.2`) | YES | `Qformer.py:24-46`, `med.py:40-48` |
| `timm` | PyPI | YES (`timm==0.9.12`) | YES | `eva_vit.py:15-16`, `dist_utils.py:14` |
| `spacy` | PyPI | YES (`spacy==3.7.2`) | YES (import only) | `img2prompt_vqa.py:15` (imported via `lavis/models/__init__.py`) |
| `salesforce-lavis` | PyPI | YES (`salesforce-lavis==1.0.2`) | **NO** | Vendored `SPRC/src/lavis` shadows distribution; provides 0 runtime symbols |
| `clip` (OpenAI)| GitHub / PyPI | **NO (undeclared)** | **YES (CRITICAL)**| `src/blip_validate.py:8` (`import clip`) |
| `pandas` | PyPI | NO (in requirements.txt) | YES | `src/blip_validate.py:10` |
| `tqdm` | PyPI | NO (in requirements.txt) | YES | `src/blip_validate.py:15` |
| `omegaconf` | PyPI | NO (in requirements.txt) | YES | `src/lavis/__init__.py:12` |
| `iopath` | PyPI | NO (in requirements.txt) | YES | `src/lavis/common/utils.py:25` |
| `decord` | PyPI | NO (in requirements.txt) | YES (import only) | `src/lavis/datasets/data_utils.py:15` (imported via `lavis/__init__.py`) |
| `webdataset` | PyPI | NO (in requirements.txt) | YES (import only) | `src/lavis/datasets/data_utils.py:16` (imported via `lavis/__init__.py`) |
| `fairscale` | PyPI | NO (in requirements.txt) | YES | `src/lavis/models/clip_vit.py:10` |
| `scikit-image` | PyPI | NO (in requirements.txt) | YES (import only) | `blip2_qformer_cir_cat.py:23` (imported as `skimage`) |
| `pycocoevalcap`| PyPI | NO (in requirements.txt) | YES (import only) | `lavis/tasks/captioning.py:104` (imported via `lavis/__init__.py`) |
| `pycocotools` | PyPI | **NO (undeclared)** | YES (import only) | `lavis/tasks/captioning.py:105` (imported via `lavis/__init__.py`) |
| `opencv-python`| PyPI | NO (in requirements.txt) | YES | `src/validate_blip.py:19` (imported as `cv2`) |

---

## 10. Contradictions Between Declared Pins (Acceptance Criteria)

1. **`salesforce-lavis==1.0.2` vs `timm==0.9.16` / `0.9.12`**:
   - `salesforce-lavis==1.0.2` pins `timm (==0.4.12)`.
   - `models.yaml` declares `timm==0.9.16` (AirKnow, ConeSep, HABIT, INTENT, HINT) and `timm==0.9.12` (SPRC).
   - *Contradiction*: Resolving `salesforce-lavis==1.0.2` with any modern `timm` fails closed with an unsatisfiable version conflict.
2. **`salesforce-lavis==1.0.2` vs `transformers==4.36.2` (SPRC)**:
   - `salesforce-lavis==1.0.2` pins `transformers (<4.27,>=4.25.0)`.
   - `models.yaml` declares `transformers==4.36.2` for SPRC.
   - *Contradiction*: `4.36.2` violates the upper bound `<4.27`.
3. **`salesforce-lavis==1.0.2` vs `fairscale==0.4.13` and `opencv-python` (SPRC)**:
   - `salesforce-lavis==1.0.2` pins `fairscale (==0.4.4)` and `opencv-python-headless (==4.5.5.64)`.
   - SPRC upstream requires `fairscale==0.4.13` and `opencv-python`.
4. **SPRC Entrypoint vs Declared Packages**:
   - `SPRC/src/blip_validate.py:8` executes `import clip`.
   - Neither `SPRC/requirements.txt` nor `models.yaml:sprc` declares package `clip`.
   - *Contradiction*: The entrypoint crashes on the 8th line of execution on a clean install matching the declaration.
5. **SPRC Unstubbed Imports vs Declared Packages**:
   - SPRC's unstubbed `from lavis.models import load_model_and_preprocess` imports `pycocotools` and `scikit-image`.
   - Neither is declared in `models.yaml:sprc`.
6. **Spurious `open-clip-torch==2.24.0` in HABIT, INTENT, HINT**:
   - Declared in `models.yaml` for all three models.
   - The vendored checkouts have **ZERO** imports of `open_clip`.
7. **Spurious `scikit-learn==1.3.2` in HINT**:
   - Declared in `models.yaml` for HINT.
   - The vendored checkout has **ZERO** imports of `sklearn`.

---

## Verified claims

### Claim 1: salesforce-lavis==1.0.2 wheel SHA-256 digest and Requires-Dist
**Command**:
```bash
python3 -c "
import urllib.request, hashlib, json
data = json.loads(urllib.request.urlopen('https://pypi.org/pypi/salesforce-lavis/1.0.2/json').read().decode())
print('Requires-Dist count:', len(data['info']['requires_dist']))
url = 'https://files.pythonhosted.org/packages/77/1b/112981873d44de113ecaab8084da82606871ad1e914e00c6ceb68f96b394/salesforce_lavis-1.0.2-py3-none-any.whl'
raw = urllib.request.urlopen(url).read()
digest = hashlib.sha256(raw).hexdigest()
print('Wheel sha256:', digest)
assert digest == '489782efc2d98d8e878112b2758f82ac2e6c10dfcfa087db6656465c7784d49f'
"
```
**Observed Output**:
```text
Requires-Dist count: 28
Wheel sha256: 489782efc2d98d8e878112b2758f82ac2e6c10dfcfa087db6656465c7784d49f
```

### Claim 2: Pip resolution conflict between salesforce-lavis 1.0.2 and declared timm 0.9.16
**Command**:
```bash
uv pip compile - <<'EOF'
salesforce-lavis==1.0.2
timm==0.9.16
EOF
```
**Observed Output**:
```text
  × No solution found when resolving dependencies:
  ╰─▶ Because salesforce-lavis==1.0.2 depends on timm==0.4.12 and you require
      salesforce-lavis==1.0.2, we can conclude that you require timm==0.4.12.
      And because you require timm==0.9.16, we can conclude that your
      requirements are unsatisfiable.
```

### Claim 3: Pip resolution conflict between salesforce-lavis 1.0.2 and SPRC transformers 4.36.2
**Command**:
```bash
uv pip compile - <<'EOF'
salesforce-lavis==1.0.2
transformers==4.36.2
timm==0.9.12
EOF
```
**Observed Output**:
```text
  × No solution found when resolving dependencies:
  ╰─▶ Because salesforce-lavis==1.0.2 depends on transformers>=4.25.0,<4.27
      and you require salesforce-lavis==1.0.2, we can conclude that you
      require transformers>=4.25.0,<4.27.
      And because you require transformers==4.36.2, we can conclude that your
      requirements are unsatisfiable.
```

### Claim 4: SPRC entrypoint crashes on undeclared clip module
**Command**:
```bash
PYTHONPATH=workbench/third_party/SPRC/src .venv/bin/python workbench/third_party/SPRC/src/blip_validate.py --help
```
**Observed Output**:
```text
Traceback (most recent call last):
  File "/home/heheboiz/data/cir/workbench/third_party/SPRC/src/blip_validate.py", line 8, in <module>
    import clip
ModuleNotFoundError: No module named 'clip'
```

### Claim 5: transformers 4.56+ breaks vendored Q-Former import
**Command**:
```bash
.venv/bin/python -c "
try:
    from transformers.modeling_utils import apply_chunking_to_forward
    print('SUCCESS')
except ImportError as e:
    print('OBSERVED FAILURE:', e)
"
```
**Observed Output**:
```text
OBSERVED FAILURE: cannot import name 'apply_chunking_to_forward' from 'transformers.modeling_utils' (/home/heheboiz/data/cir/.venv/lib/python3.13/site-packages/transformers/modeling_utils.py)
```

### Claim 6: Absence of workbench/replay/sprc.py
**Command**:
```bash
test -f workbench/replay/sprc.py && echo "EXISTS" || echo "ABSENT"
```
**Observed Output**:
```text
ABSENT
```

### Claim 7: Zero open_clip occurrences in the six target checkouts
**Command**:
```bash
python3 -c "
from pathlib import Path
for m in ['AirKnow', 'ConeSep', 'HABIT', 'INTENT', 'HINT', 'SPRC']:
    p = Path('workbench/third_party') / m
    matches = [f for f in p.glob('**/*.py') if 'import open_clip' in f.read_text(encoding='utf-8', errors='ignore')]
    print(m, 'open_clip imports:', len(matches))
"
```
**Observed Output**:
```text
AirKnow open_clip imports: 0
ConeSep open_clip imports: 0
HABIT open_clip imports: 0
INTENT open_clip imports: 0
HINT open_clip imports: 0
SPRC open_clip imports: 0
```

### Claim 8: Zero sklearn occurrences in HINT and SPRC checkouts
**Command**:
```bash
python3 -c "
from pathlib import Path
for m in ['HINT', 'SPRC']:
    p = Path('workbench/third_party') / m
    matches = [f for f in p.glob('**/*.py') if 'sklearn' in f.read_text(encoding='utf-8', errors='ignore')]
    print(m, 'sklearn occurrences:', len(matches))
"
```
**Observed Output**:
```text
HINT sklearn occurrences: 0
SPRC sklearn occurrences: 0
```

### Claim 9: Execution of text_processors parity in .venv without salesforce-lavis installed
**Command**:
```bash
.venv/bin/python -c "
from pathlib import Path
from workbench.replay.lavis_processors import text_processors
source_root = Path('workbench/third_party/HINT')
proc, boundary = text_processors(source_root, 'configs/models/blip2/blip2_pretrain.yaml')
print('Loaded processors:', sorted(proc.keys()), 'boundary:', boundary)
"
```
**Observed Output**:
```text
Loaded processors: ['eval', 'train'] boundary: lavis/models/__init__.py:load_preprocess
```

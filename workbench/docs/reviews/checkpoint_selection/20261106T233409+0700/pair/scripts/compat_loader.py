"""EXPLORATORY_PUBLIC_MAPPING loader for the released PAIR whole-model pickles.

The pickles name three modules the public repository does not ship:

    model_vssp        (Encoder, Backbone, FeatureWiseAffine)
    MHTransformer_dis (MlpTransformer, TransformerLayer, Transformer, MultiHeadAttention)
    MHTransformer      (public, present in workbench/third_party/PAIR/models)

This loader installs *subprocess-local* module aliases so that unpickling binds those names to
real public implementations. `__init__` is not re-run for unpickled modules: every tensor, buffer
and child module comes from the checkpoint; only the *methods* come from the public classes.

Mapping (explicit, per class):

    model_vssp.Encoder            -> PAIR.models.model.PAIR
    model_vssp.Backbone           -> PAIR.models.model.Backbone
    model_vssp.FeatureWiseAffine  -> PAIR.models.model.FeatureWiseAffine
    MHTransformer.*               -> PAIR.models.MHTransformer.*          (verbatim public file)
    MHTransformer_dis.MlpTransformer     -> MHTransformer.MlpTransformer
    MHTransformer_dis.TransformerLayer   -> MHTransformer.TransformerLayer
    MHTransformer_dis.Transformer        -> MHTransformer.Transformer
    MHTransformer_dis.MultiHeadAttention -> HYPOTHESIS-DEPENDENT, see ATTENTION_HYPOTHESES

Justification for the model_vssp mapping is structural, not nominal:
  * restored root __dict__ keys == public PAIR.__init__ assignments exactly
    (backbone, loss_T, local_weight[n+2], local_weight_dis[n+1], t, affine);
  * restored backbone is a strict superset of public Backbone.__init__ assignments;
  * public Backbone.extract_img_fea_patch_selection returns the 8-tuple that public
    PAIR.compose_feature unpacks.

Mappings that are NOT claimed: that MHTransformer_dis used the same attention normalisation, and
that the author's unpublished model_vssp is semantically identical to the public PAIR class.
Both remain hypotheses; the loader labels results accordingly.
"""
import hashlib
import os
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

EXPECTED_SHA256 = {
    "B1": "c4d861ecd0baa3f1427fc7a853943f5614baf87def1cdd9c4bb7583ee925fbf2",
    "B2": "74346fd7de6efdafe8dde70458316286a60ceb2d7b419efed55193418aeb3f03",
}

_HERE = Path(__file__).resolve()
# Allow an explicit override; otherwise walk up to the checkout that owns workbench/third_party.
PAIR_MODELS = Path(os.environ["PAIR_SRC"]) if os.environ.get("PAIR_SRC") else (
    next((p / "workbench" / "third_party" / "PAIR" / "models"
          for p in _HERE.parents if (p / "workbench" / "third_party" / "PAIR" / "models").is_dir()),
         _HERE.parents[5] / "third_party" / "PAIR" / "models"))

ATTENTION_HYPOTHESES = {
    # MHTransformer_dis.MultiHeadAttention is absent from every public revision. Two candidate
    # readings of its name and of public PAIR's own use of the reverse branch:
    "H_normal": ("MHTransformer.MultiHeadAttention",
                 "MHTransformer_dis holds the ordinary softmax attention; public PAIR's "
                 "transformer_b was the ordinary branch."),
    "H_rev": ("MHTransformer_rev.MultiHeadAttentionRev",
              "MHTransformer_dis holds reverse attention (1 - softmax); matches public PAIR's "
              "own transformer_b = MHTransformer_rev.Transformer assignment."),
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_public(name):
    """Import a public PAIR module by path, bypassing package semantics."""
    if name in sys.modules:
        return sys.modules[name]
    path = PAIR_MODELS / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def install_aliases(hypothesis):
    if hypothesis not in ATTENTION_HYPOTHESES:
        raise ValueError(f"unknown hypothesis {hypothesis!r}")
    attn_target = ATTENTION_HYPOTHESES[hypothesis][0]

    if str(PAIR_MODELS) not in sys.path:
        sys.path.insert(0, str(PAIR_MODELS))
    model_mod = _load_public("model")
    mht = _load_public("MHTransformer")
    mht_rev = _load_public("MHTransformer_rev")

    vssp = types.ModuleType("model_vssp")
    vssp.Encoder = model_mod.PAIR
    vssp.Backbone = model_mod.Backbone
    vssp.FeatureWiseAffine = model_mod.FeatureWiseAffine
    sys.modules["model_vssp"] = vssp

    dis = types.ModuleType("MHTransformer_dis")
    dis.MlpTransformer = mht.MlpTransformer
    dis.TransformerLayer = mht.TransformerLayer
    dis.Transformer = mht.Transformer
    src_mod, _, src_name = attn_target.rpartition(".")
    dis.MultiHeadAttention = getattr(sys.modules[src_mod], src_name)
    sys.modules["MHTransformer_dis"] = dis

    return {
        "model_vssp.Encoder": "model.PAIR",
        "model_vssp.Backbone": "model.Backbone",
        "model_vssp.FeatureWiseAffine": "model.FeatureWiseAffine",
        "MHTransformer.*": "MHTransformer.*",
        "MHTransformer_dis.MlpTransformer": "MHTransformer.MlpTransformer",
        "MHTransformer_dis.TransformerLayer": "MHTransformer.TransformerLayer",
        "MHTransformer_dis.Transformer": "MHTransformer.Transformer",
        "MHTransformer_dis.MultiHeadAttention": attn_target,
        "hypothesis": hypothesis,
        "hypothesis_rationale": ATTENTION_HYPOTHESES[hypothesis][1],
    }


def load(path, tag, hypothesis, map_location="cpu"):
    """Hash-gated whole-model load. Raises before any unpickle on a hash mismatch."""
    actual = sha256_of(path)
    expected = EXPECTED_SHA256[tag]
    if actual != expected:
        raise SystemExit(f"HASH_GATE_FAIL {tag}: {actual} != {expected}")
    mapping = install_aliases(hypothesis)
    obj = torch.load(path, map_location=map_location, weights_only=False)
    return obj, {"sha256": actual, "mapping": mapping, "path": str(path)}

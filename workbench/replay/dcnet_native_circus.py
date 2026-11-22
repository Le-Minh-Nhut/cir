"""Independent DCNet CIRCUS native capture for one ordered query/full gallery."""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

MODALITIES = ("im_feat", "inter-1", "spatial0", "spatial1", "spatial2", "spatial3", "spatial4")
EXPERT_DIMS = {"im_feat": 2048, "inter-1": 1024, **{f"spatial{i}": 2048 for i in range(5)}}


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--release-jsonl", type=Path, required=True)
    p.add_argument("--gallery-audit", type=Path, required=True)
    p.add_argument("--qid", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--feature-output", type=Path, required=True,
                   help="directory for packed native query/gallery feature memmaps")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                   help="execution device; default cuda preserves native GPU refusal")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=512)
    return p.parse_args()


def _image(root: Path, image_id: str) -> Image.Image:
    path = root / "resized_images" / f"{image_id}.jpg"
    if not path.is_file():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def _captions(release: Path, qid: str) -> tuple[str, str, list[str]]:
    for line in release.open(encoding="utf-8"):
        row = json.loads(line)
        if row["qid"] != qid:
            continue
        source = json.loads(row["query_src_content"])["candidate_img_id"]
        return source, row["query_txt"], [str(row["query_txt"])]
    raise KeyError(qid)


def _text_vectors(root: Path, text: str, qid: str) -> tuple[torch.Tensor, torch.Tensor]:
    # Prefer authoritative preprocessed row for known FashionIQ query; fallback uses only
    # saved author vocabulary, never random OOV vectors.
    cap_dir = root / "captions"
    for category in ("dress", "shirt", "toptee"):
        for split in ("train", "val", "test"):
            path = cap_dir / f"cap.{category}.glove.{split}.pkl"
            if not path.is_file():
                continue
            with path.open("rb") as stream:
                rows = pickle.load(stream)
            for row in rows:
                joined = " and ".join(str(item) for item in row.get("captions", [])) + "."
                if joined.lower() == text.lower():
                    values = np.asarray(row["wv"], dtype=np.float32)
                    return torch.from_numpy(values).unsqueeze(0), torch.tensor([values.shape[0]])
    with (cap_dir / "glove_vecs.pkl").open("rb") as stream:
        vocab = pickle.load(stream)
    import nltk
    tokens = nltk.tokenize.word_tokenize(text.lower().replace("-", " ").replace(".", ""))
    values = [np.asarray(vocab[token], dtype=np.float32) for token in tokens if token in vocab]
    if not values:
        raise ValueError(f"no source vocabulary tokens for {qid}")
    matrix = np.stack(values).astype(np.float32, copy=False)
    return torch.from_numpy(matrix).unsqueeze(0), torch.tensor([matrix.shape[0]])


def _model(checkpoint: Path, device: str | torch.device = "cuda"):
    source = Path(__file__).resolve().parents[1] / "third_party" / "DCNet"
    sys.path.insert(0, str(source))
    from model.model import CENet
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DCNet native CIRCUS diagnostic requires CUDA; refusing implicit CPU fallback")
    model = CENet(text_dim=300, composition="tirg", correction="fd", fusion="hadamard",
                  expert_dims=EXPERT_DIMS, same_dim=1024, text_feat="glove", norm_scale=5,
                  backbone="resnet", text_pooling_module="sharedconv", metric_fn="ip")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("state_dict", state), strict=True)
    return model.to(requested).eval(), requested


def _encode_gallery(model, gallery_ids: list[str], transform, root: Path, device: torch.device,
                    batch_size: int) -> dict[str, torch.Tensor]:
    parts = {modality: [] for modality in model.modalities}
    for start in range(0, len(gallery_ids), batch_size):
        ids = gallery_ids[start:start + batch_size]
        images = torch.stack([transform(_image(root, item)) for item in ids]).to(device)
        encoded = model.get_image_emb({"im_feat": images}, istgt=True)
        for modality in model.modalities:
            parts[modality].append(encoded[modality])
    return {modality: torch.cat(chunks, dim=0) for modality, chunks in parts.items()}


def _configure_deterministic(device: torch.device, seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DCNet native CIRCUS diagnostic requires CUDA; refusing implicit CPU fallback")
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


@torch.no_grad()
def main() -> int:
    args = _args()
    if args.batch_size < 1 or args.chunk_size < 1:
        raise ValueError("batch and chunk sizes must be positive")
    device = torch.device(args.device)
    _configure_deterministic(device, args.seed)
    source_id, query_text, _ = _captions(args.release_jsonl, args.qid)
    audit = json.loads(args.gallery_audit.read_text(encoding="utf-8"))
    gallery_ids = [Path(item).stem for item in audit["gallery_paths"]]
    feature_dir = args.feature_output
    feature_dir.mkdir(parents=True, exist_ok=True)
    query_memmap = np.lib.format.open_memmap(feature_dir / "query_context.npy", mode="w+",
                                             dtype=np.float32, shape=(1, len(MODALITIES) * 1024 * 3))
    gallery_memmap = np.lib.format.open_memmap(feature_dir / "gallery_context.npy", mode="w+",
                                               dtype=np.float32,
                                               shape=(len(gallery_ids), len(MODALITIES) * 1024))
    np.save(feature_dir / "gallery_ids.npy", np.asarray(gallery_ids))
    model, device = _model(args.checkpoint, device)
    transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    source_input = torch.stack([transform(_image(args.dataset_root, source_id))]).to(device)
    source_experts = model.get_image_emb({"im_feat": source_input}, istgt=False)
    text, text_len = _text_vectors(args.dataset_root, query_text, args.qid)
    text_experts = model.get_text_emb(text.to(device), text_len.to(device))
    composed = model.get_composition(source_experts, text_experts)
    query_context = torch.cat(
        [torch.cat([source_experts[mod] for mod in model.modalities], dim=1),
         torch.cat([text_experts[mod] for mod in model.modalities], dim=1),
         torch.cat([composed[mod] for mod in model.modalities], dim=1)], dim=1)
    query_memmap[:] = query_context.cpu().numpy()
    query_memmap.flush()
    comp_chunks: list[torch.Tensor] = []
    corr_chunks: list[torch.Tensor] = []
    for start in range(0, len(gallery_ids), args.chunk_size):
        ids = gallery_ids[start:start + args.chunk_size]
        gallery = _encode_gallery(model, ids, transform, args.dataset_root, device, args.batch_size)
        gallery_context = torch.cat([gallery[mod] for mod in model.modalities], dim=1)
        gallery_memmap[start:start + len(ids)] = gallery_context.cpu().numpy()
        gallery_memmap.flush()
        comp_chunks.append(model.get_score_matrix(gallery, composed, model.modalities,
                                                   l2renorm=False, dist=False, val=True))
        correction_source = {mod: value.mean((1, 2)) if model.feat2d else value
                             for mod, value in source_experts.items()}
        corrected = model.get_correction(correction_source, gallery)
        corr_chunks.append(model.get_score_matrix(corrected, text_experts, model.modalities,
                                                   l2renorm=False, dist=False, val=True))
    comp = torch.cat(comp_chunks, dim=-1)
    corr = torch.cat(corr_chunks, dim=-1)
    fused = F.log_softmax(comp, dim=-1) + F.log_softmax(corr, dim=-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, qid=args.qid, source_id=source_id,
                        gallery_ids=np.asarray(gallery_ids), composition=comp.cpu().numpy(),
                        correction=corr.cpu().numpy(), fused=fused.cpu().numpy())
    print(json.dumps({"qid": args.qid, "gallery_count": len(gallery_ids),
                      "source_id": source_id, "device": str(device),
                      "output": str(args.output)}))
    metadata = {"qid": args.qid, "query_shape": list(query_memmap.shape),
                "gallery_shape": list(gallery_memmap.shape), "batch_size": args.batch_size,
                "chunk_size": args.chunk_size, "device": str(device), "seed": args.seed,
                "tf32": False, "autocast": False}
    (feature_dir / "metadata.json").write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

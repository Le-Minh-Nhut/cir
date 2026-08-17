import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForCausalLM
from datasets.common import DirectoryImageStore
from datasets.fashioniq import load_fashioniq_split_ids


CATEGORIES = ("dress", "shirt", "toptee")
VALID_SPLITS = ("train", "val")
MODEL_NAME = "qihoo360/fg-clip2-large"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-root", type=Path, default=Path("data/fashionIQ_dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("features/fashioniq/fgclip2_large"))
    parser.add_argument("--splits", nargs="+", choices=VALID_SPLITS, default=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()

def load_all_image_ids(split_root: str | Path, split: str) -> list[str]:
    image_ids = []
    seen = set()

    for category in CATEGORIES:
        category_ids = load_fashioniq_split_ids(split_root=split_root, split=split, category=category,)

        for image_id in category_ids:
            if image_id not in seen:
                seen.add(image_id)
                image_ids.append(image_id)

    return image_ids

def load_fgclip2(device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME,)
    model.eval()

    return model, processor

# lấy ra feature 
@torch.inference_mode()
def encode_batch(model, processor, images, device):
    inputs = processor(images=images, max_num_patches=784, return_tensors="pt").to(device)
    features = model.get_image_features(**inputs)
    features = F.normalize(features.float(), dim=-1)

    return features

def precompute(image_ids, image_store, model, processor, device, batch_size):
    feature_batches = []

    for start in tqdm(range(0, len(image_ids), batch_size), desc="Encoding images"):
        batch_ids = image_ids[start:start + batch_size]

        images = [
            image_store.load(image_id)
            for image_id in batch_ids
        ]

        features = encode_batch(
            model=model,
            processor=processor,
            images=images,
            device=device,
        )

        feature_batches.append(features.cpu())

    return torch.cat(feature_batches, dim=0)

def save_features(features: torch.Tensor, image_ids: list[str], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)

    if features.ndim != 2:
        raise ValueError(f"Expected image features [N, D], got {tuple(features.shape)}")

    if features.shape[0] != len(image_ids):
        raise ValueError(f"Feature count does not match image ID count: {features.shape[0]} != {len(image_ids)}")

    if len(set(image_ids)) != len(image_ids):
        raise ValueError("image_ids contains duplicates")

    if not torch.isfinite(features).all():
        raise ValueError("Image features contain NaN or Inf")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(features, output_dir / "images.pt")

    name_to_idx = {
        image_id: index
        for index, image_id in enumerate(image_ids)
    }

    with (output_dir / "name_to_idx.json").open("w", encoding="utf-8") as file:
        json.dump(name_to_idx,file, indent=2)



def main():
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    device = torch.device(args.device)
    split_root = args.dataset_root / "image_splits"
    image_root = args.dataset_root / "images"

    if not split_root.is_dir():
        raise FileNotFoundError(f"FashionIQ image_splits not found: {split_root}")

    if not image_root.is_dir():
        raise FileNotFoundError(f"FashionIQ images not found: {image_root}")

    print(f"Device: {device}")
    print(f"Dataset root: {args.dataset_root}")
    image_store = DirectoryImageStore(image_root=image_root)
    model, processor = load_fgclip2(device=device)
    for split in args.splits:
        print(f"\n=== FashionIQ {split} ===")

        image_ids = load_all_image_ids(split_root=split_root, split=split,)
        print(f"Images: {len(image_ids)}")
        features = precompute(
            image_ids=image_ids,
            image_store=image_store,
            model=model,
            processor=processor,
            device=device,
            batch_size=args.batch_size,
        )
        output_dir = args.output_root / split
        save_features(features=features, image_ids=image_ids, output_dir=output_dir)

        print(f"Features: {tuple(features.shape)}")
        print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
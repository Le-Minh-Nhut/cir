from pathlib import Path
import torch
from transformers import AutoImageProcessor, AutoModelForCausalLM
from datasets.fashioniq import load_fashioniq_split_ids
import torch.nn.functional as F
from tqdm import tqdm
from datasets.common import DirectoryImageStore
import json


CATEGORIES = ("dress", "shirt", "toptee")
MODEL_NAME = "qihoo360/fg-clip2-large"

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

    for start in tqdm(range(0, len(image_ids), batch_size)):
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

def save_features(features, image_ids, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True,)

    torch.save(features, output_dir / "images.pt")

    name_to_idx = {
        image_id: index
        for index, image_id
        in enumerate(image_ids)
    }

    with (output_dir / "name_to_idx.json").open("w") as file:
        json.dump(name_to_idx, file)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = "val"
    split_root = Path("data/fashionIQ_dataset/image_splits")
    image_root = Path("data/fashionIQ_dataset/images")
    output_dir = Path("features/fashioniq/fgclip2_large/val")
    image_ids = load_all_image_ids(split_root=split_root, split=split)
    print(f"Images: {len(image_ids)}")
    image_store = DirectoryImageStore(image_root=image_root)
    model, processor = load_fgclip2(device=device)
    features = precompute(
        image_ids=image_ids,
        image_store=image_store,
        model=model,
        processor=processor,
        device=device,
        batch_size=8,
    )
    print(features.shape)
    save_features(features=features,image_ids=image_ids, output_dir=output_dir)


if __name__ == "__main__":
    main()
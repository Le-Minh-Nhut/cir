from pathlib import Path
import os
os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)
import hydra
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader

from cache.features import (
    load_feature_manifest,
    load_features,
    load_text_features,
    validate_feature_manifest,
    validate_text_cache_subdir,
)
from backbones.fgclip2 import (
    FGCLIP2_LARGE_MODEL_ID,
    FGCLIP2_LARGE_REVISION,
    validate_fgclip2_revision,
)
from datasets.common import collate_cir_samples
from datasets.fashioniq import (
    FashionIQDataset,
    load_correction_dict,
    validate_correction_policy,
)
from evaluation.fashioniq import evaluate_fashioniq
from models.taper import TAPER
from runtime import configure_torch_runtime, resolve_device, seed_everything
from training.engine import fit, prepare_batch


CATEGORIES = ("dress", "shirt", "toptee")

def load_fashioniq_correction_dicts(annotation_root: str | Path) -> dict[str, dict[str, str]]:
    annotation_root = Path(annotation_root)
    correction_dicts = {}
    for category in CATEGORIES:
        path = annotation_root / f"correction_dict_{category}.json"

        if not path.is_file():
            raise FileNotFoundError(f"Missing FashionIQ correction dictionary: {path}")

        correction_dicts[category] = load_correction_dict(path)

    return correction_dicts


def build_train_loader(annotation_root: str | Path, *, batch_size: int, num_workers: int, seed: int, caption_policy: str, correction_dicts: dict[str, dict[str, str]] | None,) -> DataLoader:
    dataset = FashionIQDataset(annotation_root=annotation_root, split="train", categories=CATEGORIES, caption_policy=caption_policy, seed=seed, correction_dicts=correction_dicts)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_cir_samples, pin_memory=True)


def build_val_loaders(annotation_root: str | Path, *, batch_size: int, num_workers: int, caption_policy: str, correction_dicts: dict[str, dict[str, str]] | None):
    val_loaders = {}
    val_annotations = {}

    for category in CATEGORIES:
        dataset = FashionIQDataset(annotation_root=annotation_root, split="val", categories=[category], caption_policy=caption_policy, correction_dicts=correction_dicts,)
        val_loaders[category] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_cir_samples, pin_memory=True)
        val_annotations[category] = dataset.annotations

    return val_loaders, val_annotations


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if str(cfg.experiment.get("name", "")) != "taper_e2e":
        raise ValueError(
            "src/train.py requires experiment=taper_e2e. "
            "Run: python src/train.py experiment=taper_e2e"
        )

    seed_everything(seed=cfg.seed, deterministic=cfg.runtime.deterministic)
    configure_torch_runtime(deterministic=cfg.runtime.deterministic, benchmark=cfg.runtime.benchmark)
    device = resolve_device(device_name=cfg.runtime.device, accelerator_index=cfg.runtime.accelerator_index)

    print("Device:", device)

    dataset_root = Path(cfg.dataset.root)
    annotation_root = dataset_root / "captions"
    correction_policy = validate_correction_policy(
        str(cfg.experiment.correction_policy)
    )
    text_cache_subdir = validate_text_cache_subdir(
        str(cfg.experiment.text_cache_subdir),
        correction_policy,
    )
    correction_dicts = (
        load_fashioniq_correction_dicts(annotation_root)
        if correction_policy == "fashioniq"
        else None
    )
    split_root = dataset_root / "image_splits"
    cache_root = Path(cfg.paths.cache_root)
    if str(cfg.experiment.backbone.model_id) != FGCLIP2_LARGE_MODEL_ID:
        raise ValueError(
            "A3.2 requires exactly backbone.model_id=qihoo360/fg-clip2-large"
        )
    expected_revision = validate_fgclip2_revision(str(cfg.experiment.backbone.revision))
    if expected_revision != FGCLIP2_LARGE_REVISION:
        raise ValueError(f"A3.2 requires revision={FGCLIP2_LARGE_REVISION}")

    feature_root = cache_root / "fashioniq" / "fgclip2-large"
    for split in ("train", "val"):
        image_manifest = load_feature_manifest(feature_root / split / "images")
        validate_feature_manifest(
            image_manifest,
            model_id=str(cfg.experiment.backbone.model_id),
            revision=expected_revision,
            cache_name=f"{split}/images",
        )
        text_manifest = load_feature_manifest(feature_root / split / text_cache_subdir)
        validate_feature_manifest(
            text_manifest,
            model_id=str(cfg.experiment.backbone.model_id),
            revision=expected_revision,
            cache_name=f"{split}/{text_cache_subdir}",
            correction_policy=correction_policy,
        )
    train_images, train_image_idx = load_features(feature_root / "train" / "images")
    val_images, val_image_idx = load_features(feature_root / "val" / "images")

    train_text = load_text_features(feature_root / "train" / text_cache_subdir)
    val_text = load_text_features(feature_root / "val" / text_cache_subdir)

    print("Train FG-CLIP2 images:", tuple(train_images.shape))
    print("Val FG-CLIP2 images:", tuple(val_images.shape))
    print("Train text:", tuple(train_text.states.shape))
    print("Val text:", tuple(val_text.states.shape))
    print("Correction policy:", correction_policy)
    print("Text cache subdirectory:", text_cache_subdir)

    train_loader = build_train_loader(
        annotation_root=annotation_root,
        batch_size=cfg.experiment.batch_size,
        num_workers=cfg.experiment.num_workers,
        seed=cfg.seed,
        caption_policy=cfg.experiment.train_caption_policy,
        correction_dicts=correction_dicts,
    )

    val_loaders, val_annotations = build_val_loaders(
        annotation_root=annotation_root,
        batch_size=cfg.experiment.eval_batch_size,
        num_workers=cfg.experiment.num_workers,
        caption_policy=cfg.experiment.val_caption_policy,
        correction_dicts=correction_dicts,
    )

    m = cfg.experiment.model

    model = TAPER(
        text_dim=m.text_dim,
        reference_dim=m.reference_dim,
        query_dim=m.query_dim,
        slot_dim=m.slot_dim,
        state_dim=m.state_dim,
        num_slots=m.num_slots,
        num_primitives=m.num_primitives,
        mask_temperature=m.mask_temperature,
        router_temperature=m.router_temperature,
        retrieval_temperature=m.retrieval_temperature,
        qasa_tau=m.qasa_tau,
        qasa_rho=m.qasa_rho,
        qasa_mu=m.qasa_mu,
        qasa_eps=m.qasa_eps,
        qasa_apply_at_eval=m.qasa_apply_at_eval,
        alpha_max=m.alpha_max,
        routing_mode=m.routing_mode,
        r4_theta=m.r4_theta,
        r4_lambda=m.r4_lambda,
        r4_capacity_enabled=m.r4_capacity_enabled,
        r4_slot_capacity=m.r4_slot_capacity,
        r4_solver_iters=m.r4_solver_iters,
    ).to(device)

    print("Routing mode:", model.routing_mode)
    if model.routing_mode == "qisca":
        print("R4 theta:", model.r4_theta)
        print("R4 lambda:", model.r4_lambda)
        print("R4 capacity enabled:", model.r4_capacity_enabled)
        if model.r4_capacity_enabled:
            print("R4 slot capacity:", model.r4_slot_capacity)
            print("R4 solver iterations:", model.r4_solver_iters)

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.experiment.lr,
        weight_decay=cfg.experiment.weight_decay,
    )

    def prepare_batch_fn(batch, batch_device):
        return prepare_batch(
            batch,
            batch_device,
            train_images,
            train_image_idx,
            train_text,
        )

    def evaluate_fn(model):
        return evaluate_fashioniq(
            model,
            val_loaders,
            val_annotations,
            protocol=cfg.protocol.name,
            split_root=split_root,
            split="val",
            image_features=val_images,
            image_name_to_idx=val_image_idx,
            text_cache=val_text,
            device=device,
        )

    fit(
        model,
        train_loader,
        optimizer,
        evaluate_fn,
        num_epochs=cfg.experiment.num_epochs,
        device=device,
        loss_weights=dict(cfg.experiment.loss_weights),
        primary_metric="mean_recall",
        output_dir=cfg.paths.output_root,
        use_amp=cfg.runtime.precision == "fp16",
        prepare_batch_fn=prepare_batch_fn,
    )


if __name__ == "__main__":
    main()

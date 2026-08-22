from pathlib import Path
import os
os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)
import hydra
import torch
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader

from cache.features import load_features, load_text_features
from datasets.common import collate_cir_samples
from datasets.fashioniq import FashionIQDataset, load_correction_dict
from evaluation.fashioniq import evaluate_fashioniq
from models.taper import TAPER
from runtime import configure_torch_runtime, resolve_device, seed_everything
from teachers.csmcir_compose import CSMCIRComposeTeacher
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


def build_train_loader(annotation_root: str | Path, *, batch_size: int, num_workers: int, seed: int, caption_policy: str, correction_dicts: dict[str, dict[str, str]],) -> DataLoader:
    dataset = FashionIQDataset(annotation_root=annotation_root, split="train", categories=CATEGORIES, caption_policy=caption_policy, seed=seed, correction_dicts=correction_dicts)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_cir_samples, pin_memory=True)


def build_val_loaders(annotation_root: str | Path, *, batch_size: int, num_workers: int, caption_policy: str, correction_dicts: dict[str, dict[str, str]]):
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
    correction_dicts = load_fashioniq_correction_dicts(annotation_root)
    split_root = dataset_root / "image_splits"
    cache_root = Path(cfg.paths.cache_root)

    train_retrieval, train_retrieval_idx = load_features(cache_root / "fashioniq" / "csmcir" / "train" / "retrieval")
    train_native, train_native_idx = load_features(cache_root / "fashioniq" / "csmcir" / "train" / "native")
    val_retrieval, val_retrieval_idx = load_features(cache_root / "fashioniq" / "csmcir" / "val" / "retrieval")
    val_native, val_native_idx = load_features(cache_root / "fashioniq" / "csmcir" / "val" / "native")

    train_text = load_text_features(cache_root / "fashioniq" / "csmcir" / "train" / "text")
    val_text = load_text_features(cache_root / "fashioniq" / "csmcir" / "val" / "text")

    print("Train retrieval:", tuple(train_retrieval.shape))
    print("Train native:", tuple(train_native.shape))
    print("Val retrieval:", tuple(val_retrieval.shape))
    print("Val native:", tuple(val_native.shape))
    print("Train text:", tuple(train_text.states.shape))
    print("Val text:", tuple(val_text.states.shape))

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

    teacher = CSMCIRComposeTeacher(
        csmcir_root=cfg.experiment.teacher.csmcir_root,
        checkpoint_path=cfg.experiment.teacher.checkpoint_path,
    ).to(device).eval()

    m = cfg.experiment.model

    model = TAPER(
        teacher,
        text_dim=m.text_dim,
        reference_dim=m.reference_dim,
        teacher_text_dim=m.teacher_text_dim,
        teacher_query_dim=m.teacher_query_dim,
        query_dim=m.query_dim,
        slot_dim=m.slot_dim,
        state_dim=m.state_dim,
        num_slots=m.num_slots,
        num_primitives=m.num_primitives,
        mask_temperature=m.mask_temperature,
        router_temperature=m.router_temperature,
        retrieval_temperature=m.retrieval_temperature,
        neutral_mode=m.neutral_mode,
        slot_gate_threshold=m.slot_gate_threshold,
        hard_slot_gating_during_training=m.hard_slot_gating_during_training,
        gate_mode=m.gate_mode,
        st_gate_recovery=m.st_gate_recovery,
        alpha_max=m.alpha_max,
        counterfactual_chunk_size=m.counterfactual_chunk_size,
        num_refine_iters=m.num_refine_iters,
        residual_strength=m.residual_strength,
        residual_depletion=m.residual_depletion,
        residual_eps=m.residual_eps,
    ).to(device)

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.experiment.lr,
        weight_decay=cfg.experiment.weight_decay,
    )

    prepare_batch_fn = lambda batch, device: prepare_batch(batch, device, train_retrieval, train_native, train_retrieval_idx, train_native_idx, train_text)

    def evaluate_fn(model):
        return evaluate_fashioniq(
            model,
            val_loaders,
            val_annotations,
            protocol=cfg.protocol.name,
            split_root=split_root,
            split="val",
            retrieval_features=val_retrieval,
            native_features=val_native,
            retrieval_name_to_idx=val_retrieval_idx,
            native_name_to_idx=val_native_idx,
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
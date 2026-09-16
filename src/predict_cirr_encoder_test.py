from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from cache.features import load_features, load_text_features
from datasets.cirr import CIRRDataset, load_cirr_image_mapping
from datasets.common import collate_cir_samples
from evaluate_cirr_encoder import load_cirr_encoder_checkpoint
from evaluation.cirr_encoder import generate_cirr_encoder_test_submissions
from runtime import configure_torch_runtime, resolve_device, seed_everything
from train_cirr_encoder import build_cirr_encoder_model


def _submission_name(cfg: DictConfig, checkpoint_path: Path) -> str:
    configured = cfg.get("submission_name")
    name = checkpoint_path.stem if configured is None else str(configured)
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("submission_name must be a non-empty filename component")
    return name


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if str(cfg.dataset.get("name", "")) != "cirr":
        raise ValueError("src/predict_cirr_encoder_test.py requires dataset=cirr")
    if str(cfg.protocol.get("name", "")) != "cirr_encoder":
        raise ValueError(
            "src/predict_cirr_encoder_test.py requires protocol=cirr_encoder"
        )
    if str(cfg.experiment.get("name", "")) != "taper_e2e":
        raise ValueError(
            "src/predict_cirr_encoder_test.py requires experiment=taper_e2e"
        )

    checkpoint_value = cfg.get("checkpoint")
    if checkpoint_value is None:
        raise ValueError("Provide +checkpoint=/path/to/best.pt")
    checkpoint_path = Path(checkpoint_value).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"TAPER checkpoint not found: {checkpoint_path}")

    seed_everything(seed=cfg.seed, deterministic=cfg.runtime.deterministic)
    configure_torch_runtime(
        deterministic=cfg.runtime.deterministic,
        benchmark=cfg.runtime.benchmark,
    )
    device = resolve_device(
        device_name=cfg.runtime.device,
        accelerator_index=cfg.runtime.accelerator_index,
    )

    dataset_root = Path(cfg.dataset.root)
    version = str(cfg.dataset.version)
    if version != "rc2":
        raise ValueError("CIRR test1 official submission requires dataset.version=rc2")
    cache_root = Path(cfg.paths.cache_root) / "cirr" / "csmcir" / "test1"
    retrieval_features, retrieval_name_to_idx = load_features(
        cache_root / "retrieval"
    )
    native_features, native_name_to_idx = load_features(cache_root / "native")
    text_cache = load_text_features(cache_root / "text")

    test_dataset = CIRRDataset(
        dataset_root / "captions", "test1", version=version
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.experiment.eval_batch_size,
        shuffle=False,
        num_workers=cfg.experiment.num_workers,
        collate_fn=collate_cir_samples,
        pin_memory=device.type == "cuda",
    )
    # Dict insertion order is the gallery order prescribed by split.rc2.test1.json.
    gallery_ids = list(
        load_cirr_image_mapping(
            dataset_root / "image_splits", "test1", version=version
        )
    )

    model = build_cirr_encoder_model(cfg, device)
    load_cirr_encoder_checkpoint(model, checkpoint_path, device)
    model.eval()
    global_submission, subset_submission = (
        generate_cirr_encoder_test_submissions(
            model,
            test_loader,
            gallery_ids=gallery_ids,
            retrieval_features=retrieval_features,
            native_features=native_features,
            retrieval_name_to_idx=retrieval_name_to_idx,
            native_name_to_idx=native_name_to_idx,
            text_cache=text_cache,
            device=device,
        )
    )

    output_value = cfg.get("submission_output_dir")
    output_dir = (
        Path(output_value)
        if output_value is not None
        else Path(cfg.paths.output_root) / "cirr" / "encoder" / "test1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    name = _submission_name(cfg, checkpoint_path)
    global_path = output_dir / f"CIRR_pred_ranks_recall_{name}.json"
    subset_path = output_dir / f"CIRR_pred_ranks_recall_subset_{name}.json"

    with global_path.open("w", encoding="utf-8") as file:
        json.dump(global_submission, file, indent=2, ensure_ascii=False)
        file.write("\n")
    with subset_path.open("w", encoding="utf-8") as file:
        json.dump(subset_submission, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"CIRR test1 global submission: {global_path.resolve()}")
    print(f"CIRR test1 subset submission: {subset_path.resolve()}")


if __name__ == "__main__":
    main()

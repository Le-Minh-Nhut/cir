import argparse
import json
from pathlib import Path
from datasets.fashioniq import VALID_CATEGORIES, compose_fashioniq_caption, load_fashioniq_annotations


CATEGORIES = ("dress", "shirt", "toptee")


def compose_single_caption(caption: str) -> str:
    return caption.strip(".?, ").capitalize()


def build_cases(
    annotation_root: str | Path, categories: tuple[str, ...]
) -> list[dict[str, object]]:
    cases = []

    for category in categories:
        annotations = load_fashioniq_annotations(annotation_root=annotation_root, split="val", category=category)

        for annotation in annotations:
            assert annotation.target_id is not None

            caption_1, caption_2 = annotation.captions
            full_text = compose_fashioniq_caption(captions=annotation.captions, policy="ordered_and")

            minus_1_text = compose_single_caption(caption_2)
            minus_2_text = compose_single_caption(caption_1)

            sample_id = f"fashioniq:val:{category}:{annotation.index}"

            cases.append(
                {
                    "sample_id": sample_id,
                    "category": category,
                    "reference_id": annotation.reference_id,
                    "target_id": annotation.target_id,
                    "caption_1": caption_1,
                    "caption_2": caption_2,
                    "full_text": full_text,
                    "minus_1_text": minus_1_text,
                    "minus_2_text": minus_2_text,
                }
            )

    return cases


def save_cases(cases: list[dict[str, object]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(cases, file, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("teacher/audit/fashioniq_val_cases.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    cases = build_cases(annotation_root=args.annotation_root, categories=CATEGORIES)
    save_cases(cases=cases, output_path=args.output)

    print(f"Cases: {len(cases)}")
    print(f"Saved: {args.output}")

    for category in CATEGORIES:
        count = sum(case["category"] == category for case in cases)

        print(f"{category}: {count}")


if __name__ == "__main__":
    main()

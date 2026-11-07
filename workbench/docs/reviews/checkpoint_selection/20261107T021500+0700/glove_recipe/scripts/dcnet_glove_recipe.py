"""Regenerate DCNet's GloVe caption pickles under the label WORKBENCH_SEEDED_RECIPE.

Recipe is `workbench/third_party/DCNet/preprocess/process_cap.py`. Three recorded deviations:

  1. OOV vectors: the author script samples `np.random.normal(0, 0.3, (300,))` with NO seed. This
     script seeds NumPy, so a rerun is reproducible. The OOV vectors therefore CANNOT equal the
     author's -- those bytes were never published -- and artifacts produced here must not be
     compared against the paper's numbers.
  2. The spaCy model is loaded from an explicit path rather than the `en_vectors_web_lg` package name.
  3. WRITE SEPARATION: the author script writes into `<dataset>/captions/`, which is the canonical
     dataset directory. This script reads canonical inputs and writes every output to `--out`, so
     preparation can never mutate the raw inputs (`DATA-02`).

Everything else follows the author code: vocabulary from `cap*train.json`, `-`/`.` stripping, NLTK
`word_tokenize`, occurrence threshold `cnt > 2`, and the output filenames.
"""
import argparse, glob, hashlib, json, os, pickle, sys
from collections import Counter

import nltk
import numpy as np
import spacy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="canonical FashionIQ root (read-only)")
    parser.add_argument("--vectors", required=True, help="path to the extracted en_vectors_web_lg dir")
    parser.add_argument("--out", required=True, help="output directory for the generated pickles")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    glove = spacy.load(args.vectors).vocab
    captions_dir = os.path.join(args.dataset_root, "captions")

    counter = Counter()
    for cap_file in sorted(glob.glob(os.path.join(captions_dir, "cap*train.json"))):
        with open(cap_file) as f:
            for data in json.load(f):
                for caption in data["captions"]:
                    counter.update(nltk.tokenize.word_tokenize(
                        caption.replace("-", " ").replace(".", "").lower()))

    vocab_words, oov = {}, []
    for word, cnt in counter.items():
        if glove.has_vector(word):
            vocab_words[word] = glove.get_vector(word)
        elif cnt > 2:
            oov.append(word)
            vocab_words[word] = np.random.normal(0, 0.3, (300,))

    # Recorded in `written` like every other output. It was previously written but never added to the
    # manifest, so a 10-file recipe reported 9 and the determinism claim could not be checked from the
    # manifest alone -- only from a separate ad-hoc hash of this one file.
    written = []
    vectors_out = os.path.join(args.out, "glove_vecs.pkl")
    with open(vectors_out, "wb") as f:
        pickle.dump(vocab_words, f)
    written.append(vectors_out)

    # The author script globs `cap*.json`, which in the canonical tree also matches the `cap*.glove.*`
    # pickles it just wrote. Globbing the JSON inputs only keeps this recipe idempotent.
    for cap_file in sorted(glob.glob(os.path.join(captions_dir, "cap*.json"))):
        with open(cap_file) as f:
            cap_data = json.load(f)
        parts = os.path.basename(cap_file).split(".")
        name = ".".join(parts[:2] + ["glove"] + parts[2:]).replace(".json", ".pkl")
        for data in cap_data:
            w2v = []
            for caption in data["captions"]:
                for word in nltk.tokenize.word_tokenize(
                        caption.replace("-", " ").replace(".", "").lower()):
                    if word in vocab_words:
                        w2v.append(vocab_words[word])
            data["wv"] = np.stack(w2v)
        path = os.path.join(args.out, name)
        with open(path, "wb") as f:
            pickle.dump(cap_data, f)
        written.append(path)

    manifest = {
        "label": "WORKBENCH_SEEDED_RECIPE",
        "seed": args.seed,
        "dataset_root": os.path.abspath(args.dataset_root),
        "vectors": os.path.abspath(args.vectors),
        "vocab_tokens": len(counter),
        "vocab_words": len(vocab_words),
        "oov_filled": len(oov),
        "outputs": [{"name": os.path.basename(p), "bytes": os.path.getsize(p),
                     "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()} for p in written],
    }
    with open(os.path.join(args.out, "recipe_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: manifest[k] for k in ("vocab_tokens", "vocab_words", "oov_filled")}))


if __name__ == "__main__":
    main()

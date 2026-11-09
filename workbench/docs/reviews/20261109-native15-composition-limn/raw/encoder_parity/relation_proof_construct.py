"""Read-only proof for ENCODER's uncheckpointed `BindingDecoder.learnable_relation`.

Writes its own log (no shell redirect: the shell clobbers `>` output on this host). Emits exactly
one `KEY value` line per claim so each number in `ENCODER_PARITY.vi.md` §4.0.1 has raw evidence.

  python relation_proof_construct.py            # full proof, writes relation_proof_construct.out
  python relation_proof_construct.py --child F  # internal: seeded construction, saves tensor to F
"""
import hashlib
import os
import subprocess
import sys

REPO = "/home/heheboiz/data/cir"
SRC = f"{REPO}/workbench/third_party/ENCODER"
CKPT = (f"{REPO}/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/"
        "encoder/fashioniq.pt")
OUT_DIR = f"{REPO}/workbench/docs/reviews/20261109-native15-composition-limn/raw/encoder_parity"
LOG_PATH = f"{OUT_DIR}/relation_proof_construct.out"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def build(seed=None):
    import torch

    sys.path.insert(0, SRC)
    os.chdir(SRC)
    import model_try2

    if seed is not None:
        torch.manual_seed(seed)
    return model_try2.Encoder(hidden_dim=512, dropout=0.5, local_token_num=3, t=0.1,
                              wc=2, N_p=2, weighted=True)


def main():
    if "--child" in sys.argv:
        import torch

        target = sys.argv[sys.argv.index("--child") + 1]
        model = build(seed=0)
        relation = model.backbone.binding_decoder.learnable_relation
        torch.save(relation.detach().cpu(), target)
        return 0

    import torch

    os.makedirs(OUT_DIR, exist_ok=True)
    log = open(LOG_PATH, "w", encoding="utf-8")
    sys.stdout = sys.stderr = Tee(sys.__stdout__, log)
    try:
        digest = hashlib.sha256()
        with open(CKPT, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        print(f"CHECKPOINT_SHA {digest.hexdigest()}")

        state = torch.load(CKPT, map_location="cpu", weights_only=True)
        keys = list(state.keys())
        print(f"CHECKPOINT_KEY_COUNT {len(keys)}")
        print(f"KEYS_MATCHING_learnable_relation "
              f"{[key for key in keys if 'learnable_relation' in key]}")
        print(f"KEYS_MATCHING_relation {[key for key in keys if 'relation' in key]}")

        first = build()
        second = build()
        one, two = first.backbone.binding_decoder, second.backbone.binding_decoder
        params = list(first.state_dict().keys())
        value = one.learnable_relation
        print(f"STATE_DICT_LEN {len(params)}")
        print(f"IN_STATE_DICT {'learnable_relation' in first.state_dict()}")
        print(f"HASATTR {hasattr(one, 'learnable_relation')}")
        print(f"TYPE {type(value)}")
        print(f"SHAPE {list(value.shape)}")
        print(f"MIN {float(value.min())}")
        print(f"MAX {float(value.max())}")
        print(f"NAMED_PARAMS_MATCHING_RELATION "
              f"{[name for name, _ in first.named_parameters() if 'relation' in name]}")
        print(f"TWO_CONSTRUCTIONS_MAX_ABS "
              f"{float((value - two.learnable_relation).abs().max())}")
        print(f"TWO_CONSTRUCTIONS_SUM_A {float(value.sum())}")
        print(f"TWO_CONSTRUCTIONS_SUM_B {float(two.learnable_relation.sum())}")

        # Seeded control: two separate PROCESSES, torch.manual_seed(0) before construction.
        saved = []
        for index in (1, 2):
            path = f"/tmp/relation_proof_seed_{index}.pt"
            subprocess.run([sys.executable, os.path.abspath(__file__), "--child", path],
                           check=True, capture_output=True, text=True,
                           env={**os.environ, "OMP_NUM_THREADS": "2"})
            saved.append(path)
        left = torch.load(saved[0]).float()
        right = torch.load(saved[1]).float()
        print(f"SEEDED_TWO_PROCESS_MAX_ABS {float((left - right).abs().max())}")
        print(f"SEEDED_EQUAL {bool(torch.equal(left, right))}")
        print(f"SEEDED_VALUE_SUM {float(left.sum())}")
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log.flush()
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

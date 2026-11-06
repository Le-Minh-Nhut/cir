"""Probe real tensor shapes at the HINT native boundary (query, target, sims).

Uses the wrapper's own module setup (`_load_replay_modules`) so the `lavis` package stub is installed
before unpickling -- the same order the real evaluator uses. Read-only: no ranking change.
"""
import sys
from pathlib import Path

CIR = Path("/home/heheboiz/data/cir")
SR = CIR / "workbench/third_party/HINT"
CKPT = Path(sys.argv[1])
sys.path.insert(0, str(CIR))

import numpy as np

from workbench.replay.hint import _load_replay_modules, evaluate_categories  # noqa: F401
from workbench.replay.common import sha256_file

REG_DECLARED = "bf9c72d42206b667ed586758c15c648970df821857e4067f26155a114a6517d4"
print("path:", CKPT, "size:", CKPT.stat().st_size)
actual = sha256_file(CKPT)
print("sha256:", actual, "matches_registry:", actual == REG_DECLARED)
print("size_matches_registry:", CKPT.stat().st_size == 2724451182)

torch, datasets, pinned, processors = _load_replay_modules(SR, "pretrain")
data_utils, test_module = pinned
txt_processors, boundary = processors
print("datasets module:", datasets.__file__)
print("test module:", test_module.__file__)
print("processor boundary:", boundary)

from workbench.replay.common import load_trusted_whole_model
model = load_trusted_whole_model(CKPT, REG_DECLARED, torch).cuda().eval()
print("root class:", type(model).__module__ + "." + type(model).__name__)
print("training flag:", model.training)
drops = [(n, m.p) for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout)]
print("dropouts:", len(drops), "nonzero:", [d for d in drops if d[1]])
print("query_tokens:", tuple(model.query_tokens.shape))
print("text_proj:", type(model.text_proj).__name__, tuple(model.text_proj.weight.shape))
print("vision_proj:", type(model.vision_proj).__name__, tuple(model.vision_proj.weight.shape))

from PIL import Image
transform = data_utils.targetpad_transform(1.25, 224)
rng = np.random.RandomState(0)
imgs = [Image.fromarray((rng.rand(300, 300, 3) * 255).astype("uint8")) for _ in range(2)]
fake = torch.stack([transform(i) for i in imgs]).float().cuda()
print("input:", tuple(fake.shape), fake.dtype)

with torch.no_grad():
    q = model.extract_retrieval_compose(fake, ["a red dress", "blue shirt"])
    t = model.extract_retrieval_target(fake)
print("query:", tuple(q.shape), q.dtype)
print("target:", tuple(t.shape), t.dtype)
qn, tn = q.data.cpu().numpy(), t.data.cpu().numpy()
print("np q:", qn.shape, "np t:", tn.shape)
sims = np.matmul(qn, tn)
print("matmul:", sims.shape, "-> squeeze:", sims.squeeze().shape, "-> max(-1):", sims.squeeze().max(-1).shape)
print("query rows after squeeze:", sims.squeeze().shape[0])

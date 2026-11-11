import sys
IMG = "/home/heheboiz/data/cir/data/FashionIQ/resized_image/dress/B005X4PL1G.jpg"
import torch, torchvision, clip
from PIL import Image
# EXACT call the replay makes (workbench/replay/tgcir.py:223): clip.load("ViT-B/16", ..., jit=False)
clip_model, preprocess = clip.load("ViT-B/16", device="cpu", jit=False)
t = preprocess(Image.open(IMG))
print("interpreter:", sys.executable, sys.version.split()[0])
print("torch:", torch.__version__, "torchvision:", torchvision.__version__)
print("transform:", "clip.load('ViT-B/16', device='cpu', jit=False).preprocess")
print("clip visual.input_resolution:", clip_model.visual.input_resolution)
print("tensor:", tuple(t.shape), str(t.dtype), "min", float(t.min()), "max", float(t.max()))
print("normalize_mean (0.48145466, 0.4578275, 0.40821073) std (0.26862954, 0.26130258, 0.27577711)")
print("expected (3,224,224) float32:", tuple(t.shape)==(3,224,224) and str(t.dtype)=="torch.float32")

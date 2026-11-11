import sys, json
from pathlib import Path
IMG = "/home/heheboiz/data/cir/data/FashionIQ/resized_image/dress/B005X4PL1G.jpg"
import torch, torchvision
from PIL import Image
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
src = Image.open(IMG)
t = transform(src)
print("interpreter:", sys.executable, sys.version.split()[0])
print("torch:", torch.__version__, "torchvision:", torchvision.__version__)
print("source_image:", IMG, src.mode, src.size)
print("tensor:", tuple(t.shape), str(t.dtype), "min", float(t.min()), "max", float(t.max()))
print("normalize_mean", [0.485, 0.456, 0.406], "normalize_std", [0.229, 0.224, 0.225])
print("resize 256 / centercrop 224 -> expected shape (3,224,224) matches:",
      tuple(t.shape) == (3,224,224) and str(t.dtype) == "torch.float32")

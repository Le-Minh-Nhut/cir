import open_clip, torch, random, numpy as np
from PIL import Image
_, ptrain, pval = open_clip.create_model_and_transforms('ViT-B-32', pretrained='workbench/third_party/ENCODER/open_clip_pytorch_model.bin')
ids=['B008ZAUL8I','245600258X','B0037KIXK4']
for i in ids:
    im=Image.open(f'data/FashionIQ/images/{i}.png').convert('RGB')
    v=pval(im.copy())
    # 5 unseeded train draws
    draws=[ptrain(im.copy()) for _ in range(5)]
    torch.manual_seed(0); a=ptrain(im.copy())
    torch.manual_seed(0); b=ptrain(im.copy())
    same_seed = torch.equal(a,b)
    d=[float((t-v).abs().mean()) for t in draws]
    dmax=[float((t-v).abs().max()) for t in draws]
    var=float(max(float((t-draws[0]).abs().max()) for t in draws[1:]))
    cos=[float(torch.nn.functional.cosine_similarity((t-v).flatten(),(draws[0]-v).flatten(),dim=0)) for t in draws[1:2]]
    print(f'{i} size={im.size} mean|train_val|={np.mean(d):.4f} max|train_val|={np.max(dmax):.2f} '
          f'train_draw_variability={var:.3f} same_seed_equal={same_seed}')

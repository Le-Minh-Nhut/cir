import open_clip, torch, inspect
print('open_clip file:', open_clip.__file__)
_, ptrain, pval = open_clip.create_model_and_transforms('ViT-B-32', pretrained='workbench/third_party/ENCODER/open_clip_pytorch_model.bin')
print('--- preprocess_train ---'); print(ptrain)
print('--- preprocess_val ---'); print(pval)
print('same object?', ptrain is pval)
print('train has RandomResizedCrop:', any('RandomResizedCrop' in type(t).__name__ for t in getattr(ptrain,'transforms',[])))
print('val has RandomResizedCrop:', any('RandomResizedCrop' in type(t).__name__ for t in getattr(pval,'transforms',[])))

import open_clip, torch
tok=open_clip.get_tokenizer('ViT-B-32')
for s in ['is long and has a floral print','', '   ']:
    t=tok([s])
    ids=t[0].tolist()
    eot=int(torch.argmax((t[0]==49407).int())) if (t[0]==49407).any() else None
    print(repr(s),'shape',tuple(t.shape),'ids[:8]',ids[:8],'pad_count',int((t[0]==0).sum()),'eot_idx',eot)

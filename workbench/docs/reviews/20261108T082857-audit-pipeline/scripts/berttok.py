from transformers import BertTokenizer
import os
p=os.path.expanduser('~/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594')
tok=BertTokenizer.from_pretrained(p, truncation_side="right")
tok.add_special_tokens({"bos_token":"[DEC]"})
print('bos id',tok.bos_token_id,'eos',tok.eos_token_id,'pad',tok.pad_token_id,'cls',tok.cls_token_id,'sep',tok.sep_token_id)
for s in ['is long and has a floral print','', 'x']:
    enc=tok(s,padding='max_length',truncation=True,max_length=32,return_tensors='pt')
    ids=enc['input_ids'][0].tolist()
    nz=[i for i,v in enumerate(ids) if v!=tok.pad_token_id]
    print(repr(s),'-> nonpad',len(nz),'ids',ids[:8],'mask_sum',int(enc['attention_mask'].sum()))
    print('    decoded',repr(tok.decode([ids[i] for i in nz])))

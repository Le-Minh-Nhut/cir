from locale import normalize
from multiprocessing import reduction
from sys import prefix
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open_clip
from torch.autograd import Variable
import Transformer
import math
import os


class Backbone(nn.Module):
    def __init__(self, img_encoder = 'ViT-L/14', hidden_dim=768, dropout=0.0, global_num=8, prefix_num=8):
        super().__init__()
        self.clip, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=os.path.join('[CLIP path]/laionCLIP-ViT-L-14-DataComp.XL-s13B-b90K', 'open_clip_pytorch_model.bin'))
        self.clip = self.clip.float()
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.image_backbone = self.clip.visual

        self.hidden_dim=hidden_dim  # 768
        self.prefix_num=prefix_num

        self.img_fc = nn.Linear(1024, 768)
        self.text_fc = nn.Linear(768, 768)

        self.global_token_num = global_num

    def extract_img_fea(self, x):

        if self.image_backbone.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(x.shape[0], x.shape[1], self.image_backbone.grid_size[0], self.image_backbone.patch_size[0], self.image_backbone.grid_size[1], self.image_backbone.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.image_backbone.grid_size[0] * self.image_backbone.grid_size[1], -1)
            x = self.image_backbone.patchnorm_pre_ln(x)
            x = self.image_backbone.conv1(x)
        else:
            x = self.image_backbone.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [self.image_backbone.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.image_backbone.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.image_backbone.patch_dropout(x)
        x = self.image_backbone.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.image_backbone.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.image_backbone.attn_pool is not None:
            x = self.image_backbone.attn_pool(x)
            x = self.image_backbone.ln_post(x)
            pooled, tokens = self.image_backbone._global_pool(x)
        else:
            pooled, tokens = self.image_backbone._global_pool(x)
            pooled = self.image_backbone.ln_post(pooled)

        if self.image_backbone.proj is not None:
            pooled = pooled @ self.image_backbone.proj

        return self.img_fc(tokens), pooled
        
    def extract_text_fea(self, txt):

        text = self.tokenizer(txt).cuda()
        cast_dtype = self.clip.transformer.get_cast_dtype()

        x = self.clip.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.clip.ln_final(x)
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
        pooled = pooled @ self.clip.text_projection

        return self.text_fc(tokens), pooled


    
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.embedding_dim = embedding_dim

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, num_tokens, embedding_dim]

        x = x * math.sqrt(self.embedding_dim)
        seq_len = x.size(1)

        # 添加位置编码
        x = x + self.pe[:, :seq_len]

        return x

class LIMN(nn.Module):
    def __init__(self, prefix_num=8, head=8, layer=2, hidden_dim=768, temperature=3.0, dropout=0.3):
        super().__init__()
        self.backbone = Backbone(img_encoder = 'ViT-L/14', hidden_dim=hidden_dim, dropout=0.0, prefix_num=prefix_num)
        self.transformer_query = Transformer.Transformer(dim_self=hidden_dim, num_heads=head, dim_ref=hidden_dim, num_layers=layer) 
        self.prefix_query = nn.Parameter(torch.zeros(1, prefix_num + 1, hidden_dim))
        self.prefix_target = nn.Parameter(torch.zeros(1, prefix_num + 1, hidden_dim))
        
        self.loss_weight = torch.nn.Parameter(torch.FloatTensor((10.,)))

        self.prefix_num = prefix_num
        self.hiddem_dim = hidden_dim

        self.temperature = torch.nn.Parameter(torch.FloatTensor((temperature, temperature)))
        self.dataset_type_embedding = nn.Embedding(20, hidden_dim)

        self.query_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
            )
    
        self.image_positional_embedding = PositionalEncoding(hidden_dim)
        self.text_positional_embedding = PositionalEncoding(hidden_dim)
        self.modality_embeddimg = nn.Embedding(3, hidden_dim)

        self.masks_query = torch.nn.Embedding(prefix_num, hidden_dim)
        self.masks_target = torch.nn.Embedding(prefix_num, hidden_dim)

        self.local_weight = nn.Parameter(torch.tensor([1.0 for _ in range(prefix_num)]))

    
    def compose_query(self, img, text, dataset_type):
        img_local_fea, img_global_fea = self.backbone.extract_img_fea(img)
        text_local_fea, text_global_fea = self.backbone.extract_text_fea(text)
        batch_size, token_num, dim = img_local_fea.size()
        
        fuse_mask = self.query_fuse(torch.cat([img_global_fea, text_global_fea], dim=-1))
        fuse_fea = img_global_fea * fuse_mask + text_global_fea * (1 - fuse_mask)# + self.residual_token[:,0,:]

        global_tokens = []
        for idx in range(self.prefix_num):
            concept_idx = np.zeros((fuse_fea.shape[0],), dtype=int)
            concept_idx += idx
            concept_idx = torch.from_numpy(concept_idx)
            concept_idx = concept_idx.cuda()
            concept_idx = Variable(concept_idx)
            txt_mask = self.masks_query(concept_idx)
            txt_mask = torch.nn.functional.relu(txt_mask)
            masked_embedding = fuse_fea * txt_mask # batch_size, dim
            global_tokens.append(masked_embedding)
        global_tokens = torch.stack(global_tokens).permute(1,0,2).contiguous()

        initial_prefix = torch.cat([(fuse_fea + self.prefix_query[:,0,:]).unsqueeze(1), (global_tokens + self.prefix_query[:,1:,:])], dim=1)
        dataset_token = self.dataset_type_embedding(dataset_type.long().unsqueeze(1))
        
        img_local_fea = self.image_positional_embedding(img_local_fea)
        text_local_fea = self.text_positional_embedding(text_local_fea)
        img_local_fea = img_local_fea + self.modality_embeddimg(torch.zeros(batch_size,1).long().to(img_local_fea.device))
        text_local_fea = text_local_fea + self.modality_embeddimg(torch.ones(batch_size,1).long().to(img_local_fea.device))

        cat_feature = torch.cat([initial_prefix, dataset_token, img_local_fea, text_local_fea], dim=1)
        batch_size, n, dim = cat_feature.size()
        ones = torch.ones((n, n)).to(cat_feature.device)
        ones[0, 0] = 0
        ones[0, self.prefix_num + 1:] = 0
        ones[1:, 0] = 0
        mask = ones.bool()
        output_, attentions_ = self.transformer_query.forward_with_attention(cat_feature, mask=mask)
        output_prefix = output_[:,:self.prefix_num + 1,:]
        global_fea = output_prefix[:, 0, :]
        peer_fea = output_prefix[:, 1:, :]

        norm_global_fea = F.normalize(global_fea, p=2, dim=-1)
        norm_peer_fea = F.normalize(peer_fea, p=2, dim=-1)
        return norm_global_fea, norm_peer_fea

    def compose_target(self, img, dataset_type):
        img_local_fea, img_global_fea = self.backbone.extract_img_fea(img)
        batch_size, token_num, dim = img_local_fea.size()
        dataset_token = self.dataset_type_embedding(dataset_type.long().unsqueeze(1))

        global_tokens = []
        for idx in range(self.prefix_num):
            concept_idx = np.zeros((img_global_fea.shape[0],), dtype=int)
            concept_idx += idx
            concept_idx = torch.from_numpy(concept_idx)
            concept_idx = concept_idx.cuda()
            concept_idx = Variable(concept_idx)
            txt_mask = self.masks_target(concept_idx)
            txt_mask = torch.nn.functional.relu(txt_mask)
            masked_embedding = img_global_fea * txt_mask # batch_size, dim
            global_tokens.append(masked_embedding)
        global_tokens = torch.stack(global_tokens).permute(1,0,2).contiguous()

        initial_prefix = torch.cat([(img_global_fea + self.prefix_target[:,0,:]).unsqueeze(1), (global_tokens + self.prefix_target[:,1:,:])], dim=1)

        img_local_fea = self.image_positional_embedding(img_local_fea)
        img_local_fea = img_local_fea + self.modality_embeddimg(torch.zeros(batch_size,1).long().to(img_local_fea.device))
        
        cat_feature = torch.cat([initial_prefix, dataset_token, img_local_fea], dim=1)
        batch_size, n, dim = cat_feature.size()
        ones = torch.ones((n, n)).to(cat_feature.device)
        ones[0, 0] = 0
        ones[0, self.prefix_num + 1:] = 0
        ones[1:, 0] = 0
        mask = ones.bool()
        output_, attentions_ = self.transformer_query.forward_with_attention(cat_feature, mask=mask)
        output_prefix = output_[:,:self.prefix_num + 1,:]
        global_fea = output_prefix[:, 0, :]
        peer_fea = output_prefix[:, 1:, :]

        norm_global_fea = F.normalize(global_fea, p=2, dim=-1)
        norm_peer_fea = F.normalize(peer_fea, p=2, dim=-1)
        return norm_global_fea, norm_peer_fea

    @torch.no_grad()
    def many2many_match(self, query, target, n):
        # batch_size, n, dim
        b, k, d = query.size()
        similarity_matrix = torch.matmul(query, target.transpose(1,2)) # batch_size, n, n
        mask = torch.ones_like(similarity_matrix).to(similarity_matrix.device)

        selected_query = torch.zeros((b, n, d)).to(similarity_matrix.device)
        selected_target = torch.zeros((b, n, d)).to(similarity_matrix.device)

        for _ in range(n):
            masked_matrix = similarity_matrix * mask
            max_val, max_idx = masked_matrix.view(b,-1).max(-1)
            row_idx, col_idx = max_idx // k, max_idx % k  # b  b
            selected_query[:,_,:] = query[torch.arange(b),row_idx]
            selected_target[:,_,:] = target[torch.arange(b),col_idx]
            mask[:, row_idx, :] = 0
            mask[:, :, col_idx] = 0
        
        return selected_query.flatten(1), selected_target.flatten(1)

    @torch.no_grad()
    def derive_fn_mask(self, ref, mod, tag, dataset_type, pre_model, p=0.8): #filter based on pre model
        pre_model.eval()
        pre_query = pre_model.compose_query(ref, mod, dataset_type)[0]
        pre_target = pre_model.compose_target(tag, dataset_type)[0]
        sim_matrix = torch.mm(pre_query, pre_target.transpose(0, 1)) # batch_size, batch_size
        N = sim_matrix.shape[0]
        diagonal = torch.diag(sim_matrix)
        diagonal = diagonal.unsqueeze(1).expand(N, N)
        mask = torch.zeros_like(sim_matrix).to(sim_matrix.device)

        rand_matrix = torch.randn(N,N).to(mask.device)
        mask[(sim_matrix > diagonal) & (rand_matrix < p)] = 1
        return mask.bool()

    def compute_loss(self, ref, mod, tag, dataset_type, pre_model, p_origin, p_aug):
        compose_global_fea, compose_peer_fea = self.compose_query(ref, mod, dataset_type)
        target_global_fea, target_peer_fea = self.compose_target(tag, dataset_type)

        selected_compose_peer = (F.normalize(compose_peer_fea, p=2, dim=-1) * self.local_weight.unsqueeze(0).unsqueeze(-1)).flatten(1)
        selected_target_peer = (F.normalize(target_peer_fea, p=2, dim=-1) * self.local_weight.unsqueeze(0).unsqueeze(-1)).flatten(1)

        if dataset_type[0] <= 6:
            bool_ind = self.derive_fn_mask(ref, mod, tag, dataset_type, pre_model, p_origin)
        else:
            bool_ind = self.derive_fn_mask(ref, mod, tag, dataset_type, pre_model, p_aug)
    
        loss = {}
        loss['leader_ranking'] = self.ranking_nce_loss(compose_global_fea, target_global_fea, bool_ind, norm=False)
        loss['peer_ranking'] = self.ranking_nce_loss(selected_compose_peer, selected_target_peer, bool_ind, norm=False)
        loss['norm'] = (torch.norm(self.masks_query.weight, p=1) + torch.norm(self.masks_target.weight, p=1)) / (2.0 * self.prefix_num * self.hiddem_dim)
                                                                                                     
        return loss


    def ranking_nce_loss(self, mod_img1, img2, bool_ind, norm=True, negtive=None):
        if norm:
            mod_img1 = F.normalize(mod_img1, p=2, dim=-1)
            img2 = F.normalize(img2, p=2, dim=-1)
        x = torch.mm(mod_img1, img2.transpose(0, 1))
        x[bool_ind] = -1
        if negtive is not None:
            negtive = F.normalize(negtive)
            y = torch.mm(mod_img1, negtive.transpose(0, 1)).diag().unsqueeze(-1)
            x = torch.cat([x, y], dim=-1)
        
        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()

        loss = F.cross_entropy(self.loss_weight * x, labels)
        return loss


    

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip
from torch.distributions import Categorical
from torch.autograd import Variable

class SpatialAttention(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=dim, out_channels=1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, L, D = x.shape # (B,49,1024)  text: (B,L,1024)
        x = x.permute(0,2,1) # (B,1024,49)
        weight_map = self.conv(x) # (B,1,L)
        out = torch.mean(x * weight_map, dim=-1) # (b,1,49) * (b,1024,49) => (b,1024,49) =>(b,1024)
        return out, weight_map

class TokenLearner(nn.Module):
    def __init__(self, S, dim):
        super().__init__()
        self.S = S
        self.tokenizers = nn.ModuleList([SpatialAttention(dim) for _ in range(S)])
    
    def forward(self, x):
        B, L, C = x.shape
        Z = torch.Tensor(B, self.S, C).cuda() # (B,8,1024)
        for i in range(self.S):
            Ai,_ = self.tokenizers[i](x)
            Z[:, i, :] = Ai
        return Z

class Backbone(nn.Module):
    def __init__(self, img_encoder = 'RN50', hidden_dim=1024, dropout=0.0, local_token_num=8, global_token_num=8):
        super().__init__()
        self.clip, _ = clip.load(img_encoder, device='cuda', jit=False)
        self.clip = self.clip.float()
        self.image_backbone = self.clip.visual
        self.img_encoder = img_encoder
        self.tokenlearn = TokenLearner(S=local_token_num, dim=hidden_dim)
        self.hidden_dim = hidden_dim
        if img_encoder == 'RN50':
            self.fc = nn.Linear(2048, 1024)
            self.text_fc = nn.Linear(512, 1024)
        elif img_encoder == 'ViT-B/16':
            self.fc = nn.Linear(768,512)
            self.text_fc = nn.Linear(512,512)

        self.masks = torch.nn.Embedding(global_token_num, hidden_dim)
        mask_array = np.zeros([global_token_num, hidden_dim])
        mask_array.fill(0.1)
        mask_len = int(hidden_dim / global_token_num)
        for i in range(global_token_num):
            mask_array[i, i*mask_len:(i+1)*mask_len] = 1
        self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=True)

        self.local_token_num = local_token_num
        self.global_token_num = global_token_num

    def extract_img_fea(self, x):
        def stem(x):
            x = self.image_backbone.relu1(self.image_backbone.bn1(self.image_backbone.conv1(x)))
            x = self.image_backbone.relu2(self.image_backbone.bn2(self.image_backbone.conv2(x)))
            x = self.image_backbone.relu3(self.image_backbone.bn3(self.image_backbone.conv3(x)))
            x = self.image_backbone.avgpool(x)
            return x

        if self.img_encoder == 'RN50':
            x = x.type(self.image_backbone.conv1.weight.dtype)
            x = stem(x)
            x = self.image_backbone.layer1(x)
            x = self.image_backbone.layer2(x)
            x = self.image_backbone.layer3(x)
            local_fea = self.image_backbone.layer4(x)
            global_fea = self.image_backbone.attnpool(local_fea)
            B,D,H,W = local_fea.shape
            local_fea = self.fc(local_fea.float().detach().flatten(2).permute(0,2,1)) # (B,D,7,7) => (B,D,49) =>(B,49,1024)
            # local_fea = self.fc(local_fea.float().flatten(2).permute(0,2,1))
            tokens, weight_map = self.tokenlearn(local_fea.view(B,-1,self.hidden_dim)) # (B, 49, 1024)
            return tokens.float(), global_fea.float(), weight_map.float()
        
        elif self.img_encoder == 'ViT-B/16':
            x = self.image_backbone.conv1(x) # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1) # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1) # shape = [*, grid ** 2, width]
            x = torch.cat([self.image_backbone.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) # shape = [*, grid ** 2 + 1, width]
            x = x + self.image_backbone.positional_embedding.to(x.dtype)
            x = self.image_backbone.ln_pre(x)

            x = x.permute(1, 0, 2) # NLD -> LND
            x = self.image_backbone.transformer(x)
            x = x.permute(1, 0, 2) # LND -> NLD

            global_fea = self.image_backbone.ln_post(x[:, 0, :]) @ self.image_backbone.proj

            #mask_norm = None 
            global_tokens = []
            for idx in range(self.global_token_num):
                concept_idx = np.zeros((len(x),), dtype=int)
                concept_idx += idx
                concept_idx = torch.from_numpy(concept_idx)
                concept_idx = concept_idx.cuda()
                concept_idx = Variable(concept_idx)
                self.mask = self.masks(concept_idx)
                self.mask = torch.nn.functional.relu(self.mask)
                masked_embedding = global_fea * self.mask # batch_size, dim
                global_tokens.append(masked_embedding)
            global_tokens = torch.stack(global_tokens).permute(1,0,2).contiguous()

            local_fea = self.fc(x.float())
            local_tokens = self.tokenlearn(self.fc(x.float()))
            return torch.cat([global_tokens, local_tokens], dim=1)
        
    def extract_text_fea(self, txt):
        text = clip.tokenize(txt).cuda()

        x = self.clip.token_embedding(text).type(self.clip.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip.positional_embedding.type(self.clip.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x).type(self.clip.dtype)
        global_fea = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip.text_projection

        global_tokens = []
        for idx in range(self.global_token_num):
            concept_idx = np.zeros((len(x),), dtype=int)
            concept_idx += idx
            concept_idx = torch.from_numpy(concept_idx)
            concept_idx = concept_idx.cuda()
            concept_idx = Variable(concept_idx)
            self.mask = self.masks(concept_idx)
            self.mask = torch.nn.functional.relu(self.mask)
            masked_embedding = global_fea * self.mask # batch_size, dim
            global_tokens.append(masked_embedding)
        global_tokens = torch.stack(global_tokens).permute(1,0,2).contiguous()
        local_tokens = self.tokenlearn(self.text_fc(x.float()))

        return torch.cat([global_tokens, local_tokens], dim=1)

class VCG_CIR(nn.Module):
    def __init__(self, img_encoder = 'RN50', hidden_dim=1024, dropout=0.0, local_token_num=8, global_token_num=8, t=10):
        super().__init__()
        self.backbone = Backbone(img_encoder, hidden_dim, dropout, local_token_num, global_token_num)
        self.loss_T = nn.Parameter(torch.tensor([10.]))
        self.local_weight = nn.Parameter(torch.tensor([1.0 for _ in range(local_token_num + global_token_num)]))

        self.s_remain_map = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.t_remain_map = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.t_replace_map = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # self.vib = VIB(hidden_dim, hidden_dim // 2)

        self.t = t

    def t_compose_feature(self, ref, mod, tag):
        ref_token = self.backbone.extract_img_fea(ref)
        mod_token = self.backbone.extract_text_fea(mod)
        tag_token = self.backbone.extract_img_fea(tag)
        ref_token = ref_token.detach()
        mod_token = mod_token.detach()

        remain_mask = self.t_remain_map(torch.cat([ref_token, tag_token], dim=-1))
        replace_mask = self.t_replace_map(torch.cat([mod_token, tag_token], dim=-1))
        
        t_fuse_local = remain_mask * ref_token + replace_mask * mod_token
        return t_fuse_local, remain_mask, replace_mask, tag_token
    
    def s_compose_feature(self, ref, mod):
        ref_token = self.backbone.extract_img_fea(ref)
        mod_token = self.backbone.extract_text_fea(mod)

        remain_mask = self.s_remain_map(torch.cat([ref_token, mod_token], dim=-1))
        #replace_mask = self.s_replace_map(torch.cat([ref_token, mod_token], dim=-1))
        replace_mask = 1-remain_mask
        
        s_fuse_local = remain_mask * ref_token + replace_mask * mod_token
        return s_fuse_local, remain_mask, replace_mask, ref_token, mod_token
    
    def extract_retrieval_compose(self, ref, mod):
        s_fuse_local, _, _, _, _ = self.s_compose_feature(ref, mod)
        s_fuse_local = F.normalize(torch.mean(s_fuse_local, dim=1), p=2, dim=-1)

        return s_fuse_local

    def extract_retrieval_target(self, tag):
        tag_local = self.backbone.extract_img_fea(tag)
        tag_local = F.normalize(torch.mean(tag_local, dim=1), p=2, dim=-1)
        return tag_local
    
    def compute_loss(self, ref, mod, tag):
        s_fuse_local, s_remain_mask, s_replace_mask, ref_local, txt_local = self.s_compose_feature(ref, mod)
        t_fuse_local, t_remain_mask, t_replace_mask, tag_local = self.t_compose_feature(ref, mod, tag)

        t_compose_feature = (F.normalize(t_fuse_local, p=2, dim=-1) * self.local_weight.unsqueeze(0).unsqueeze(-1)).flatten(1)
        tag_feature = (F.normalize(tag_local, p=2, dim=-1) * self.local_weight.unsqueeze(0).unsqueeze(-1)).flatten(1)

        s_retrieval_query = F.normalize(torch.mean(s_fuse_local, dim=1), p=2, dim=-1)
        s_retrieval_target = F.normalize(torch.mean(tag_local, dim=1), p=2, dim=-1)
        loss = {}
        loss['stu_rank'] = self.info_nce(s_retrieval_query, s_retrieval_target)
        loss['tea_rank'] = self.info_nce(t_compose_feature, tag_feature)
        loss['tea_mask'] = self.mask_constraint(t_remain_mask, t_replace_mask)
        loss['ortho'] = (self.orthogonal_regularization(ref_local) + self.orthogonal_regularization(tag_local) + self.orthogonal_regularization(txt_local) ) / 3.0
        loss['ckd'] = F.mse_loss(s_remain_mask, t_remain_mask.detach()) + F.mse_loss(s_replace_mask, t_replace_mask.detach())
        loss['kl'] = self.kl_div(s_retrieval_query, s_retrieval_target, tag_feature, tag_feature, self.t)



        loss['s_info_nce'] = self.info_nce(s_retrieval_query, s_retrieval_target)
        loss['ortho_loss'] = (self.orthogonal_regularization(ref_local) + self.orthogonal_regularization(tag_local) + self.orthogonal_regularization(txt_local) ) / 3.0

        loss['t_info_nce'] = self.info_nce(t_compose_feature, tag_feature)
        loss['remain_replace_sum'] = self.mask_constraint(t_remain_mask, t_replace_mask)
        
        loss['distill_mask'] = F.mse_loss(s_remain_mask, t_remain_mask.detach()) + F.mse_loss(s_replace_mask, t_replace_mask.detach())
        loss['distill_kl'] = self.kl_div(s_retrieval_query, s_retrieval_target, tag_feature, tag_feature, self.t)
        return loss

    
    def mask_constraint(self, mask1, mask2):
        mask = mask1 + mask2
        y = torch.ones_like(mask).float().cuda()
        return F.mse_loss(mask,y)

    def info_nce(self, query, target):
        # query = F.normalize(query, p=2, dim=-1)
        # target = F.normalize(query, p=2, dim=-1)
        x = torch.mm(query, target.T)
        labels = torch.arange(query.shape[0]).long().cuda()
        return F.cross_entropy(x * self.loss_T, labels)

    
    def kl_div(self, x1, y1, x2, y2, t):
        x1 = F.normalize(x1, p=2, dim=-1)
        y1 = F.normalize(y1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        y2 = F.normalize(y2, p=2, dim=-1)

        x1_y1 = torch.mm(x1, y1.T) / t
        x2_y2 = torch.mm(x2, y2.T) / t

        log_soft_x1 = F.log_softmax(x1_y1, dim=1)
        soft_x2 = F.softmax(torch.autograd.Variable(x2_y2), dim=1)
        kl = F.kl_div(log_soft_x1, soft_x2, reduction='batchmean')

        return kl

    def orthogonal_regularization(self, templates):
        # batch_size, length, dim
        batch_size, length, dim = templates.size()
        device = templates.device
        norm_templates = F.normalize(templates, p=2, dim=-1)
        # (B,L,D) * (B,D,L)
        cosine_score = torch.matmul(norm_templates, norm_templates.permute(0,2,1).contiguous()) # batch_size, length, length 
        eye_matrix = torch.eye(length).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        l2_loss = torch.nn.MSELoss()
        return l2_loss(cosine_score, eye_matrix)


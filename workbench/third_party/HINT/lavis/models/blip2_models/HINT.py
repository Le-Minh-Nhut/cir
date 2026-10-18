import numpy as np
from torch.backends import cudnn
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from lavis.common.registry import registry
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip2_models.crossmodal_context_transformer import CrossModalContextTransformerEncoderLayer
from lavis.models.blip2_models.compute_loss import ComputeFinalLoss


def l2norm(X, dim=-1):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def l1norm(X, dim):
    norm = torch.abs(X).sum(dim=dim, keepdim=True)
    X = torch.div(X, norm)
    return


class VisualSA(nn.Module):
    def __init__(self, embed_dim, dropout_rate, num_region):
        super(VisualSA, self).__init__()
        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.BatchNorm1d(num_region),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.BatchNorm1d(embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))
        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)
        return new_global


class QCRModule(nn.Module):
    def __init__(self, embed_size, v_SA_dropout, nhead, dim_feedforward, v_dropout, vl_dropout, v_transformer_layer=6,
                 vl_transformer_layer=6):
        super().__init__()
        self.num_region = 32
        self.v_global_w = VisualSA(embed_size, v_SA_dropout, self.num_region)
        self.vl_transformer = nn.ModuleList([CrossModalContextTransformerEncoderLayer(embed_size, nhead,
                                                                                      dim_feedforward,
                                                                                      vl_dropout) for i in
                                             range(vl_transformer_layer)])
        self.distill_loss = DistillLoss(embed_size)
        self.init_weights()

    def compute_pairwise_similarity(self, src_feats, tgt_feats):
        sim = torch.matmul(tgt_feats, src_feats.transpose(1, 2))
        sim = nn.LeakyReLU(0.1)(sim)
        return sim

    def pairwise_similarity_to_attn(self, pairwise_similarities):
        attn = pairwise_similarities.clamp(min=-1e10)
        attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]
        attn = F.softmax(attn, dim=-1)
        return attn

    def compute_pairwise_cosine_similarity(self, src_feats, tgt_feats, eps=1e-8):
        src_norm = src_feats / (src_feats.norm(dim=-1, keepdim=True) + eps)
        tgt_norm = tgt_feats / (tgt_feats.norm(dim=-1, keepdim=True) + eps)
        sim = torch.matmul(tgt_norm, src_norm.transpose(1, 2))
        return sim

    def forward(self, img_emb, cap_emb, indices, t_mask, global_feats, current_test_turns):
        bsize, n_regions, embed_size = img_emb.size()
        if self.training:
            cap_emb_select = torch.index_select(cap_emb, 1, indices)
            current_turns = cap_emb_select.size(1)
        else:
            cap_emb_select = cap_emb[:, :current_test_turns, :]
            current_turns = cap_emb_select.size(1)
        cls_emb = global_feats.unsqueeze(1)
        src_emb = torch.cat([cls_emb, img_emb], dim=1)
        for module in self.vl_transformer:
            src_emb = module(src_emb, need_weights=False)
        img_emb = src_emb[:, 1:, :]
        region_feats = img_emb.view(1, bsize, n_regions, embed_size)
        region_feats = region_feats.expand(bsize, bsize, n_regions, embed_size).contiguous()
        region_feats = region_feats.view(bsize, bsize * n_regions, embed_size).contiguous()
        sim_local_1 = torch.zeros(bsize, bsize, current_turns).cuda()
        for i in range(current_turns):
            cap_emb_i = cap_emb_select[:, :i + 1, :]
            sim_region = self.compute_pairwise_similarity(cap_emb_i, region_feats)
            attn_region = self.pairwise_similarity_to_attn(sim_region)
            sim_CurrentRound_local_1 = torch.sum(sim_region * attn_region, dim=-1)
            sim_CurrentRound_local_1 = sim_CurrentRound_local_1.view(bsize, bsize, n_regions)
            sim_CurrentRound_local_1 = torch.mean(sim_CurrentRound_local_1, -1)
            sim_local_1[:, :, i] = sim_CurrentRound_local_1
        vl_src = torch.cat([img_emb, cap_emb], dim=1)
        t_mask = t_mask.to(torch.bool)
        v_mask = torch.zeros((bsize, img_emb.size(1))).cuda().to(torch.bool)
        vl_mask = torch.cat([v_mask, t_mask], dim=1)
        for module in self.vl_transformer:
            vl_src = module(vl_src, vl_mask, need_weights=False)
        v_src = vl_src[:, 0:(self.num_region), :]
        region_feats_2 = v_src.view(1, bsize, n_regions, embed_size)
        region_feats_2 = region_feats_2.expand(bsize, bsize, n_regions, embed_size).contiguous()
        region_feats_2 = region_feats_2.view(bsize, bsize * n_regions, embed_size).contiguous()
        sim_local_2 = torch.zeros(bsize, bsize, current_turns).cuda()
        for i in range(current_turns):
            cap_emb_i = cap_emb_select[:, :i + 1, :]
            sim_region_2 = self.compute_pairwise_similarity(cap_emb_i, region_feats_2)
            attn_region_2 = self.pairwise_similarity_to_attn(sim_region_2)
            sim_CurrentRound_local_2 = torch.sum(sim_region_2 * attn_region_2, dim=-1)
            sim_CurrentRound_local_2 = sim_CurrentRound_local_2.view(bsize, bsize, n_regions)
            sim_CurrentRound_local_2 = torch.mean(sim_CurrentRound_local_2, -1)
            sim_local_2[:, :, i] = sim_CurrentRound_local_2
        return sim_local_1, sim_local_2

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DistillLoss(nn.Module):
    def __init__(self, embed_size):
        super(DistillLoss, self).__init__()
        self.proj = nn.Linear(256, embed_size)

    def forward(self, cls_emb, global_features_proj):
        loss_dist = 0
        if torch.is_tensor(global_features_proj) == True:
            global_features_proj = global_features_proj.cuda()
            loss_dist = F.l1_loss(cls_emb, global_features_proj)
        else:
            pass

        return loss_dist


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        if x.dim() == 2:
            for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
                x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        elif x.dim() == 3:
            B, N, D = x.size()
            x = x.reshape(B * N, D)
            for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
                x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
            x = x.view(B, N, self.output_dim)
        return x


class EncoderImage(nn.Module):
    def __init__(self, img_dim, embed_size):
        super(EncoderImage, self).__init__()

    def forward(self, images):
        return images


class QueryTokenSampler(nn.Module):
    def __init__(self, embed_size, sample_option, dropped_ratio):
        super(QueryTokenSampler, self).__init__()
        self.embed_size = embed_size
        self.sample_option = sample_option
        self.dropped_ratio = dropped_ratio

    def forward(self, captions):
        bsize, max_turns, embed_size = captions.size()
        if self.training:
            if self.sample_option:
                num_query = captions.shape[1]
                rand_list = np.random.rand(num_query)
                ind = np.where(rand_list > self.dropped_ratio)[0]
                indices = torch.tensor(ind).cuda()
                t_mask = torch.ones(max_turns)
                if indices.numel() == 0:
                    id = np.random.permutation(range(max_turns))[0]
                    t_mask[id] = 0
                    indices = torch.tensor(id).cuda()
                else:
                    for i in range(len(ind)):
                        select_ind = ind[i]
                        t_mask[select_ind] = 0
                new_t_mask = t_mask.repeat(bsize, 1).cuda()
            else:
                t_mask = torch.zeros(max_turns)
                new_t_mask = t_mask.repeat(bsize, 1).cuda()
                indices = torch.arange(0, max_turns, 1).cuda()
        else:
            indices = None
            new_t_mask = None

        return captions, indices, new_t_mask


@registry.register_model("HINT")
class HINT(Blip2Base):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            num_query_token=32,
            cross_attention_freq=2,
            embed_dim=256,
            max_txt_len=32,
            img_dim=256,
            grad_clip=2.,
            loss_weight=0.2,
            v_transformer_layer=1,
            vl_transformer_layer=1,
            sampled_option=True,
            dropped_ratio=0.7,
            v_SA_dropout=0.4,
            nhead=1,
            dim_feedforward=512,
            v_dropout=0.1,
            vl_dropout=0.1,
            tau=15
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.max_txt_len = max_txt_len
        self.prompt_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)

        self.grad_clip = grad_clip
        self.loss_weight = loss_weight
        self.img_enc = EncoderImage(img_dim, embed_dim)
        self.txt_enc = QueryTokenSampler(embed_dim, sampled_option, dropped_ratio)
        self.sim_enc = QCRModule(embed_dim, v_SA_dropout, nhead, dim_feedforward, v_dropout, vl_dropout,
                                  v_transformer_layer, vl_transformer_layer)
        self.compute_loss = ComputeFinalLoss(tau)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.sim_enc.cuda()
            cudnn.benchmark = True

    def forward_emb(self, images, captions):
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        img_embs = self.img_enc(images)
        cap_embs, indices, t_mask = self.txt_enc(captions)
        return img_embs, cap_embs, indices, t_mask

    def info_nce(self, query, target):
        sim_t2q = torch.matmul(
            query.unsqueeze(1).unsqueeze(1), target.permute(0, 2, 1)
        ).squeeze()
        bs = query.size(0)
        targets = torch.linspace(0, bs - 1, bs, dtype=int).to(query.device)
        sim_i2t, _ = sim_t2q.max(-1)
        sim_i2t = sim_i2t / self.temp
        return F.cross_entropy(sim_i2t, targets)

    def forward_sim(self, img_embs, cap_embs, indices, t_mask, global_feats, current_test_turns):
        sim_local_1, sim_local_2 = self.sim_enc(img_embs, cap_embs, indices, t_mask,
                                                global_feats, current_test_turns)
        return sim_local_1, sim_local_2

    def forward_loss(self, sim_local_1, sim_local_2, **kwargs):
        loss = self.compute_loss(sim_local_1, sim_local_2)
        return loss

    def get_final_loss(self, image_feats, caption_feats, global_feats):
        img_embs, cap_embs, indices, t_mask = self.forward_emb(image_feats, caption_feats)
        sim_local_1, sim_local_2 = self.forward_sim(img_embs, cap_embs, indices, t_mask, global_feats,
                                                    current_test_turns=32)
        loss = self.loss_weight * self.forward_loss(sim_local_1, sim_local_2)
        return loss

    def forward(self, samples, device):
        image = samples["image"]
        target = samples["target"]
        text = samples["text_input"]
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)  # (1,Q,D) -> (B,Q,D)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        taregt_embeds = self.ln_vision(self.visual_encoder(target))
        target_atts = torch.ones(taregt_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        target_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=taregt_embeds,
            encoder_attention_mask=target_atts,
            use_cache=True,
            return_dict=True,
        )
        target_feats = F.normalize(
            self.vision_proj(target_output.last_hidden_state), dim=-1
        )
        fusion_feats_ = self.text_proj(fusion_output.last_hidden_state)
        global_feats = F.normalize(fusion_feats_[:, 32, :], dim=-1)
        fusion_feats = F.normalize(fusion_feats_[:, :32, :], dim=-1)
        
        loss_context = self.get_final_loss(target_feats, fusion_feats, global_feats)
        loss_rank = self.info_nce(global_feats, target_feats)
        total_loss = loss_rank + loss_context
        return {'loss_stu_rank': total_loss}


    @torch.no_grad()
    def extract_retrieval_compose(self, img, mod, return_attns=False):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(img))
        image_embeds_frozen = image_embeds_frozen.float()
        reference_embeds = image_embeds_frozen
        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(
            reference_embeds.device
        )
        query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        text_tokens = self.tokenizer(
            mod,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_embeds.device)
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            output_attentions=return_attns
        )
        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, 32, :]), dim=-1
        )
        return fusion_feats.unsqueeze(1).unsqueeze(1)

    @torch.no_grad()
    def extract_retrieval_target(self, img):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(img))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
            output_attentions=True
        )
        image_embeds = query_output.last_hidden_state
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features.permute(0, 2, 1)

    @torch.no_grad()
    def extract_target_features(self, image, mode='mean'):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features, image_embeds_frozen

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        image = samples.get("image")
        caption = samples.get("text_input")
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None
        if mode == "image":
            assert (
                    image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                    caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)
        elif mode == "multimodal":
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        k_test = task_cfg.k_test
        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)

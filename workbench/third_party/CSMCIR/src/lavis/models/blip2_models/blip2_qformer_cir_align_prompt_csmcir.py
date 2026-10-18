
"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
import math
# import peft
from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures
import random

@registry.register_model("blip2_cir_align_prompt_csmcir")
class Blip2QformerCirAlignPromptCSMCIR(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

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
        cross_attention_freq=6,
        embed_dim=256,
        max_txt_len=32,
        memory_bank_size=512, # 新增：Memory Bank 的大小
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
            
        # 冻结ViT参数，除了LoRA权重
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                    param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            # logging.info("freeze vision encoder")

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
        # new tokens
        # self.prompt_tokens = nn.Parameter(
        #     torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        # )
        # self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)
        # 新增：初始化 Memory Bank
        self.memory_bank_visual = torch.randn(257 *1408 , memory_bank_size)#1408。 vit-l是1024
        self.memory_bank_visual = F.normalize(self.memory_bank_visual, dim=0)
        self.memory_bank_text = torch.zeros(max_txt_len, memory_bank_size, dtype=torch.long)
        # 新增：用于存储文本的 attention_mask
        self.memory_bank_text_mask = torch.zeros(max_txt_len, memory_bank_size, dtype=torch.long)
        self.memory_bank_ptr = 0
        self.memory_bank_size = memory_bank_size
        self.attn_proj = nn.Linear(embed_dim, embed_dim)  # 映射fusion_feats为注意力查询
        self.attn_temp = nn.Parameter(torch.tensor(1.0))  # 注意力温度参数，控制权重锐度
        # 额外初始化用于信息熵计算的CLS token存储
        self.memory_bank_cls_tokens = torch.zeros(1408, memory_bank_size)#1408
        self.memory_bank_cls_tokens = F.normalize(self.memory_bank_cls_tokens, dim=0)
        self.token_importance = nn.Parameter(torch.ones(num_query_token)).unsqueeze(0).unsqueeze(-1)

    def compute_entropy(self, cls_tokens, reference_cls_tokens):
        """
        使用CLS token计算特征向量的信息熵
        
        Args:
            cls_tokens: 要计算熵的CLS token特征 [B, D]
            reference_cls_tokens: 参考CLS token特征集合 [N, D]
            
        Returns:
            entropy: 每个特征的信息熵 [B]
        """
        # 计算相似度
        similarities = torch.matmul(cls_tokens, reference_cls_tokens.t())
        # 转换为概率分布
        prob_dist = F.softmax(similarities /100, dim=1)#100self.temp 
        # 计算熵
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-10), dim=1)
        return entropy

    def entropy_based_memory_update(self, target_embeds, new_text_features, new_text_mask):
        """
        基于信息熵和时间衰减的Memory Bank更新策略
        按照论文公式：高熵样本替换低retention熵样本
        
        Args:
            target_embeds: 目标图像的frozen ViT特征 [B, 257, 1408]
            new_text_features: 文本token ids [B, L]
            new_text_mask: 文本attention mask [B, L]
        """
        batch_size = target_embeds.size(0)
        device = target_embeds.device
        
        # 提取CLS token
        cls_tokens = target_embeds[:, 0, :]  # [B, 1408]
        
        # 跟踪Memory Bank中有效样本的数量
        valid_size = min(self.memory_bank_size, self.memory_bank_ptr)
        
        # ===== 初次填充：直接添加样本 =====
        if valid_size == 0:
            for i in range(min(batch_size, self.memory_bank_size)):
                self.memory_bank_visual[:, i] = target_embeds[i].view(-1).cpu()
                self.memory_bank_cls_tokens[:, i] = cls_tokens[i].cpu()
                self.memory_bank_text[:, i] = new_text_features[i].cpu()
                self.memory_bank_text_mask[:, i] = new_text_mask[i].cpu()
            
            self.memory_bank_ptr = min(batch_size, self.memory_bank_size)
            if not hasattr(self, 'memory_sample_age'):
                self.memory_sample_age = torch.zeros(self.memory_bank_size, device='cpu')
            return
        
        # 添加样本年龄信息（如果不存在）
        if not hasattr(self, 'memory_sample_age'):
            self.memory_sample_age = torch.zeros(self.memory_bank_size, device='cpu')
        
        # ===== 关键修改1：增加所有样本的年龄（保持原有逻辑）=====
        self.memory_sample_age += 1
        
        # 将Memory Bank的CLS tokens移到device
        memory_cls_tokens = self.memory_bank_cls_tokens.t().to(device)  # [M, 1408]
        
        # ===== 修改2：计算熵值（论文公式3-4）=====
        # H^B_i: batch样本相对于memory样本的熵
        H_batch = self.compute_entropy(cls_tokens, memory_cls_tokens)  # [B]
        
        # H^M_i: memory样本相对于memory样本的熵
        H_memory = self.compute_entropy(memory_cls_tokens, memory_cls_tokens)  # [M]
        
        # ===== 修改3：计算retention entropy（论文公式5）=====
        # Delta_t就是累加的memory_sample_age
        delta_t = self.memory_sample_age.to(device)
        
        # N_max默认值100，可在__init__中设置：self.N_max = 100
        N_max = getattr(self, 'N_max', 10)
        
        # freshness factor: max(0, 1 - Delta_t / N_max)
        freshness_factor = torch.clamp(1.0 - delta_t.float() / N_max, min=0.0)
        
        # retention entropy: H_hat^M_i = freshness_factor * H^M_i
        H_retention = freshness_factor * H_memory  # [M]
        
        # ===== 修改4：执行替换策略（H^B_j > H_hat^M_i）=====
        # 按H_batch降序排序
        _, batch_indices = torch.sort(H_batch, descending=True)
        
        # 记录已用于替换的batch样本
        replaced_batch_indices = set()
        
        # 遍历batch样本，尝试替换
        for idx in batch_indices:
            if len(replaced_batch_indices) >= valid_size:
                break
            
            # 找到retention entropy最低的memory位置
            min_retention_idx = torch.argmin(H_retention)
            
            # 替换条件：H^B_j > H_hat^M_i
            if H_batch[idx] > H_retention[min_retention_idx]:
                mem_idx = min_retention_idx.item()
                batch_idx = idx.item()
                
                # 更新Memory Bank
                self.memory_bank_visual[:, mem_idx] = target_embeds[batch_idx].view(-1).cpu()
                self.memory_bank_cls_tokens[:, mem_idx] = cls_tokens[batch_idx].cpu()
                self.memory_bank_text[:, mem_idx] = new_text_features[batch_idx].cpu()
                self.memory_bank_text_mask[:, mem_idx] = new_text_mask[batch_idx].cpu()
                
                # 重置被替换位置的年龄
                self.memory_sample_age[mem_idx] = 0
                
                # 标记该位置已被替换
                H_retention[min_retention_idx] = float('inf')
                replaced_batch_indices.add(batch_idx)
            else:
                # 当前样本不满足条件，后续更低熵的样本也不会满足
                break
        
        # ===== 修改5：Memory Bank未满时继续填充 =====
        if self.memory_bank_ptr < self.memory_bank_size:
            space_left = self.memory_bank_size - self.memory_bank_ptr
            
            # 选择熵值较高且未被替换的样本
            num_added = 0
            for idx in batch_indices:
                if num_added >= space_left:
                    break
                
                batch_idx = idx.item()
                if batch_idx in replaced_batch_indices:
                    continue
                
                pos = self.memory_bank_ptr + num_added
                self.memory_bank_visual[:, pos] = target_embeds[batch_idx].view(-1).cpu()
                self.memory_bank_cls_tokens[:, pos] = cls_tokens[batch_idx].cpu()
                self.memory_bank_text[:, pos] = new_text_features[batch_idx].cpu()
                self.memory_bank_text_mask[:, pos] = new_text_mask[batch_idx].cpu()
                self.memory_sample_age[pos] = 0  # 新添加样本年龄为0
                num_added += 1
            
            self.memory_bank_ptr += num_added

    def forward(self, samples):
        image = samples["image"]
        target = samples["target"]
        text = samples["text_input"]
        target_image_caption = samples['image_captions']

        ###============== reference text fusion ===================###
        # reference image feature  
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        # query tokens
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        # fusion reference image and text tokens into a set of multi-modal tokens
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, 32, :]), dim=-1#32
        )

        ###============== Fusion-target Contrastive ===================###
        # target image feature  
        taregt_embeds = self.ln_vision(self.visual_encoder(target))
        target_atts = torch.ones(taregt_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        #target_image_caption tokens
        target_image_caption_tokens = self.tokenizer(
            target_image_caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        target_attention_mask = torch.cat([query_atts, target_image_caption_tokens.attention_mask], dim=1)
        target_output = self.Qformer.bert(
            target_image_caption_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=target_attention_mask,
            encoder_hidden_states=taregt_embeds,
            encoder_attention_mask=target_atts,
            use_cache=True,
            return_dict=True,
        )
        target_feats = F.normalize(
            self.vision_proj(target_output.last_hidden_state[:, :32, :]), dim=-1#64
        )

        # #方法3
        weighted_feats = target_feats.cuda() * self.token_importance.cuda()
        zq = weighted_feats.mean(dim=1)  # [batch_size, embed_dim]
        
        # 2. 计算余弦相似度
        cos_sim = F.cosine_similarity(zq, fusion_feats, dim=1)
        # cos_sim = F.cosine_similarity(target_feats.mean(1), fusion_feats, dim=1)
        # 保持原有loss计算逻辑
        loss_align = (1 - cos_sim).mean()

        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        ).squeeze()

        sim_i2t, max_indices = sim_t2q.max(-1)  # 获取最大值的索引
        sim_i2t = sim_i2t / self.temp
        bs = image.size(0)
        targets = torch.linspace(0, bs - 1, bs, dtype=int).to(
            image.device
        )
        #加loss_itc = F.cross_entropy(sim_i2t, targets)对比loss2
        sim_q2t=torch.matmul(
            target_feats.mean(dim=1),fusion_feats.permute(1,0)
        )
        sim_q2t = sim_q2t / self.temp
        loss_q2t = F.cross_entropy(sim_q2t, targets)



        # 从 Memory Bank 中获取之前的视觉和文本特征
        memory_bank_visual = self.memory_bank_visual.to(image.device)
        memory_bank_text = self.memory_bank_text.to(image.device)
        memory_bank_text_mask = self.memory_bank_text_mask.to(image.device)

        # 计算 Memory Bank 中特征的跨模态特征
        memory_feats_list = []
        batch_size_memory = 64  # 可以根据显存情况调整批次大小
        for i in range(0, memory_bank_visual.shape[1], batch_size_memory):
            end_idx = min(i + batch_size_memory, memory_bank_visual.shape[1])
            memory_query_tokens_batch = self.query_tokens.expand(end_idx - i, -1, -1)
            memory_query_atts_batch = torch.ones(memory_query_tokens_batch.size()[:-1], dtype=torch.long).to(
                image.device
            )
            memory_attention_mask_batch = torch.cat([memory_query_atts_batch, memory_bank_text_mask[:, i:end_idx].T], dim=1)
            memory_output_batch = self.Qformer.bert(
                memory_bank_text[:, i:end_idx].T,
                query_embeds=memory_query_tokens_batch,
                attention_mask=memory_attention_mask_batch,
                encoder_hidden_states=memory_bank_visual[:, i:end_idx].T.view(end_idx - i, 257,1408),#1408. vit-l 1024
                encoder_attention_mask=torch.ones(end_idx - i, 1, dtype=torch.long).to(image.device),
                use_cache=True,
                return_dict=True,
            )
            memory_feats_batch = F.normalize(
                self.vision_proj(memory_output_batch.last_hidden_state[:, :32, :]), dim=-1#32
            )
            memory_feats_list.append(memory_feats_batch)
        memory_feats = torch.cat(memory_feats_list, dim=0)

        # 合并当前轮次和 Memory Bank 中的特征
        all_feats = torch.cat([target_feats, memory_feats], dim=0)
        sim_t2all = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), all_feats.permute(0, 2, 1)
        ).squeeze()

        sim_i2all, max_indices_all = sim_t2all.max(-1)  # 获取最大值的索引
        sim_i2all = sim_i2all / self.temp
        targets_all = torch.arange(sim_i2t.shape[0], dtype=torch.long).to(image.device)
        loss_itc_memory = F.cross_entropy(sim_i2all, targets_all)

        # 释放不必要的中间变量
        del image_embeds, image_atts, query_tokens, query_atts, text_tokens
        torch.cuda.empty_cache()

        # 使用基于CLS token信息熵的策略更新Memory Bank
        with torch.no_grad():
            self.entropy_based_memory_update(
                taregt_embeds,  # 目标图像特征
                target_image_caption_tokens.input_ids,
                target_image_caption_tokens.attention_mask
            )

        return {
            'loss_itc_memory': loss_itc_memory,  # 新增：包含 Memory Bank 的损失
            'loss_align':loss_align,
        }

    @torch.no_grad()
    def inference(self, reference_embeds, target_feats, text,text_cot):
        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(
            reference_embeds.device
        )
        # query tokens
        query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            text,
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
        )

        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, 32, :]), dim=-1)

        #inference gai
        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_i2t, _ = sim_t2q.max(-1)
        # sim_i2t, _ = torch.topk(sim_t2q, k=5, dim=-1)
        # sim_i2t = sim_i2t.mean(-1)
        return sim_i2t

    @torch.no_grad()
    def inference_tsen(self, reference_embeds, target_feats, text,text_cot):
        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(
            reference_embeds.device
        )
        # query tokens
        query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            text,
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
        )

        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, 32, :]), dim=-1)

        #inference gai
        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        ).squeeze()

        sim_i2t, _ = sim_t2q.max(-1)

        return sim_t2q,fusion_feats



    
    @torch.no_grad()
    def extract_target_caption_features(self, image, text,mode='mean'):
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
        # text tokens
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

        query_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state[:, :32, :]#32

        # return image_embeds
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features, image_embeds_frozen



    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token",16)
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)
        
        # # LoRA 相关配置
        # use_lora = cfg.get("use_lora", False)
        # lora_r = cfg.get("lora_r", 8)
        # lora_alpha = cfg.get("lora_alpha", 32)
        # lora_dropout = cfg.get("lora_dropout", 0)

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
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
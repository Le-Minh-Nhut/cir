"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import torch
from omegaconf import OmegaConf
from lavis.common.registry import registry

from lavis.models.base_model import BaseModel

from lavis.models.albef_models.albef_classification import AlbefClassification
from lavis.models.albef_models.albef_feature_extractor import AlbefFeatureExtractor
from lavis.models.albef_models.albef_nlvr import AlbefNLVR
from lavis.models.albef_models.albef_pretrain import AlbefPretrain
from lavis.models.albef_models.albef_retrieval import AlbefRetrieval
from lavis.models.albef_models.albef_vqa import AlbefVQA
from lavis.models.alpro_models.alpro_qa import AlproQA
from lavis.models.alpro_models.alpro_retrieval import AlproRetrieval

from lavis.models.blip_models.blip import BlipBase
from lavis.models.blip_models.blip_caption import BlipCaption
from lavis.models.blip_models.blip_classification import BlipClassification
from lavis.models.blip_models.blip_feature_extractor import BlipFeatureExtractor
from lavis.models.blip_models.blip_cir_base import BlipCirBase
from lavis.models.blip_models.blip_image_text_matching import BlipITM
from lavis.models.blip_models.blip_nlvr import BlipNLVR
from lavis.models.blip_models.blip_pretrain import BlipPretrain
from lavis.models.blip_models.blip_retrieval import BlipRetrieval
from lavis.models.blip_models.blip_vqa import BlipVQA

from lavis.models.blip2_models.blip2 import Blip2Base
from lavis.models.blip2_models.blip2_opt import Blip2OPT
from lavis.models.blip2_models.blip2_t5 import Blip2T5
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from lavis.models.blip2_models.blip2_image_text_matching import Blip2ITM

from lavis.models.blip2_models.blip2_qformer_prompt import Blip2QformerPrompt
from lavis.models.blip2_models.blip2_qformer_cir_cat import Blip2QformerCirCat
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt import Blip2QformerCirAlignPrompt
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_caption_query import Blip2QformerCirAlignPromptCaptionQuery
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_daul_contrastive import Blip2QformerCirAlignPromptDaulContrastive
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_alignloss import Blip2QformerCirAlignPromptAlignLoss
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption import Blip2QformerCirAlignPromptTargetImageCaption
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_logsumexp import Blip2QformerCirAlignPromptLogsumexp
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_attentionpooling import Blip2QformerCirAlignPromptAttentionpooling
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_gatepool import Blip2QformerCirAlignPromptGatepool
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_aggregator import Blip2QformerCirAlignPromptAggregator
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan4 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan4
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan5 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan5
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan6 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan6
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan7 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan7
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan7wocaption import Blip2QformerCirAlignPromptTargetImageCaptionshiyan7wocaption
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan7cross4 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan7cross4
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan7cross6 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan7cross6
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan7cross1 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan7cross1
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan7cross3 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan7cross3
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan7prompttoken import Blip2QformerCirAlignPromptTargetImageCaptionshiyan7prompttoken




from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan8 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan8
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan9 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan9
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan10 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan10
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan11 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan11
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan12 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan12
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan13 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan13
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan14 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan14
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan15 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan15
# from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan16 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan16
# from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan16cirr import Blip2QformerCirAlignPromptTargetImageCaptionshiyan16cirr
# from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan17 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan17
# from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan18 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan18
# from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan16_17 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan16_17
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan19 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan19
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan20 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan20
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan21 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan21
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan22 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan22
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan23 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan23
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan24 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan24
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan25 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan25
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan26 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan26
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan27 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan27
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan28 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan28
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan29 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan29
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan30 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan30
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan31 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan31
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan32 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan32
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan33 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan33
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan34 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan34
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan35 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan35
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan36 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan36
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan37 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan37
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan38 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan38
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan39 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan39
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan40 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan40
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan41 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan41
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan42 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan42
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan43 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan43
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan44 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan44
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan45 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan45
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan46 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan46
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan47 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan47
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan48 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan48
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan49 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan49
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan50 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan50
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan51 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan51
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan52 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan52
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan53 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan53
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan54 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan54
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan52openvit import Blip2QformerCirAlignPromptTargetImageCaptionshiyan52openvit
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan55 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan55
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan56 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan56
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan57 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan57
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan58 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan58
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan59 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan59
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan60 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan60
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan61 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan61
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan62 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan62
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan63 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan63
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan64 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan64
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan65 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan65
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan66 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan66
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan67 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan67
# from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan68 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan68
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan69 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan69
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan70_pretrain import Blip2QformerCirAlignPromptTargetImageCaptionshiyan70_pretrain
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan71 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan71
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan72 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan72
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan73 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan73
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan74 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan74
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan76 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan76
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan77 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan77
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan78 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan78
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan79 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan79
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan80 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan80
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan81 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan81
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan82 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan82
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan83 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan83
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan84 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan84
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan85 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan85
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan86 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan86
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan88 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan88
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan89 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan89
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan90 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan90
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan91 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan91
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan92 import Blip2QformerCirAlignPromptTargetImageCaptionshiyan92
from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_csmcir import Blip2QformerCirAlignPromptCSMCIR


from lavis.models.blip2_models.blip2_qformer_cir_cls import Blip2QformerCirCls
from lavis.models.blip2_models.blip2_qformer_cir_slt import Blip2QformerCirSlt
from lavis.models.blip2_models.blip2_qformer_cir_neg import Blip2QformerCirNeg
from lavis.models.blip2_models.blip2_qformer_cir_rel import Blip2QformerCirRel
from lavis.models.blip2_models.blip2_qformer_cir_full import Blip2QformerCirFull
from lavis.models.blip2_models.blip2_qformer_cir_neg_rel import Blip2QformerCirNegRel
from lavis.models.blip2_models.blip2_qformer_cir_rerank import Blip2QformerCirRerank
from lavis.models.blip2_models.blip2_qformer_cir_z_learn_pos_align import Blip2QformerCirZLearnPosAlign
from lavis.models.blip2_models.blip2_instruct_cir import Blip2InstructCir

from lavis.models.blip2_models.blip2_t5_instruct import Blip2T5Instruct
from lavis.models.blip2_models.blip2_vicuna_instruct import Blip2VicunaInstruct

from lavis.models.pnp_vqa_models.pnp_vqa import PNPVQA
from lavis.models.pnp_vqa_models.pnp_unifiedqav2_fid import PNPUnifiedQAv2FiD
from lavis.models.img2prompt_models.img2prompt_vqa import Img2PromptVQA
from lavis.models.med import XBertLMHeadDecoder
from lavis.models.vit import VisionTransformerEncoder
from lavis.models.clip_models.model import CLIP

from lavis.models.gpt_models.gpt_dialogue import GPTDialogue

from lavis.processors.base_processor import BaseProcessor


__all__ = [
    "load_model",
    "AlbefClassification",
    "AlbefFeatureExtractor",
    "AlbefNLVR",
    "AlbefVQA",
    "AlbefPretrain",
    "AlbefRetrieval",
    "AlproQA",
    "AlproRetrieval",
    "BaseModel",
    "BlipBase",
    "BlipCirBase",
    "BlipFeatureExtractor",
    "BlipCaption",
    "BlipClassification",
    "BlipITM",
    "BlipNLVR",
    "BlipPretrain",
    "BlipRetrieval",
    "BlipVQA",
    "Blip2Qformer",
    "Blip2QformerPrompt",
    "Blip2QformerCirCat",
    "Blip2QformerCirCls",
    "Blip2QformerCirSlt",
    "Blip2QformerCirNeg",
    "Blip2QformerCirRel",
    "Blip2QformerCirNegRel",
    "Blip2QformerCirZLearnPosAlign",
    "Blip2QformerCirAlignPrompt",
    "Blip2QformerCirFull",
    "Blip2Base",
    "Blip2ITM",
    "Blip2OPT",
    "Blip2T5",
    "Blip2T5Instruct",
    "Blip2VicunaInstruct",
    "Blip2InstructCir",
    "PNPVQA",
    "Img2PromptVQA",
    "PNPUnifiedQAv2FiD",
    "CLIP",
    "VisionTransformerEncoder",
    "XBertLMHeadDecoder",
    "GPTDialogue",
]


def load_model(name, model_type, is_eval=False, device="cpu", checkpoint=None):
    """
    Load supported models.

    To list all available models and types in registry:
    >>> from lavis.models import model_zoo
    >>> print(model_zoo)

    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".
        checkpoint (str): path or to checkpoint. Default: None.
            Note that expecting the checkpoint to have the same keys in state_dict as the model.

    Returns:
        model (torch.nn.Module): model.
    """

    model = registry.get_model_class(name).from_pretrained(model_type=model_type)

    if checkpoint is not None:
        model.load_checkpoint(checkpoint)

    if is_eval:
        model.eval()

    if device == "cpu":
        model = model.float()

    return model.to(device)


def load_preprocess(config):
    """
    Load preprocessor configs and construct preprocessors.

    If no preprocessor is specified, return BaseProcessor, which does not do any preprocessing.

    Args:
        config (dict): preprocessor configs.

    Returns:
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.

        Key is "train" or "eval" for processors used in training and evaluation respectively.
    """

    def _build_proc_from_cfg(cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else BaseProcessor()
        )

    vis_processors = dict()
    txt_processors = dict()

    vis_proc_cfg = config.get("vis_processor")
    txt_proc_cfg = config.get("text_processor")

    if vis_proc_cfg is not None:
        vis_train_cfg = vis_proc_cfg.get("train")
        vis_eval_cfg = vis_proc_cfg.get("eval")
    else:
        vis_train_cfg = None
        vis_eval_cfg = None

    vis_processors["train"] = _build_proc_from_cfg(vis_train_cfg)
    vis_processors["eval"] = _build_proc_from_cfg(vis_eval_cfg)

    if txt_proc_cfg is not None:
        txt_train_cfg = txt_proc_cfg.get("train")
        txt_eval_cfg = txt_proc_cfg.get("eval")
    else:
        txt_train_cfg = None
        txt_eval_cfg = None

    txt_processors["train"] = _build_proc_from_cfg(txt_train_cfg)
    txt_processors["eval"] = _build_proc_from_cfg(txt_eval_cfg)

    return vis_processors, txt_processors


def load_model_and_preprocess(name, model_type, is_eval=False, device="cpu"):
    """
    Load model and its related preprocessors.

    List all available models and types in registry:
    >>> from lavis.models import model_zoo
    >>> print(model_zoo)

    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".

    Returns:
        model (torch.nn.Module): model.
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.
    """
    model_cls = registry.get_model_class(name)

    # load model
    model = model_cls.from_pretrained(model_type=model_type)

    if is_eval:
        model.eval()

    # load preprocess
    cfg = OmegaConf.load(model_cls.default_config_path(model_type))
    if cfg is not None:
        preprocess_cfg = cfg.preprocess

        vis_processors, txt_processors = load_preprocess(preprocess_cfg)
    else:
        vis_processors, txt_processors = None, None
        logging.info(
            f"""No default preprocess for model {name} ({model_type}).
                This can happen if the model is not finetuned on downstream datasets,
                or it is not intended for direct use without finetuning.
            """
        )

    if device == "cpu" or device == torch.device("cpu"):
        model = model.float()

    return model.to(device), vis_processors, txt_processors


class ModelZoo:
    """
    A utility class to create string representation of available model architectures and types.

    >>> from lavis.models import model_zoo
    >>> # list all available models
    >>> print(model_zoo)
    >>> # show total number of models
    >>> print(len(model_zoo))
    """

    def __init__(self) -> None:
        self.model_zoo = {
            k: list(v.PRETRAINED_MODEL_CONFIG_DICT.keys())
            for k, v in registry.mapping["model_name_mapping"].items()
        }

    def __str__(self) -> str:
        return (
            "=" * 50
            + "\n"
            + f"{'Architectures':<30} {'Types'}\n"
            + "=" * 50
            + "\n"
            + "\n".join(
                [
                    f"{name:<30} {', '.join(types)}"
                    for name, types in self.model_zoo.items()
                ]
            )
        )

    def __iter__(self):
        return iter(self.model_zoo.items())

    def __len__(self):
        return sum([len(v) for v in self.model_zoo.values()])


model_zoo = ModelZoo()

import torch 
import numpy as np 
import torch.nn as nn 
import torchvision
import text_model as text_model_
import torch.nn.functional as F
import sys
from torch.nn.parameter import Parameter
import resnet
import math
from blocks import *

class local_conv(nn.Module):
    def __init__(self, img_channel=2048, text_dim=1024, T=7.0):
        super().__init__()
        self.img1x1conv = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)  
        self.T = T
        #self.T = Parameter(torch.FloatTensor([T]))
        self.inception_gamma = Inception2d(in_dim=2048)
        self.inception_beta = Inception2d(in_dim=2048)
        
    def forward(self, img_tensor, text_embed):         #  text_embed   N,L,1024
        img_embed = self.img1x1conv(img_tensor)  
        n,c,h,w = img_embed.size()
        img_embed = img_embed.view(n,c,h*w)
        dot_product = torch.bmm(text_embed, img_embed)   # N,L,49
        atten = self.softmax(dot_product / self.T)
        sentence_cat = []
        for i in range(img_tensor.size(2)*img_tensor.size(3)):
            sentence = torch.sum(text_embed * atten[:,:,i].unsqueeze(-1), dim=1)  # N,1024
            sentence_cat.append(sentence)
        sentence_cat = torch.stack(sentence_cat).permute(1, 2, 0).contiguous()  # N,1024,49
 
        x = torch.cat([img_embed, sentence_cat], dim=1).view(n,-1,h,w)  # N,2048,7,7
        gamma = self.inception_gamma(x)
        beta = self.inception_beta(x)
        return gamma * img_tensor + beta, atten

class global_conv(nn.Module):
    def __init__(self, img_channel=2048, text_dim=1024, T=4.0):
        super().__init__()
        self.img1x1conv = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.T = T
        #self.T = Parameter(torch.FloatTensor([T]))
        self.inception_gamma = Inception1d(in_dim=2048)
        self.inception_beta = Inception1d(in_dim=2048)

    def forward(self, img_tensor, text_embed):    # text_embed N,L,1024

        img_embed = self.img1x1conv(img_tensor)   # N,1024,7,7
        n,c,h,w = img_embed.size()
        img_embed = img_embed.view(n,c,h*w)
        dot_product = torch.bmm(text_embed, img_embed)
        attn = self.softmax(dot_product / self.T)
        img_cat = []
        for i in range(text_embed.shape[1]):
            img = torch.sum(img_embed * attn[:,i,:].unsqueeze(1), dim=-1) # N,1024
            img_cat.append(img)
        img_cat = torch.stack(img_cat).permute(1, 2, 0).contiguous()   # N,1024,L
        
        x = torch.cat([text_embed.permute(0, 2, 1).contiguous(), img_cat], dim=1) # N,2048,L

        gamma = self.inception_gamma(x)
        beta = self.inception_beta(x)
        return gamma * img_cat + beta, attn

class img_backbone(nn.Module):
    def __init__(self, dim=1024, dropout_p=0.2):
        super().__init__()
        self.img_model = resnet.resnet50(pretrained=True)
        self.img_pool = GeM(p=3)
        self.img_fc = nn.Linear(2048, dim)
        self.img_model.fc = nn.Sequential()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, imgs):
        imgs = self.img_model.conv1(imgs)
        imgs = self.img_model.bn1(imgs)
        imgs = self.img_model.relu(imgs)
        imgs = self.img_model.maxpool(imgs)
        imgs = self.img_model.layer1(imgs)
        imgs = self.img_model.layer2(imgs)
        imgs = self.img_model.layer3(imgs)
        imgs = self.img_model.layer4(imgs)
        img_feature = self.img_fc(self.img_pool(imgs))
        return imgs, img_feature

class LabelSmooth1(torch.nn.Module):
    def __init__(self, epsilon=0.1, reduction='none'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = torch.nn.functional.log_softmax(preds, dim=-1)
        loss = -log_preds.sum(dim=-1)
        nll = torch.nn.functional.nll_loss(log_preds, target, reduction=self.reduction)
        return (self.epsilon* loss / n + (1-self.epsilon) * nll)

class LabelSmooth2(torch.nn.Module):
    def __init__(self, epsilon=0.1, reduction='none'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    def forward(self, preds, target):
        n_class = preds.size(-1)
        one_hot = torch.zeros_like(preds).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.epsilon) + (1 - one_hot) * self.epsilon / (n_class - 1)
        log_prb = torch.nn.functional.log_softmax(preds, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        return loss

class compose_local(nn.Module):
    def __init__(self, texts, T, word_dim=512, lstm_dim=1024, dim=1024, dropout_p=0.2):
        super().__init__()
        self.loss_weight = torch.nn.Parameter(torch.FloatTensor((10.,)))
        self.text_model = text_model_.TextLSTMModel(
            texts_to_build_vocab=texts,
            word_embed_dim=word_dim,
            lstm_hidden_dim=lstm_dim)
        self.img_model = img_backbone()
        self.local_fuse = local_conv(T = T)
        self.relu = nn.ReLU()
        self.local_pool = GeM(p=3)
        self.local_fc = nn.Linear(2048, dim)
        self.dropout = nn.Dropout(dropout_p)

    def extract_img_feature(self, imgs):
        imgs, img_feature = self.img_model(imgs)
        return imgs, img_feature

    def compose_img_text(self, imgs, texts):
        lstm_tensor = self.text_model(texts) # N,L,C
        imgs, img_feature = self.extract_img_feature(imgs)

        local_out, attn = self.local_fuse(imgs, lstm_tensor) 
        local_feature = self.local_fc(self.local_pool(local_out))  
        return local_out, local_feature, attn

    def compute_loss(self, img1, mods, img2, target_global_tensor, target_global_feature, query_global_feature):
        target_local_tensor, target_local_feature = self.extract_img_feature(img2)
        query_local_tensor, query_local_feature, attn = self.compose_img_text(img1, mods)

        loss = {}
        loss['img'] = self.compute_l2(target_local_feature, target_global_feature) + self.compute_l2(query_local_feature, query_global_feature)
        loss['class'] = self.compute_batch_based_classification_loss_(query_local_feature, target_local_feature)
        loss['perceptual'] = self.compute_l2(query_local_tensor, target_local_tensor)
        loss['mul_kl'] = self.mutual_learning(query_local_feature, target_local_feature, query_global_feature, target_global_feature)

        return loss

    def mutual_learning(self, query1, target1, query2, target2):
        query1 = F.normalize(query1, p=2, dim=-1)
        query2 = F.normalize(query2, p=2, dim=-1)
        target1 = F.normalize(target1, p=2, dim=-1)
        target2 = F.normalize(target2, p=2, dim=-1)
        x1 = 10.0 * torch.mm(query1, target1.transpose(0, 1)) 
        x2 = 10.0 * torch.mm(query2, target2.transpose(0, 1))

        log_soft_x1 = F.log_softmax(x1, dim=1)
        soft_x2 = F.softmax(torch.autograd.Variable(x2), dim=1)
        kl = F.kl_div(log_soft_x1, soft_x2, reduction='batchmean')
        return kl


    def compute_batch_based_classification_loss_(self, mod_img1, img2, negtive=None):
        mod_img1 = F.normalize(mod_img1, p=2, dim=-1)
        img2 = F.normalize(img2, p=2, dim=-1)
        x = torch.mm(mod_img1, img2.transpose(0, 1)) 
        if negtive is not None:
            negtive = F.normalize(negtive)
            y = torch.mm(mod_img1, negtive.transpose(0, 1)).diag().unsqueeze(-1)
            x = torch.cat([x, y], dim=-1)
        
        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()
        loss = F.cross_entropy(self.loss_weight * x, labels)   # loss_weight temperature
        return loss
    
    def compute_l2(self, x1, x2):
        l2_loss = torch.nn.MSELoss()
        return l2_loss(x1, x2)


class compose_global(nn.Module):
    def __init__(self, texts, T, word_dim=512, lstm_dim=1024, dim=1024, dropout_p=0.2):
        super().__init__()
        self.loss_weight = torch.nn.Parameter(torch.FloatTensor((10.,)))
        self.text_model = text_model_.TextLSTMModel(
            texts_to_build_vocab=texts,
            word_embed_dim=word_dim,
            lstm_hidden_dim=lstm_dim)
        self.img_model = img_backbone()
        self.global_fuse = global_conv(T=T)
        self.relu = nn.ReLU()
        self.global_pool = GeM(p=3)
        self.global_fc = nn.Linear(1024, dim)
        self.dropout = nn.Dropout(dropout_p)

    def extract_img_feature(self, imgs):
        imgs, img_feature = self.img_model(imgs)
        return imgs, img_feature

    def compose_img_text(self, imgs, texts):
        lstm_tensor = self.text_model(texts) # N,L,C
        imgs, img_feature = self.extract_img_feature(imgs)

        global_out, attn = self.global_fuse(imgs, lstm_tensor)
        global_feature = self.global_fc(self.global_pool(global_out))
        return global_feature, attn

    def compute_loss(self, img1, mods, img2, target_local_tensor, target_local_feature, query_local_feature):
        target_global_tensor, target_global_feature = self.extract_img_feature(img2)
        query_global_feature, attn = self.compose_img_text(img1, mods)

        loss = {}
        loss['img'] =  self.compute_l2(target_local_feature, target_global_feature) + self.compute_l2(query_local_feature, query_global_feature)
        loss['class'] = self.compute_batch_based_classification_loss_(query_global_feature, target_global_feature)
        loss['mul_kl'] = self.mutual_learning(query_global_feature, target_global_feature, query_local_feature, target_local_feature)
        return loss


    def mutual_learning(self, query1, target1, query2, target2):
        query1 = F.normalize(query1, p=2, dim=-1)
        query2 = F.normalize(query2, p=2, dim=-1)
        target1 = F.normalize(target1, p=2, dim=-1)
        target2 = F.normalize(target2, p=2, dim=-1)
        x1 = 10.0 * torch.mm(query1, target1.transpose(0, 1)) 
        x2 = 10.0 * torch.mm(query2, target2.transpose(0, 1))

        log_soft_x1 = F.log_softmax(x1, dim=1)
        soft_x2 = F.softmax(torch.autograd.Variable(x2), dim=1)
        kl = F.kl_div(log_soft_x1, soft_x2, reduction='batchmean')
        return kl


    def compute_batch_based_classification_loss_(self, mod_img1, img2, negtive=None):
        mod_img1 = F.normalize(mod_img1, p=2, dim=-1)
        img2 = F.normalize(img2, p=2, dim=-1)
        x = torch.mm(mod_img1, img2.transpose(0, 1)) 
        if negtive is not None:
            negtive = F.normalize(negtive)
            y = torch.mm(mod_img1, negtive.transpose(0, 1)).diag().unsqueeze(-1)
            x = torch.cat([x, y], dim=-1)

        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()

        loss = F.cross_entropy(self.loss_weight * x, labels)   # loss_weight temperature
        return loss
    
    def compute_l2(self, x1, x2):
        l2_loss = torch.nn.MSELoss()
        return l2_loss(x1, x2)






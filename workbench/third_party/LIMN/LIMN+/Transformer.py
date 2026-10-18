import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union

class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, out_d),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.fc(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5

        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_key_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        
    def forward(self, x, y=None, mask=None): # mask: n,m
        device = x.device
        y = y if y is not None else x 
        b, n, c = x.shape
        _, m, d = y.shape 

        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        keys_values = self.to_key_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            mask = ~mask
            mask = mask.unsqueeze(0).unsqueeze(3)
            attention = attention.masked_fill(mask, -float("inf"))
        
        # m_r = torch.ones_like(attention).to(attention.device) * 0.1
        # attention = attention + torch.bernoulli(m_r) * -1e12

        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention

class TransformerLayer(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0.,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), dropout=dropout)
        
    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x 
    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention
    
class Transformer(nn.Module):
    def __init__(self, dim_self, num_heads, num_layers, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super().__init__()
        self.enc_dec = enc_dec
        dim_ref = dim_ref if dim_ref is not None else dim_self
        if enc_dec:
            num_layers = num_layers * 2

        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, norm_layer=norm_layer))
            elif enc_dec:
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, norm_layer=norm_layer))
            else:
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec:
                x = layer(x, y)
            elif self.enc_dec:
                x = layer(x, x, mask)
            else:
                x = layer(x, y, mask)
        return x  

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions



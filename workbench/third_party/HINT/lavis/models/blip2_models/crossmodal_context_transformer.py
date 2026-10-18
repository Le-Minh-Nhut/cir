import torch.nn as nn


class CrossModalContextTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.with_pos_embed_v = PositionalEncoder(d_model, 0)
        self.with_pos_embed_vl = PositionalEncoder(d_model, 1)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_key_padding_mask=None, need_weights=False):
        if src_key_padding_mask is None:
            q = k = self.with_pos_embed_v(src)
            src2, attn_weights = self.self_attn(q, k, value=src)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        else:
            q = k = self.with_pos_embed_vl(src)
            src2, attn_weights = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, is_v):
        super().__init__()
        hidden_dim = d_model
        if is_v is 0: 
            num_total = 33
        else: 
            num_total = 64
        self.src_pos_embed = nn.Embedding(num_total, hidden_dim)

    def forward(self, src):
        bsize = src.size(0)
        src_pos = self.src_pos_embed.weight.unsqueeze(0).repeat(bsize, 1, 1)
        src = src + src_pos
        return src

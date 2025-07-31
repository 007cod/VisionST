import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(name):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU()
    elif name == 'prelu':
        return nn.PReLU()
    elif name == 'elu':
        return nn.ELU()
    elif name == 'selu':
        return nn.SELU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'softmax':
        return nn.Softmax(dim=1)  # 注意默认在 dim=1 上做 softmax
    elif name == 'softplus':
        return nn.Softplus()
    elif name == 'none' or name == 'identity':
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation function: {name}")

class SelfAttention(nn.Module):
    def __init__(self, D, num_heads=1, dropout=0.1):
        super().__init__()
        assert D % num_heads == 0, "D must be divisible by num_heads"
        self.D = D
        self.num_heads = num_heads
        self.head_dim = D // num_heads

        self.to_qkv = nn.Linear(D, D * 3)  # Q, K, V 合并计算
        self.out_proj = nn.Linear(D, D)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (N, D)
        """
        N, D = x.shape
        qkv = self.to_qkv(x)  # (N, 3D)
        qkv = qkv.reshape(N, 3, self.num_heads, self.head_dim)  # (N, 3, H, D_head)
        q, k, v = qkv.unbind(dim=1)  # 各自为 (N, H, D_head)

        # 转置为 (H, N, D_head)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # Attention score: (H, N, N)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attention output: (H, N, D_head)
        attn_output = torch.matmul(attn_weights, v)

        # 合并 heads: (N, D)
        attn_output = attn_output.transpose(0, 1).reshape(N, D)
        return self.out_proj(attn_output)
    
class DownDim(nn.Module):
    def __init__(self, in_num: int, out_num: int):
        super().__init__()
        self.att = nn.Parameter(torch.empty(out_num, in_num))
        self._reset_parameters()
    
    def _reset_parameters(self):
        # 使用 Xavier 均匀初始化
        nn.init.xavier_uniform_(self.att)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = F.softmax(self.att, dim=-1)
        return torch.matmul(attention_weights, x)


class RelationalCrossAttention(nn.Module):
    def __init__(self, D1, D2, num_node, is_global=True, num_heads=4, dropout=0.1, ff_ratio=2, act = "gelu", node_cv_num=5):
        super().__init__()
        self.D1 = D1
        self.num_heads = num_heads
        self.dropout = dropout
        self.processP = nn.Linear(D2, D2)
        # self.processP2 = nn.Linear(716 * D2, D2)
        # 投影 P -> K, V
        self.is_global = is_global
        self.node_cv_num = node_cv_num
        if is_global:
            self.down = DownDim(num_node, num_node//100)
            self.sa = SelfAttention(D2, num_heads=num_heads, dropout=dropout)
        
        self.to_q = nn.Linear(D1, D1)
        self.to_k = nn.Linear(D2, D1)
        self.to_v = nn.Linear(D2, D1)
        self.norm_k = nn.LayerNorm(D1)
        self.norm_v = nn.LayerNorm(D1)

        # 用于输出残差和前馈的 LayerNorm
        self.attn_norm = nn.LayerNorm(D1)
        self.ffn_norm = nn.LayerNorm(D1)

        # 注意力后的通道注意力
        self.att1 = nn.Conv2d(in_channels=D1, out_channels=1, kernel_size=(1, 1), bias=True)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(D1, D1 * ff_ratio),
            get_activation(act),
            nn.Dropout(dropout),
            nn.Linear(D1 * ff_ratio, D1),
            nn.Dropout(dropout)
        )

    def forward(self, C, P:torch.Tensor, G:torch.Tensor):
        """
        C: (B, D1, N, T)
        P: (Ns+1, D2)
        """
        B, D1, N, T = C.shape
        # P = self.processP(P)
        # P = P.mean(dim=0, keepdim=True)
        if self.is_global:
            P = self.down(P)
            P = torch.cat([G,P], dim=0)
            P = self.sa(P)  
            P = P[:self.node_cv_num, :]
        Ns, D2 = P.shape
        
        # 1) Query Q: (B, N*T, D1)
        Q = C.permute(0, 2, 3, 1).reshape(B, N * T, D1)
        # Q = self.to_q(Q)

        # 2) K/V 从 P 投影 + Norm, 扩展到 batch 维度 -> (B, Ns, D1)
        K = self.norm_k(self.to_k(P))  # (Ns, D1)
        V = self.norm_v(self.to_v(P))  # (Ns, D1)
        
        K = K.unsqueeze(0).expand(B, Ns, D1)
        V = V.unsqueeze(0).expand(B, Ns, D1)

        # 3) FlashAttention 调用: q/k/v 形状 (B, L, D)
        # 输出 attn_out: (B, N*T, D1)
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout,
            is_causal=False
        )

        # 4) 恢复形状 (B, D1, N, T)
        attn_out = attn_out.reshape(B, N, T, D1).permute(0, 3, 1, 2)

        # 5) 残差 + LayerNorm
        C_res = self.attn_norm((C + attn_out).permute(0, 2, 3, 1))  # (B, N, T, D1)
        C_res = C_res.permute(0, 3, 1, 2)  # (B, D1, N, T)

        # 6) 前馈网络
        ffn_in = C_res.permute(0, 2, 3, 1)  # (B, N, T, D1)
        ffn_out = self.ffn(ffn_in)         # (B, N, T, D1)
        ffn_out = self.ffn_norm(ffn_in + ffn_out)
        ffn_out = ffn_out.permute(0, 3, 1, 2)  # (B, D1, N, T)

        # 7) 通道注意力加权
        # out = ffn_out * torch.sigmoid(self.att1(C_res))
        out = ffn_out + C
        return out  # (B, D1, N, T)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, hidden_dim, num_heads=4, dropout=0.1, act = "gelu"):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        # Projections for query, key, value
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj   = nn.Linear(context_dim, hidden_dim)
        self.value_proj = nn.Linear(context_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context, mask=None):
        """
        query:   (B, Nq, query_dim)
        context: (B, Nk, context_dim)
        mask:    (B, Nq, Nk) or (B, 1, Nk), optional
        """
        query = query.unsqueeze(0) 
        context = context.unsqueeze(0)
        B, Nq, _ = query.size()
        Nk = context.size(1)

        # Project to multi-head Q, K, V
        Q = self.query_proj(query).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Nq, D)
        K = self.key_proj(context).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Nk, D)
        V = self.value_proj(context).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Nk, D)

        # Attention: scaled dot-product
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, Nq, Nk)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)  # (B, H, Nq, D)
        output = output.transpose(1, 2).contiguous().view(B, Nq, self.hidden_dim)  # (B, Nq, hidden_dim)

        output = self.out_proj(output)  # (B, Nq, query_dim)
        return output + query

import torch
from torch import nn
import torch.nn.functional as F
from models.layer import RelationalCrossAttention, CrossAttention, get_activation
from models.vis import CVModel  

class ContentAgree(nn.Module):
    def __init__(self, in_dim, num_heads=4, dropout=0.1,ff_ratio=2, activation = "gelu"):
        super().__init__()
        self.in_dim = in_dim
        self.heads = num_heads
        self.heads_dim = in_dim // num_heads
        assert in_dim % num_heads == 0, "in_dim must be divisible by heads"
        
        self.dropout = dropout
        
        self.query_proj1 = nn.Linear(in_dim, in_dim)
        self.key_proj1   = nn.Linear(in_dim, in_dim) 
        self.value_proj1 = nn.Linear(in_dim, in_dim)
        
        self.newc_proj = nn.Linear(in_dim, in_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(self.in_dim, self.in_dim)

        self.ffn = nn.Sequential(
            nn.Linear(in_dim, in_dim * ff_ratio),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(in_dim * ff_ratio, in_dim),
            nn.Dropout(dropout)
        )
        self.attn_norm = nn.LayerNorm(in_dim)
        self.ffn_norm = nn.LayerNorm(in_dim)
        
    def forward(self, x:torch.Tensor, C:torch.Tensor, G:torch.Tensor):
        """
        Args:
            x: Input tensor of shape (batch_size, dim, num_nodes, 1)
            C: Content tensor of shape (batch_size, num_nodes, dim)
        """
        B, dim, N, _ = x.shape
        k = C.shape[2]
        # k, N = G.shape
        # C = C.expand(B, -1, -1)  # Expand C to match batch size
        # G = G.expand(1, 1, -1, -1)
        
        x = x.permute(0, 2, 3, 1).reshape(B, N, dim)
        K1 = self.query_proj1(C).view(B, k, self.heads, self.heads_dim).permute(0, 2, 1, 3) 
        Q1 = self.key_proj1(x).view(B, N, self.heads, self.heads_dim).permute(0, 2, 1, 3)
        V1 = self.value_proj1(x).view(B, N, self.heads, self.heads_dim).permute(0, 2, 1, 3)
        attn_scores = torch.einsum('bhnd,bhkd->bhnk', Q1, K1) / ((self.heads_dim ** 0.5))
        attn1 = self.attn_dropout(F.softmax(attn_scores, dim=-1))
        
        attn2 = torch.einsum('bhnd,bhkd->bhnk', K1, K1) / ((self.heads_dim ** 0.5))
        attn2 = self.attn_dropout(F.softmax(attn2, dim=-1))
        
        w1 = torch.einsum('bhnk,bhkd->bhnd',attn1, attn2)
        
        attn3 = self.attn_dropout(F.softmax(attn_scores.transpose(-2, -1), dim=-1))
        
        w2 = torch.einsum('bhkn,bhnd->bhkd', attn3, V1)

        out = torch.einsum('bhnk,bhkd->bhnd', w1, w2)

        out = out.permute(0, 3, 1, 2).reshape(B, dim, N) .contiguous()  # [B, dim, N, 1]
        ffn_in = out.permute(0, 2, 1)  # [B, N, 1, dim]
        ffn_in = self.attn_norm(ffn_in + x)  # residual connection
        ffn_out = self.ffn(ffn_in)  # [B, N, 1, dim]
        ffn_out = self.ffn_norm(ffn_in + ffn_out)
        ffn_out = ffn_out.permute(0, 2, 1).unsqueeze(3) 

        return ffn_out
    
class norm(nn.Module):
    def __init__(self, hidden_dim):
        super(norm, self).__init__()
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x_permuted = x.permute(0, 2, 3, 1) 
        x_normed = self.ln(x_permuted)
        out = x_normed.permute(0, 3, 1, 2)  
        return out
    
class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim, dropout,act = "gelu") -> None:
        super().__init__()
        # self.ln = nn.LayerNorm(input_dim)
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = get_activation(act)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden

class Block(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1, act = "gelu", relation_patterns=True) -> None:
        super().__init__()
        self.mlp = MultiLayerPerceptron(input_dim, hidden_dim, dropout, act = act)
        # self.gcn = gcn(input_dim, hidden_dim, supports_len = supports_len, dropout=dropout)
        self.relation_patterns = relation_patterns
        if relation_patterns:
            self.gcn = ContentAgree(input_dim, num_heads=2, dropout=dropout, activation= act)
        else:
            self.gcn = MultiLayerPerceptron(input_dim, hidden_dim, dropout, act = act)
        self.atten1 = nn.Conv2d(
            in_channels=hidden_dim,  out_channels=1, kernel_size=(1, 1), bias=True)
        self.atten2 = nn.Conv2d(
            in_channels=hidden_dim,  out_channels=1, kernel_size=(1, 1), bias=True)
        self.gate_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=True)
        self.act  = get_activation(act)
        self.ln = norm(hidden_dim)
        self.mlp2 = MultiLayerPerceptron(hidden_dim, hidden_dim, dropout)
        
    def forward(self, x, A, G):
        x_mlp = self.mlp(x)
        if self.relation_patterns:
            x_gcn = self.gcn(x, A, G)
        else:
            x_gcn = self.gcn(x)
        # x = x + x_mlp* F.sigmoid(self.atten1(x)) + x_gcn * F.sigmoid(self.atten2(x))

         # Compute GRU-style gate (value between 0 and 1)
        gate = torch.sigmoid(torch.mean(self.gate_conv(x), dim=(2, 3), keepdim=True))  # Shape: [B, C, N, T]

        # Gated fusion
        x_fused = gate * x_gcn + (1 - gate) * x_mlp  # Like GRU's update gate

        # Add residual and normalize
        x = self.ln(self.act(x + x_fused))
        x = self.mlp2(x)  # Apply MLP after normalization
        
        return x

class STID(nn.Module):
    def __init__(
        self,
        device,
        num_nodes: int,
        node_dim=32,
        input_len=12,
        in_dim=1,
        hidden_dim=32,
        cvrelation_dim=32,
        out_dim=12,
        layers=4,
        support_num=0,
        dropout=0.3,
        tod_dims=32,
        dow_dims=32,
        tod=96,
        dow=7,
        partten_dim=7,
        if_T_i_D=True,
        if_D_i_W=True,
        if_node=True,
        gcn_bool=True,
        cv_token=True,
        cv_pattern = True,
        relation_patterns = True,
        st_encoder = True,
        node_cv_num = 5,
        lc = None,
        if_lc=True,
        act = "GELU",
        **kwargs
    ):
        super().__init__()
        # attributes
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.input_len = input_len
        self.input_dim = in_dim
        self.embed_dim = hidden_dim
        self.cvrelation_dim = cvrelation_dim
        self.output_len = out_dim
        self.num_layer = layers
        self.support_num = support_num
        self.dropout = dropout
        
        self.temp_dim_tid = tod_dims
        self.temp_dim_diw = dow_dims
        self.time_of_day_size = tod
        self.day_of_week_size = dow

        self.if_time_in_day = if_T_i_D
        self.if_day_in_week = if_D_i_W
        self.if_node = if_node
        self.gcn_bool = gcn_bool
        # cv arguments
        self.cv_token = cv_token
        self.cv_pattern = cv_pattern  
        self.lc = lc
        self.if_lc = if_lc
        self.lc_dim = node_dim
        self.partten_dim = partten_dim
        self.relation_patterns = relation_patterns
        self.st_encoder = st_encoder
        self.node_cv_num = node_cv_num
        self.act = act
        
        if self.if_lc:
            self.lc_conv = nn.Conv2d(
            in_channels=2, out_channels=self.lc_dim, kernel_size=(1, 1), bias=True)
        
        # node embeddings
        if self.if_node:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        if self.gcn_bool:
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10))
            self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes))


        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # ekoding
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_node)+self.temp_dim_tid*int(self.if_time_in_day) + \
            self.temp_dim_diw*int(self.if_day_in_week) + int(self.if_lc)*self.lc_dim
            
        self.C = nn.Parameter(torch.empty(self.partten_dim, self.hidden_dim))
        nn.init.xavier_uniform_(self.C)
        
        self.G = nn.Parameter(torch.empty(self.partten_dim, self.num_nodes))
        nn.init.xavier_uniform_(self.G)
        
        # if self.cv_token and not self.cv_rela:
        #     self.two_node_sp_emb = nn.Parameter(torch.empty(num_nodes, node_dims))
        #     nn.init.xavier_uniform_(self.two_node_sp_emb)
        #     lc = lc.T  # 转置为 [N, 2]，方便计算
        #     # 计算距离矩阵 [N, N]
        #     diff = lc.unsqueeze(1) - lc.unsqueeze(0)  # [N, 1, 2] - [1, N, 2] => [N, N, 2]
        #     dist_matrix = torch.norm(diff, dim=2)     # [N, N]
        #     dist_matrix.fill_diagonal_(float('inf'))# 将对角线（自身距离）设为无穷大，避免选到自己
        #     self.nearest_indices = torch.argmin(dist_matrix, dim=1)# 每个节点最近邻的下标 [N]
        #     self.attn = RelationalCrossAttention(self.hidden_dim, 3*node_dims, self.num_nodes) 
        #     self.cv_global = nn.Parameter(torch.empty(1, 3*node_dims))
        if self.cv_token:
            self.attn = RelationalCrossAttention(self.hidden_dim, node_dim + self.cvrelation_dim, self.num_nodes, act = self.act, node_cv_num = self.node_cv_num) 
            self.cv_global = nn.Parameter(torch.empty(self.node_cv_num, node_dim + self.cvrelation_dim))
            nn.init.xavier_uniform_(self.cv_global)
        
        self.first_mlp = MultiLayerPerceptron(self.hidden_dim, self.hidden_dim, dropout)
        
        self.update_C = CrossAttention(self.hidden_dim, 2*node_dim + self.embed_dim*3, self.hidden_dim, act = self.act) 
        
        
        self.ekoder = nn.ModuleList(
            [Block(self.hidden_dim, self.hidden_dim, dropout=self.dropout, act = self.act, relation_patterns = self.relation_patterns) if self.st_encoder else MultiLayerPerceptron(self.hidden_dim, self.hidden_dim, dropout=self.dropout, act = self.act)  for _ in range(self.num_layer)])
        
        self.parallel = nn.Conv2d(in_channels=self.hidden_dim*(self.num_layer+1), out_channels=self.hidden_dim, kernel_size=(1, 1), bias=True)
        
        self.reslayer = nn.Conv2d(in_channels=self.embed_dim, out_channels=self.hidden_dim, kernel_size=(1, 1), bias=True)

        # regression
        self.mlayer = MultiLayerPerceptron(self.hidden_dim, self.hidden_dim, dropout)
        self.regression_layer = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
    
    def forward(self, history_data: torch.Tensor, TE, cv_supports, node_cv_feature=None, vision_rela=None) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """
        B, L, N, C = history_data.shape
        history_data = history_data.transpose(1, 3)
        TE = TE.transpose(1, 3)
        # prepare data
        input_data = history_data[..., range(self.input_dim)]

        if self.if_time_in_day:
            t_i_d_data = TE[..., 0].long()
            # In the datasets used in STID, the time_of_day feature is normalized to [0, 1]. We multiply it by 288 to get the index.
            # If you use other datasets, you may need to change this line.
            time_in_day_emb = self.time_in_day_emb[t_i_d_data[:, -1, :].type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = TE[..., 1].long()
            day_in_week_emb = self.day_in_week_emb[d_i_w_data[:, -1, :].type(torch.LongTensor)]
        else:
            day_in_week_emb = None
            
        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        node_emb = []
        if self.if_node:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
            
        if self.if_lc:
            lc_emb = self.lc.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(-1)
            lc_emb = self.lc_conv(lc_emb)
        else:
            lc_emb = None
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))
        
        supports = [] 
        if self.gcn_bool:
            supports.append(F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1))
        if self.support_num:
            supports = supports + list(cv_supports)

        # if self.cv_token and cv_relation is None:
        #     nei_emb = self.node_emb[self.nearest_indices]
        #     two_emb = self.two_node_sp_emb 
        #     new_two_node_sp_emb = torch.cat([two_emb, nei_emb, self.node_emb], dim=1)  # [N, hidden_dim+2*node_dims]
        #     new_two_node_sp_emb = torch.cat([self.cv_global, new_two_node_sp_emb], dim=0)  # [N+1, hidden_dim+2*node_dims]
        if  self.cv_token and node_cv_feature is not None:
            new_two_node_sp_emb = torch.cat([node_cv_feature, self.node_emb], dim=1)
            # new_two_node_sp_emb = torch.cat([self.cv_global, new_two_node_sp_emb], dim=0)
        
        # cokate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb + [lc_emb], dim=1)
        hidden = self.first_mlp(hidden)
        hidden = self.attn(hidden, new_two_node_sp_emb, self.cv_global) if self.cv_token else hidden
        
        if self.cv_pattern and self.relation_patterns:
            new_C = self.update_C(self.C, vision_rela)
        else:
            new_C  = self.C
        
        new_G = F.softmax(self.G, dim=-1)
        new_C = new_C.repeat(B, L, 1, 1)
        temps = [hidden, ]
        for fuk in self.ekoder:
            if self.st_encoder:
                temp = fuk(hidden, new_C, new_G)
            else:
                temp = fuk(hidden)
            temps.append(temp)
        
        hidden = torch.cat(temps, dim=1)  # [B, hidden_dim*(num_layer+1), N, 1]
        hidden = self.parallel(hidden)

        hidden = self.mlayer(hidden)
        hidden = self.reslayer(time_series_emb) + hidden
        
        prediction = self.regression_layer(hidden)

        return prediction
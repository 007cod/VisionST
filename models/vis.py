import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.feature_extraction import create_feature_extractor

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出 shape: (B, C, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.size()
        y = self.avg_pool(x).view(B, C)           # squeeze
        y = self.fc(y).view(B, C, 1, 1)           # excitation
        return x * y.expand_as(x)                 # scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        x_cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        attn = self.sigmoid(self.conv(x_cat))  # [B, 1, H, W]
        return x * attn  # Element-wise multiplication


def up_block(in_channels, out_channels, scale_factor):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ConvTranspose2d(out_channels, out_channels, kernel_size=scale_factor, stride=scale_factor),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
        
class Resnet(nn.Module):
    def __init__(self, pretrained=True, hidden_dim=64):
        super(Resnet, self).__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        return_nodes = {'layer1': 'feat1', 'layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4'}
        self.extractor = create_feature_extractor(backbone, return_nodes=return_nodes)

        self.up2 = up_block(128, 64, 2)
        self.up3 = up_block(256, 64, 4)
        self.up4 = up_block(512, 64, 8)

        self.sas = nn.ModuleList([SpatialAttention() for _ in range(4)])
        self.fusion = nn.Sequential(
            nn.Conv2d(64 * 4, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.extractor(x)
        x1 = self.sas[0](out["feat1"])
        x2 = self.sas[1](self.up2(out["feat2"]))
        x3 = self.sas[2](self.up3(out["feat3"]))
        x4 = self.sas[3](self.up4(out["feat4"]))
        return self.fusion(torch.cat([x1, x2, x3, x4], dim=1))

class CVModel(nn.Module):
    def __init__(self, device, position_relat, img_h, img_w, node_dim=32,
                 hidden_dim = 32, cv_fea_dim=32, support_num=5, backbone='resnet'):
        super(CVModel, self).__init__()
        
        self.position_relat = position_relat
        num_nodes = position_relat.size(0)
        self.num_nodes = num_nodes
        self.support_num = support_num
        self.device = device
        self.hidden_dim = hidden_dim
        self.cv_fea_dim = cv_fea_dim
        self.node_dim = node_dim
        N = self.position_relat.size(0)
        self.img_h = img_h
        self.img_w = img_w

        self.x_idx = (self.position_relat[:, :, 0] / 0.05 *img_h).long().to(device)
        self.y_idx = (self.position_relat[:, :, 1] / 0.05 *img_w).long().to(device)

        self.mask = (self.x_idx > 0) & (self.x_idx < img_h) & (self.y_idx >0) & (self.y_idx < img_w)
        self.mask = self.mask.to(device)
        
        self.num_vis_node = [0, ]
        for i in range(N):
            self.num_vis_node.append(self.num_vis_node[-1] + torch.sum(self.mask[i, :]).item())
        self.num_vis_node = torch.tensor(self.num_vis_node, device=device)
        
        self.i_idx = torch.arange(N, device=device).view(1, N).expand(N, N).to(device)

        #  ResNet
        if backbone == "resnet":
            self.backbone = Resnet(pretrained=True, hidden_dim=self.hidden_dim)
 
        self.outconv = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)        
        self.atten = nn.Sequential(nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                                   nn.AdaptiveMaxPool2d((1, 1)),)
        
        self.center_conv = nn.Conv2d(self.hidden_dim, self.cv_fea_dim, kernel_size=1)
        self.ADP = nn.AdaptiveAvgPool2d((1,1))
        
        self.two_relate_conv = nn.Linear(self.hidden_dim, self.cv_fea_dim)
        
        self.C_atten = ChannelAttention(self.hidden_dim*3 + self.node_dim*2)
        self.c_nor = nn.LayerNorm(self.hidden_dim*3 + self.node_dim*2)
        
        self.agre = nn.Sequential(nn.Conv2d(self.hidden_dim*3 + self.node_dim*2, self.support_num, kernel_size=1, padding=0),
                                    nn.ReLU(),)
    
    def forward(self, x, start_idx=0, end_idx=1, node_emb=None):
        z = self.backbone(x)
        out = F.relu(self.outconv(z))
        b, c, h,w = out.size()
        
        # center_feature_only = out[:,:,h//2:h//2+1, w//2:w//2 +1]
        # center_feature_only = self.center_conv(center_feature_only)
        center_feature_only = F.normalize(self.center_conv(self.ADP(out)),dim=1)
        
        out = out.permute(0, 2, 3, 1)
        
        support, vis_relat_fea = self.process_img_feac(out, start_idx, end_idx, node_emb = node_emb)
        # vis_relat_fea = self.two_relate_conv(vis_relat_fea)
        
        return out, center_feature_only, support, vis_relat_fea  # [b, h, w, c], [b, 1, 1, d]
    
    def process_img_feac(self, cvfeature, start_idx, end_idx, node_emb=None):
        b, h, w, c = cvfeature.size()
        device = self.device
        self.x_idx = (self.x_idx /self.img_h * h).long()
        self.y_idx = (self.y_idx /self.img_w * w).long()
        
        kk = self.get_global(cvfeature, start_idx, end_idx)
        n = kk.size(0)
        center_feature_only = cvfeature[:,h//2:h//2+1, w//2:w//2 +1, :]
        center_feature_only = center_feature_only.squeeze(1).squeeze(1)
        
        ids = torch.arange(start_idx, end_idx)
        i_idx = torch.arange(b, device=self.device).view(b, 1).expand(b, self.num_nodes)
        
        masks = self.mask[ids]
        node_id = torch.nonzero(masks).long() 
        node_id[:, 0] = node_id[:, 0] + start_idx  
        
        i_indices = i_idx[masks]
        x_indices = self.x_idx[ids][masks]
        y_indices = self.y_idx[ids][masks]
        
        center_feature_only = center_feature_only[i_indices]
        
        indice = torch.nonzero(masks, as_tuple=True)  
        
        # support = 1 + support
        if node_emb is not None:
            support_add = torch.cat([kk, center_feature_only, cvfeature[i_indices, x_indices, y_indices, :], node_emb[node_id[:,0],...], node_emb[node_id[:,1],...]], dim=1)
        else:
            support_add = torch.cat([kk, center_feature_only, cvfeature[i_indices, x_indices, y_indices, :]], dim=1)
        vis_relat_fea = support_add.view(n, -1, 1, 1).contiguous()
        vis_relat_fea = self.C_atten(vis_relat_fea)
        if self.support_num == 0:
            return None, vis_relat_fea.view(n, -1)
        
        support_add = self.agre(vis_relat_fea)
        support_add = support_add.squeeze(-1).squeeze(-1)
        # support_add = F.softmax(support_add, dim=1)
        
        support = torch.zeros((b, self.num_nodes, self.support_num), device=device)
        
        support[masks] =  support[masks] + support_add 
        # print(torch.max(support))
        # print(torch.min(support))
        support = F.softmax(support, dim=1)  

        return support, vis_relat_fea.view(n, -1)
    
    def get_cv_relation(self, cvfeature, start_idx, end_idx):
        # cvfeature = cvfeature.permute(0, 3, 1, 2)  # b, c, h, w
        b, h, w, c = cvfeature.size()
        device = self.device
        self.x_idx = (self.x_idx /self.img_h * h).long()
        self.y_idx = (self.y_idx /self.img_w * w).long()
        
        kk = self.get_global(cvfeature, start_idx, end_idx)
        n = kk.size(0)
        center_feature_only = cvfeature[:,h//2:h//2+1, w//2:w//2 +1, :]
        center_feature_only = center_feature_only.squeeze()
        
        ids = torch.arange(start_idx, end_idx)
        i_idx = torch.arange(b, device=self.device).view(b, 1).expand(b, self.num_nodes)
        
        masks = self.mask[ids]
        i_indices = i_idx[masks]
        x_indices = self.x_idx[ids][masks]
        y_indices = self.y_idx[ids][masks]
        
        center_feature_only = center_feature_only[i_indices]
        
        indice = torch.nonzero(masks, as_tuple=True)  
        support = torch.zeros((b, self.num_nodes, self.support_num), device=device)
        # support = 1 + support
        support_add = torch.cat([kk, center_feature_only, cvfeature[i_indices, x_indices, y_indices, :]], dim=1)
        support_add = support_add.view(n, -1, 1, 1).contiguous()
        support_add = self.agre(support_add)
        support_add = support_add.squeeze(-1).squeeze(-1)
        
        support[masks] =  support[masks] + support_add 
        # print(torch.max(support))
        # print(torch.min(support))
        support = F.softmax(support, dim=1)  

        return support
    
    def get_global(self, cvfeature, start_idx, end_idx):
        cvfeature = cvfeature.permute(0, 3, 1, 2)  # b, c, h, w
        b, c, h, w = cvfeature.size()
        device = self.device
        
        ids = torch.arange(start_idx, end_idx, device=device)
        i_idx = torch.arange(b, device=device).view(b, 1).expand(b, self.num_nodes)
        
        masks = self.mask[ids]
        i_indices = i_idx[masks]
        x_indices = self.x_idx[ids][masks]
        y_indices = self.y_idx[ids][masks]

        x_1 = torch.clamp(x_indices - (h // 2 - x_indices)//2, min=0, max=h - 1)
        x_2 = torch.clamp(h // 2 - (x_indices - h // 2)//2, min=0, max=h - 1)
        xx = torch.stack([x_1, x_2], dim=1).sort(dim=1)[0]

        y_1 = torch.clamp(y_indices - (w // 2 - y_indices)//2, min=0, max=w - 1)
        y_2 = torch.clamp(w // 2 - (y_indices - w // 2)//2, min=0, max=w - 1)
        yy = torch.stack([y_1, y_2], dim=1).sort(dim=1)[0]

        desired_size = (h//4, w//4) 
        batch_features = []
        for i in range(len(i_indices)):
            x_start, x_end = xx[i, 0], xx[i, 1]
            x_end = torch.max(x_start+1, x_end)
            y_start, y_end = yy[i, 0], yy[i, 1]
            y_end = torch.max(y_start+1, y_end) 
            region = cvfeature[i_indices[i], :, x_start:x_end, y_start:y_end]
            region = F.interpolate(region.unsqueeze(0), size=desired_size, mode='bilinear', align_corners=False).squeeze(0)
            batch_features.append(region)
        
        padded_features = torch.stack(batch_features)
        
        kk = self.atten(padded_features)
        
        kk = kk.permute(0, 2, 3, 1)
        kk = kk.view(kk.size(0),-1)


        return kk

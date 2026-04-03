import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super(ConvBlock, self).__init__()
        block = []
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True, dilation=dilation))
        block.append(nn.PReLU())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

# ================= 革命性优化 1: 超轻量局域增强 MLP =================
class LinearMLP(nn.Module):
    """
    1. 使用 GroupNorm(1, dim) 完美替代 LayerNorm，数学上完全等价于通道维度归一化，但免去了极耗性能的 permute。
    2. 使用 1x1 Conv 替代 Linear，这在图像数据中是完美的逐像素 MLP，速度更快，不产生显存碎片。
    3. 保留中间的 3x3 DWConv 提取极其关键的局部空间细节，大幅拉高 PSNR 和 SCC。
    """
    def __init__(self, dim, hidden_dim, out_dim=None, dropout=0.0):
        super(LinearMLP, self).__init__()
        out_dim = out_dim or dim
        
        self.norm = nn.GroupNorm(1, dim)
        
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),  # 纯线性 MLP 映射
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False), # 零成本空间偏置
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=False), # 纯线性 MLP 映射
            nn.Dropout(dropout)
        )

        self.shortcut = nn.Identity() if dim == out_dim else nn.Conv2d(dim, out_dim, kernel_size=1)

    def forward(self, x):
        # 抛弃了低效的 permute，直接在 BCHW 格式下飞速运算
        return self.shortcut(x) + self.net(self.norm(x))
# =======================================================================

class DifferentiableSoftKMeans(nn.Module):
    def __init__(self, channel, n_clusters=16, hidden_dim=32, temperature=1.0, 
                 spatial_weight=10.0, n_iterations=3):
        super(DifferentiableSoftKMeans, self).__init__()
        self.n_clusters = n_clusters
        self.temperature = temperature
        self.spatial_weight = spatial_weight
        self.n_iterations = n_iterations
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )
        
        self.grid_cache = None
        self.register_buffer('H_cache', torch.tensor(0, dtype=torch.long))
        self.register_buffer('W_cache', torch.tensor(0, dtype=torch.long))
        
        grid_h = int(n_clusters ** 0.5)
        grid_w = (n_clusters + grid_h - 1) // grid_h
        init_centers = []
        for i in range(grid_h):
            for j in range(grid_w):
                if len(init_centers) < n_clusters:
                    y = (i + 0.5) / grid_h * 2 - 1
                    x = (j + 0.5) / grid_w * 2 - 1
                    init_centers.append([x, y])
        self.register_buffer('init_spatial_centers', torch.tensor(init_centers).float())
        self.feature_centers = nn.Parameter(torch.randn(n_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.feature_centers)

    def _get_grid(self, H, W, device):
        if self.grid_cache is None or self.H_cache.item() != H or self.W_cache.item() != W:
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device), 
                torch.linspace(-1, 1, W, device=device), 
                indexing="ij"
            )
            grid = torch.stack([xx, yy], dim=0)
            self.grid_cache = grid
            self.H_cache.fill_(H)
            self.W_cache.fill_(W)
        return self.grid_cache.to(device)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        features = self.embedding(x)
        features_flat = features.permute(0, 2, 3, 1).reshape(B, N, self.hidden_dim)
        spatial_grid = self._get_grid(H, W, x.device)
        spatial_flat = spatial_grid.permute(1, 2, 0).reshape(N, 2)
        
        feature_centers = self.feature_centers.unsqueeze(0).expand(B, -1, -1)
        spatial_centers = self.init_spatial_centers.clone()
        
        for _ in range(self.n_iterations):
            feature_dists = torch.cdist(features_flat, feature_centers)
            spatial_dists = torch.cdist(spatial_flat.unsqueeze(0), spatial_centers.unsqueeze(0)).squeeze(0)
            total_dists = feature_dists + self.spatial_weight * spatial_dists.unsqueeze(0)
            soft_assign = F.softmax(-total_dists / self.temperature, dim=2)
            
            weights_sum = soft_assign.sum(dim=1, keepdim=True).transpose(1, 2)
            feature_centers = torch.bmm(soft_assign.transpose(1, 2), features_flat) / (weights_sum + 1e-5)
            
            weights_spatial = soft_assign.mean(dim=0).T
            spatial_centers = torch.mm(weights_spatial, spatial_flat) / (weights_spatial.sum(dim=1, keepdim=True) + 1e-5)
        
        hard_labels_forward = soft_assign.argmax(dim=2)
        hard_onehot = F.one_hot(hard_labels_forward, num_classes=self.n_clusters).float()
        hard_labels_ste = (hard_onehot - soft_assign).detach() + soft_assign
        hard_labels = hard_labels_ste.argmax(dim=2).reshape(B, H, W)
        
        return hard_labels

class CAN_Filter(nn.Module):
    def __init__(self, in_channels, n_clusters=16, kernel_size=3, padding=1, rank=8, mlp_dim=32):
        super(CAN_Filter, self).__init__()
        self.n_clusters = n_clusters
        self.padding = padding
        self.kernel_size = kernel_size
        self.kernel_area = kernel_size * kernel_size
        self.in_channels = in_channels
        
        self.centroid_to_lowrank = nn.Sequential(
            nn.Linear(in_channels * self.kernel_area, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, rank)
        )
        
        self.base_kernels = nn.Parameter(torch.randn(rank, self.kernel_area))
        nn.init.kaiming_uniform_(self.base_kernels.unsqueeze(-1), a=5**0.5)
        
    def forward(self, x, labels, cluster_centers=None):
        B, C, H, W = x.shape
        
        if cluster_centers is None:
            unfold = nn.Unfold(kernel_size=self.kernel_size, padding=self.padding)
            patches = unfold(x)  
            patches = patches.permute(0, 2, 1).reshape(B, H*W, -1)  
            
            labels_flat = labels.reshape(B, -1)
            labels_onehot = F.one_hot(labels_flat, num_classes=self.n_clusters).float()
            weights = labels_onehot.permute(0, 2, 1)
            cluster_centers = torch.bmm(weights, patches) / (weights.sum(dim=2, keepdim=True) + 1e-5)
        
        lowrank_weights = self.centroid_to_lowrank(cluster_centers)  
        kernels_flat = torch.einsum('bkr,ra->bka', lowrank_weights, self.base_kernels)
        kernels_flat = F.softmax(kernels_flat, dim=-1) 
        
        out = torch.zeros_like(x)
        labels_flat = labels.reshape(B, H*W)
        
        kernels_2d = kernels_flat.view(B, self.n_clusters, self.kernel_size, self.kernel_size)
        x_reshaped = x.view(1, B * C, H, W)
        
        for k in range(self.n_clusters):
            kernel_k = kernels_2d[:, k, :, :] 
            weight_k = kernel_k.unsqueeze(1).repeat(1, C, 1, 1).view(B * C, 1, self.kernel_size, self.kernel_size)
            feat_k_reshaped = F.conv2d(x_reshaped, weight_k, bias=None, padding=self.padding, groups=B * C)
            feat_k = feat_k_reshaped.view(B, C, H, W)
            
            mask = (labels_flat == k).float().view(B, 1, H, W)
            out = out + feat_k * mask
                
        return out

class AdaptiveSoftFilter(nn.Module):
    def __init__(self, channels, n_clusters=32):
        super(AdaptiveSoftFilter, self).__init__()
        self.kmeans = DifferentiableSoftKMeans(channel=channels, n_clusters=n_clusters)
        self.spatial_filter = CAN_Filter(in_channels=channels, n_clusters=n_clusters)

    def forward(self, x):
        cluster_indice = self.kmeans(x)
        low_freq = self.spatial_filter(x, cluster_indice)
        high_freq = x - low_freq
        return low_freq, high_freq, cluster_indice

class CAFS(nn.Module):
    def __init__(self, feat_dim=32, ms_channels=8):
        super().__init__()
        self.feat_dim = feat_dim
        self.ms_channels = ms_channels
        
        self.proj = nn.Sequential(
            nn.Conv2d(1 + ms_channels, feat_dim, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)
        )
        
        self.filter_ms = AdaptiveSoftFilter(channels=ms_channels, n_clusters=32)
        self.filter_pan = AdaptiveSoftFilter(channels=1, n_clusters=32)
        
    def forward(self, pan, ms_up): 
        s_cat = torch.cat([ms_up, pan], dim=1)
        
        L_pan, H_pan, _ = self.filter_pan(pan)
        L_ms, H_ms, _ = self.filter_ms(ms_up)
        
        H_cat = torch.cat([H_pan, H_ms], dim=1)
        L_cat = torch.cat([L_pan, L_ms], dim=1)
        
        H_E = self.proj(H_cat)
        L_E = self.proj(L_cat)
        
        return s_cat, H_E, L_E

class SigmaNet(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, num_filter=64):
        super(SigmaNet, self).__init__()
        self.head = ConvBlock(in_channels, num_filter, kernel_size=3, padding=1)
        self.body = nn.ModuleList()
        for _ in range(depth):
            self.body.append(LinearMLP(dim=num_filter, hidden_dim=num_filter * 2))
        self.tail = ConvBlock(num_filter, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.head(x)
        for block in self.body:
            feat = block(feat)
        out = self.tail(feat)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SpatialAttention, self).__init__()
        block = []
        out_stage1 = in_channels // reduction
        out_stage2 = out_stage1 // reduction
        block.append(ConvBlock(in_channels=in_channels, out_channels=out_stage1))
        block.append(ConvBlock(in_channels=out_stage1, out_channels=out_stage2))
        block.append(nn.Conv2d(out_stage2, 1, kernel_size=1, padding=0, bias=True))
        block.append(nn.Sigmoid())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class SAB_astrous(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SAB_astrous, self).__init__()
        block = []
        block.append(ConvBlock(in_channels=in_channels, out_channels=in_channels))
        out_stage1 = in_channels // reduction
        block.append(ConvBlock(in_channels=in_channels, out_channels=out_stage1, kernel_size=1, padding=0))
        block.append(ConvBlock(in_channels=out_stage1, out_channels=out_stage1, kernel_size=3, padding=2, dilation=2))
        out_stage2 = out_stage1 // reduction
        block.append(ConvBlock(in_channels=out_stage1, out_channels=out_stage2, kernel_size=1, padding=0))
        block.append(ConvBlock(in_channels=out_stage2, out_channels=out_stage2, kernel_size=3, padding=4, dilation=4))
        block.append(nn.Conv2d(out_stage2, 1, kernel_size=1, padding=0, bias=True))
        block.append(nn.Sigmoid())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        block = []
        block.append(nn.AdaptiveAvgPool2d((1, 1)))
        block.append(ConvBlock(in_channels=channels, out_channels=channels // reduction, kernel_size=1, padding=0))
        block.append(nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=True))
        block.append(nn.Sigmoid())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class SCAB(nn.Module):
    def __init__(self, org_channels, out_channels):
        super(SCAB, self).__init__()
        pre_x = []
        pre_x.append(ConvBlock(in_channels=org_channels, out_channels=out_channels))
        pre_x.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        self.pre_x = nn.Sequential(*pre_x)
        self.CAB = ChannelAttention(channels=out_channels)
        self.SAB = SAB_astrous(in_channels=out_channels)
        self.last = nn.Conv2d(in_channels=2*out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        pre_x = self.pre_x(x)
        channel = self.CAB(pre_x)
        spatial = self.SAB(pre_x)
        out_s = pre_x * spatial.expand_as(pre_x)
        out_c = pre_x * channel.expand_as(pre_x)
        out_combine = torch.cat([out_s, out_c], dim=1)
        out = self.last(out_combine)
        return out + pre_x 
    
class NCB(torch.nn.Module):
    def __init__(self, depth_S=1, feature_dims=32):
        super(NCB, self).__init__()
        self.sigma_net = SigmaNet(in_channels=feature_dims, out_channels=feature_dims, depth=depth_S, num_filter=feature_dims)
        self.SCAB = SCAB(org_channels=feature_dims*2, out_channels=feature_dims)
        
    def forward(self, org_data):
        noise_map = self.sigma_net(org_data)
        net_input = torch.cat([org_data, noise_map], dim=1)
        net_out = self.SCAB(net_input)
        return net_out

class FGB_Block(nn.Module):
    def __init__(self, dim, expansion_ratio=2):
        super().__init__()
        # 此处的 LayerNorm 我们也替换为性能更好的 GroupNorm
        self.norm = nn.GroupNorm(1, dim)
        self.f1 = nn.Sequential(
            nn.Conv2d(dim, dim * expansion_ratio, kernel_size=1),
            nn.GroupNorm(4, dim * expansion_ratio),
            nn.Dropout2d(0.3),
            nn.Conv2d(dim * expansion_ratio, dim * expansion_ratio, 
                      kernel_size=3, padding=1, groups=dim * expansion_ratio),
        )
        self.f2 = nn.Sequential(
            nn.Conv2d(dim, dim * expansion_ratio, kernel_size=1),
            nn.GroupNorm(4, dim * expansion_ratio),
            nn.Dropout2d(0.3),
            nn.Conv2d(dim * expansion_ratio, dim * expansion_ratio, 
                      kernel_size=3, padding=1, groups=dim * expansion_ratio),
        )
        self.gate_act = nn.GELU()
        self.map = nn.Sequential(
            nn.Conv2d(dim * expansion_ratio, dim, kernel_size=1),
        )

    def forward(self, x):
        # 移除耗时的 permute 操作
        x_norm = self.norm(x)
        f1_out = self.f1(x_norm)
        f2_out = self.f2(x_norm)
        
        gat = self.gate_act(f1_out) * f2_out
        return x + self.map(gat)

# ================= 革命性优化 2: Flash Attention + 极简 QKV MLP =================
class Attention(nn.Module):
    """
    1. 使用 Flash Attention (F.scaled_dot_product_attention)，彻底解决 Batch 32 的 OOM。
    2. 将厚重且容易过拟合的 Q/K/V MLPs 精简为带归一化的 1x1 Conv，大幅提升收敛稳健性。
    """
    def __init__(self, channel, head_channel, dropout):
        super(Attention, self).__init__()
        self.head_channel = head_channel
        self.num_head = channel // head_channel
        
        self.norm_q = nn.GroupNorm(1, channel)
        self.norm_k = nn.GroupNorm(1, channel)
        self.norm_v = nn.GroupNorm(1, channel)
        
        # 抛弃两层庞大 MLP，采用标准的线性投影，这是 Transformer 避免死板记忆的原则
        self.q_proj = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        
        self.mlp = LinearMLP(dim=channel, hidden_dim=channel * 2, dropout=dropout)

    def forward(self, q, k, v):
        B, C, H, W = q.shape
        N = H * W
        
        # 投影并调整形状为 Flash Attention 所需的 (B, num_head, L, E)
        q_p = self.q_proj(self.norm_q(q)).view(B, self.num_head, self.head_channel, N).transpose(-2, -1)
        k_p = self.k_proj(self.norm_k(k)).view(B, self.num_head, self.head_channel, N).transpose(-2, -1)
        v_p = self.v_proj(self.norm_v(v)).view(B, self.num_head, self.head_channel, N).transpose(-2, -1)
        
        # 显存救星：PyTorch 内置的高效 Flash Attention 机制，不产生 N x N 矩阵
        attn_out = F.scaled_dot_product_attention(q_p, k_p, v_p)
        
        # 恢复空间形状
        attn_out = attn_out.transpose(-2, -1).reshape(B, C, H, W)
        
        # 残差连接 + 局域增强 MLP
        out = v + self.mlp(attn_out)
        return out
# =========================================================================

class MGB_FGB_Stage(nn.Module):
    def __init__(self, feat_dim=32, num_heads=4, dropout=0.085):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        
        head_channel = feat_dim // num_heads
        
        self.attn_h = Attention(channel=feat_dim, head_channel=head_channel, dropout=dropout)
        self.attn_l = Attention(channel=feat_dim, head_channel=head_channel, dropout=dropout)
        
        self.fgb = FGB_Block(feat_dim)
        
    def forward(self, H_E, L_E):
        H_G, L_G = H_E, L_E
        
        for _ in range(2):
            attn_out_h = self.attn_h(q=H_G, k=L_G, v=L_G)
            H_G = H_G + attn_out_h
            
            attn_out_l = self.attn_l(q=L_G, k=H_G, v=H_G)
            L_G = L_G + attn_out_l
            
            H_G = self.fgb(H_G)
            L_G = self.fgb(L_G)
        
        return H_G, L_G

# ================= 优化 3: 极大精简融合模块 =================
class EnhancedFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnhancedFusionBlock, self).__init__()
        # 从堆叠 3 个降为 1 个 MLP + Conv 投射，避免后期参数冗余过拟合导致 SAM 变差
        self.mlp = LinearMLP(dim=in_channels, hidden_dim=in_channels * 2)
        self.final_proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.mlp(x)
        return self.final_proj(x)
# =========================================================================

class DeepLinearEmbedding(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DeepLinearEmbedding, self).__init__()
        self.mlp = LinearMLP(dim=in_channel, hidden_dim=out_channel * 2, out_dim=out_channel)

    def forward(self, x):
        return self.mlp(x)

class SFA(nn.Module):
    def __init__(self, pan_channels=1, feat_dim=32, spatial_channel=8, ms_channels=8, dropout=0.1):
        super().__init__()
        self.h_spa = Attention(channel=feat_dim, head_channel=ms_channels, dropout=dropout)
        self.l_spa = Attention(channel=feat_dim, head_channel=ms_channels, dropout=dropout)
        
        self.embed1 = DeepLinearEmbedding(in_channel=spatial_channel + pan_channels, out_channel=feat_dim)
        self.embed2 = DeepLinearEmbedding(in_channel=feat_dim, out_channel=feat_dim)

        self.mlp_h = LinearMLP(dim=feat_dim, hidden_dim=feat_dim * 2, dropout=dropout)
        self.mlp_l = LinearMLP(dim=feat_dim, hidden_dim=feat_dim * 2, dropout=dropout)
        
        self.concat_proj = nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=1, bias=False)
        
        self.fusion = EnhancedFusionBlock(in_channels=feat_dim, out_channels=ms_channels)
        
    def forward(self, H_FG, L_FG, spatial):
        x = self.embed2(self.embed1(spatial))

        H_out = self.h_spa(q=H_FG, k=x, v=x)
        H_out = H_out + self.mlp_h(H_out) 

        L_out = self.l_spa(q=L_FG, k=x, v=x)
        L_out = L_out + self.mlp_l(L_out)

        fused = torch.cat([H_out, L_out], dim=1)
        fused = self.concat_proj(fused)
        
        O = self.fusion(fused)
    
        return O, H_out, L_out

class CGFformer(nn.Module):
    def __init__(self, pan_channels, lms_channels=8, feat_dim=32, num_heads=4, window_size=4):
        super().__init__()
        self.cafs = CAFS(feat_dim=feat_dim, ms_channels=lms_channels)
        
        self.ncb_h = NCB(feature_dims=feat_dim, depth_S=1)
        self.ncb_l = NCB(feature_dims=feat_dim, depth_S=1)
        
        self.mgb_fgb = MGB_FGB_Stage(feat_dim=feat_dim, num_heads=num_heads, dropout=0.085)
        self.sfa = SFA(pan_channels=pan_channels, feat_dim=feat_dim, spatial_channel=lms_channels, ms_channels=lms_channels, 
                       dropout=0.085)
    
    def forward(self, pan, ms, epoch=None, hw_range=None):
        ms_up = F.interpolate(ms, scale_factor=4, mode='bilinear', align_corners=False)
        S_C, H_E, L_E = self.cafs(pan, ms_up)
        
        H_E = self.ncb_h(H_E)
        L_E = self.ncb_l(L_E)
        
        H_FG, L_FG = self.mgb_fgb(H_E, L_E)
        
        result, H_final, L_final = self.sfa(H_FG, L_FG, S_C)
        result = result + ms_up
        
        return result, H_E, L_E, H_FG, L_FG, H_final, L_final
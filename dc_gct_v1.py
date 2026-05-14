import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath
from einops.einops import rearrange

from model.block.gcn_conv import Gcn_block
from model.block.graph_frames import Graph
from model.block.transformer import Attention, Mlp, Orthogonal_Cross_Fusion


# ==========================================
# 零参数生物力学躯干汇聚层
# ==========================================
class Anatomical_Pooling(nn.Module):
    """
    将四肢边缘的高风险特征强制收敛并注入稳定的躯干核心节点，
    增强局部图卷积在面对深层自遮挡时的抗畸变能力。
    """

    def __init__(self):
        super().__init__()
        self.r_leg = [2, 3]  # 右膝+右踝
        self.l_leg = [5, 6]  # 左膝+左踝
        self.l_arm = [12, 13]  # 左肘+左腕
        self.r_arm = [15, 16]  # 右肘+右腕

    def forward(self, x):
        out = x.clone()
        out[:, 1, :] = out[:, 1, :] + x[:, self.r_leg, :].mean(dim=1)  # 右髋
        out[:, 4, :] = out[:, 4, :] + x[:, self.l_leg, :].mean(dim=1)  # 左髋
        out[:, 11, :] = out[:, 11, :] + x[:, self.l_arm, :].mean(dim=1)  # 左肩
        out[:, 14, :] = out[:, 14, :] + x[:, self.r_arm, :].mean(dim=1)  # 右肩
        return out


class Local(nn.Module):
    def __init__(self, dim, h_dim, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.graph = Graph('hm36_gt', 'spatial', pad=0)

        # 使用 register_buffer 自动管理设备流转
        A_tensor = torch.tensor(self.graph.A, dtype=torch.float32)
        self.register_buffer('A', A_tensor)
        kernel_size = self.A.size(0)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 挂载解剖学先验池化模块
        self.anat_pool = Anatomical_Pooling()

        self.gcn1 = Gcn_block(in_channels=dim, out_channels=h_dim, kernel_size=kernel_size, residual=False)
        self.norm_gcn1 = norm_layer(dim)

        self.gcn2 = Gcn_block(in_channels=h_dim, out_channels=dim, kernel_size=kernel_size, residual=False)
        self.norm_gcn2 = norm_layer(dim)

    def forward(self, x):
        res = x

        # 前置生物学先验强制稳定末端节点
        x_pooled = self.anat_pool(x)

        x_gcn, _ = self.gcn1(self.norm_gcn1(x_pooled), self.A)
        x_gcn, _ = self.gcn2(x_gcn, self.A)

        x = res + self.drop_path(self.norm_gcn2(x_gcn))
        return x


class Global(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, length=1):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_attn = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, length=length)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm_attn(x)))
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, length=1):
        super().__init__()

        self.dim1 = dim1 = int(dim / 5)
        self.dim2 = dim2 = dim - dim1
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.local1 = Local(dim1, dim1 * 2, drop_path=drop_path, norm_layer=nn.LayerNorm)
        self.global1 = Global(dim1, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0.1, attn_drop=attn_drop,
                              drop_path=drop_path, norm_layer=nn.LayerNorm, length=length)

        self.norm_fusion = norm_layer(dim)


        self.fusion = Orthogonal_Cross_Fusion(
            in_features=dim,
            hidden_features=dim2,
            dim1=self.dim1,
            dim2=self.dim2,
            drop=drop
        )

        self.local2 = Local(dim2, dim2 * 2, drop_path=drop_path, norm_layer=nn.LayerNorm)
        self.global2 = Global(dim2, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0.1, attn_drop=attn_drop,
                              drop_path=drop_path, norm_layer=nn.LayerNorm, length=length)

        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim * 4, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim1, self.dim2], -1)

        x1 = self.local1(x1)
        x2 = self.global2(x2)

        x_fusion = torch.cat([x1, x2], -1)
        x_fusion_temp = x_fusion + self.fusion(self.norm_fusion(x_fusion))
        x_fusion_1, x_fusion_2 = torch.split(x_fusion_temp, [self.dim1, self.dim2], -1)

        x1 = self.global1(x1 + x_fusion_1)
        x2 = self.local2(x2 + x_fusion_2)

        x = torch.cat([x1, x2], -1) + x_fusion
        x = x + self.drop_path(self.mlp(self.norm_mlp(x)))
        return x


class DC_GCT(nn.Module):
    def __init__(self, args, depth=3, embed_dim=160, mlp_hidden_dim=1024, h=8, drop_rate=0.1, length=9):
        super().__init__()

        depth, embed_dim, mlp_hidden_dim, length = args.layers, args.channel, args.d_hid, args.frames
        self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints

        drop_path_rate = 0.3
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None

        self.patch_embed = nn.Linear(2, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_joints_in, embed_dim))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        dpr = [x.item() for x in torch.linspace(0.1, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, length=length)
            for i in range(depth)])
        self.Temporal_norm = norm_layer(embed_dim)

        self.fcn = nn.Linear(embed_dim, 3)

    def forward(self, x):
        x = rearrange(x, 'b f j c -> (b f) j c').contiguous()
        x = self.patch_embed(x)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.Temporal_norm(x)
        x = self.fcn(x)
        x = x.view(x.shape[0], -1, self.num_joints_out, x.shape[2])
        return x
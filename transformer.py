import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Orthogonal_Cross_Fusion(nn.Module):
    """
    基于单位方向向量的正交特征对齐融合 (Orthogonal Alignment Fusion)
    逐层解耦冗余特征，通过提取正交分量确保局部与全局特征的绝对物理隔离与纯净增量融合。
    [已注入梯度安全锁与恒等映射零初始化]
    """

    def __init__(self, in_features, hidden_features, dim1=32, dim2=128, drop=0.1):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

        # Block 1: 维度对齐层
        self.align_local = nn.Linear(dim1, dim2)

        # Block 4: 零冗余漏斗融合层
        self.fc_reduce = nn.Linear(dim2 * 2, hidden_features)
        self.fc_expand = nn.Linear(hidden_features, in_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

        # ==========================================
        # 绝对防御：零初始化输出层
        # 确保 Epoch 1 为完美恒等映射，避免残差干涉
        # ==========================================
        nn.init.zeros_(self.fc_expand.weight)
        nn.init.zeros_(self.fc_expand.bias)

    def forward(self, x):
        # [B, J, 160] -> 分解为局部 [B, J, 32] 和全局 [B, J, 128]
        x_local, x_global = torch.split(x, [self.dim1, self.dim2], dim=-1)

        # ==========================================
        # Block 1: 维度流形对齐
        # ==========================================
        aligned_local = self.align_local(x_local)  # [B, J, 128]

        # ==========================================
        # Block 2: 单位方向向量归一化 (注入梯度安全锁)
        # ==========================================
        # 使用安全的平方和根号计算，避免 torch.norm 导数为 NaN
        squared_sum = torch.sum(x_global ** 2, dim=-1, keepdim=True)
        safe_norm = torch.sqrt(squared_sum + 1e-6)
        unit_global = x_global / safe_norm  # [B, J, 128]

        # ==========================================
        # Block 3: 正交分量提纯 (剥离平行冗余)
        # ==========================================
        dot_product = torch.sum(aligned_local * unit_global, dim=-1, keepdim=True)  # [B, J, 1]
        parallel_local = dot_product * unit_global  # [B, J, 128]
        orthogonal_local = aligned_local - parallel_local  # [B, J, 128]

        # ==========================================
        # Block 4: 零冗余漏斗融合
        # ==========================================
        x_mod = torch.cat([orthogonal_local, x_global], dim=-1)  # [B, J, 256]

        out = self.act(self.fc_reduce(x_mod))
        out = self.drop(out)
        out = self.fc_expand(out)
        out = self.drop(out)

        return out

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., length=27):
        super().__init__()

        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads)
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, torch.div(C, self.num_heads, rounding_mode='floor')).permute(
            2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



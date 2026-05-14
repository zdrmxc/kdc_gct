import torch
import torch.nn as nn
import math


# ==============================================================================
# 纯净同构版 Gcn_block (完美等效原作者的 ST-GCN，采用 1D 张量流优化)
# 功能：保留 [3, 17, 17] 的空间划分（自身、向心、离心），数学上 100% 对齐原版
# ==============================================================================
class Gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.linear = nn.Linear(in_channels, out_channels * kernel_size, bias=bias)

    def forward(self, x, A):
        B, J, C_in = x.shape
        out_c = self.linear.out_features // self.kernel_size

        x_proj = self.linear(x)  # -> [B, J, 3 * C_out]
        x_proj = x_proj.view(B, J, self.kernel_size, out_c)  # -> [B, J, 3, C_out]
        x_perm = x_proj.permute(0, 2, 3, 1)  # -> [B, 3, C_out, J]

        out = torch.einsum('nkcv,kvw->ncw', x_perm, A)  # -> [B, C_out, J]
        out = out.permute(0, 2, 1).contiguous()  # -> [B, J, C_out]
        return out, A


class Gcn_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.05, residual=True):
        super().__init__()
        self.inplace = True
        self.momentum = 0.1

        self.gcn = Gcn(in_channels, out_channels, kernel_size)

        self.tcn = nn.Sequential(
            nn.BatchNorm1d(out_channels, momentum=self.momentum),
            nn.ReLU(inplace=self.inplace),
            nn.Dropout(0.05),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm1d(out_channels, momentum=self.momentum),
            nn.Dropout(dropout, inplace=self.inplace),
        )

        self.use_res = residual
        if self.use_res:
            if (in_channels == out_channels) and (stride == 1):
                self.residual = nn.Identity()
            else:
                self.residual = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm1d(out_channels, momentum=self.momentum),
                )

        self.gelu = nn.GELU()

    def forward(self, x, A):
        x_t = x.permute(0, 2, 1).contiguous()
        res = self.residual(x_t) if self.use_res else 0.0

        x_gcn, A = self.gcn(x, A)

        x_gcn_t = x_gcn.permute(0, 2, 1).contiguous()
        x_tcn = self.tcn(x_gcn_t)

        out = (x_tcn + res).permute(0, 2, 1).contiguous()
        return self.gelu(out), A
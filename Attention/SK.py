import torch
import torch.nn as nn
import torch.nn.functional as F

class SKAttention(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_sizes=(3, 5), reduction=16, group=1):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.n = len(kernel_sizes)
        self.d = max(in_channels // reduction, 32)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, k, padding=k//2, groups=group, bias=False)
            for k in kernel_sizes
        ])
        self.fc1 = nn.Conv2d(out_channels, self.d, 1, bias=False)
        self.fc2 = nn.Conv2d(self.d, out_channels * self.n, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        feats = torch.stack([conv(x) for conv in self.convs], dim=1)  # (B, n, C, H, W)
        U = torch.sum(feats, dim=1)  # (B, C, H, W)
        s = F.adaptive_avg_pool2d(U, 1)  # (B, C, 1, 1)
        z = self.fc1(s)  # (B, d, 1, 1)
        z = F.relu(z, inplace=True)
        z = self.fc2(z)  # (B, n*C, 1, 1)
        z = z.view(B, self.n, C, 1, 1)
        attention = self.softmax(z)  # (B, n, C, 1, 1)
        out = torch.sum(feats * attention, dim=1)  # (B, C, H, W)
        return out

# 测试调用
if __name__ == "__main__":
    sk_att = SKAttention(in_channels=64)
    x = torch.randn(2, 64, 32, 32)
    print(sk_att(x).shape)  # 输出：torch.Size([2, 64, 32, 32])
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        # 构建多层感知机（MLP）用于通道注意力
        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module(f'gate_c_fc_{i}', nn.Linear(gate_channels[i], gate_channels[i+1]))
            self.gate_c.add_module(f'gate_c_bn_{i+1}', nn.BatchNorm1d(gate_channels[i+1]))
            self.gate_c.add_module(f'gate_c_relu_{i+1}', nn.ReLU())
        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, in_tensor):
        # 修复：将 tensor 转为 int，避免 pool 层参数错误
        b, c, h, w = in_tensor.size()
        avg_pool = F.avg_pool2d(in_tensor, kernel_size=(h, w), stride=(h, w))  # 全局平均池化
        # 经过 MLP 得到通道注意力（形状：b, c → 扩展为 b, c, 1, 1 以匹配特征图维度）
        channel_att = self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)
        return channel_att

class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential(
            nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1),
            nn.BatchNorm2d(gate_channel // reduction_ratio),
            nn.ReLU(),
        )
        # 空洞卷积（扩张卷积）捕获多尺度空间信息
        for i in range(dilation_conv_num):
            self.gate_s.add_module(
                f'gate_s_conv_di_{i}',
                nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio,
                          kernel_size=3, padding=dilation_val, dilation=dilation_val)
            )
            self.gate_s.add_module(f'gate_s_bn_di_{i}', nn.BatchNorm2d(gate_channel//reduction_ratio))
            self.gate_s.add_module(f'gate_s_relu_di_{i}', nn.ReLU())
        # 1×1 卷积输出单通道空间注意力图
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1))

    def forward(self, in_tensor):
        # 空间注意力（形状：b, 1, h, w → 扩展为 b, c, h, w）
        spatial_att = self.gate_s(in_tensor).expand_as(in_tensor)
        return spatial_att

class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)  # 通道注意力分支
        self.spatial_att = SpatialGate(gate_channel)  # 空间注意力分支

    def forward(self, in_tensor):
        # 融合通道+空间注意力：1 + Sigmoid(通道注意力 × 空间注意力)
        att = 1 + F.sigmoid(self.channel_att(in_tensor) * self.spatial_att(in_tensor))
        # 注意力加权原始特征
        return att * in_tensor

# 测试代码（可运行）
if __name__ == "__main__":
    bam = BAM(gate_channel=64)  # 输入通道数64
    x = torch.randn(2, 64, 32, 32)  # 批量2，通道64，尺寸32×32
    out = bam(x)
    print(f"输入形状：{x.shape}")
    print(f"输出形状：{out.shape}")  # 输出形状与输入一致：torch.Size([2, 64, 32, 32])
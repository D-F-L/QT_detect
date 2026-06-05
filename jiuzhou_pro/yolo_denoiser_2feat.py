import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
# Load a model

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class UpsampleLayer(nn.Module):
    def __init__(self, in_planes, out_planse, stride=(1, 1), outpadding=(1,1)):
        # torch.set_default_dtype(torch.float64)
        super(UpsampleLayer, self).__init__()

        self.conv = nn.ConvTranspose2d(in_planes, out_planse, kernel_size=(3, 3), stride=stride, bias=False, padding=1, output_padding=outpadding)
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_planse)

    def forward(self, data):

        return self.act(self.norm(self.conv(data)))

class ECAAttention(nn.Module):
    """
    轻量通道注意力
    """
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv1d(y)                   # [B, 1, C]
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y.expand_as(x)

class FeatureFusionBlock(nn.Module):
    """
    P2 + upsample(P3) 融合
    """
    def __init__(self, p2_channels, p3_channels, out_channels):
        super().__init__()
        self.p3_reduce = ConvBNAct(p3_channels, out_channels, k=1, s=1, p=0)
        self.p2_reduce = ConvBNAct(p2_channels, out_channels, k=1, s=1, p=0)
        self.fuse = ConvBNAct(out_channels * 2, out_channels, k=3, s=1, p=1)
        self.attn = ECAAttention(out_channels)
    def forward(self, p2, p3):
        # p3 上采样到 p2 尺度
        p3 = self.p3_reduce(p3)
        p3 = F.interpolate(p3, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.p2_reduce(p2)
        x = torch.cat([p2, p3], dim=1)
        x = self.fuse(x)
        x = self.attn(x)
        return x

class YoloDenoiser(nn.Module):
    def __init__(self, out_channel=1, p2_channels=64, p3_channels=128, fuse_channels=64):
        # torch.set_default_dtype(torch.float64)
        super(YoloDenoiser, self).__init__()

        self.out_channel = out_channel
        yolomodel = YOLO(model="yolo26s-denoise-2feat.yaml")
        yolomodel.load("yolo26_weights/yolo26s.pt")
        self.yolomodel = yolomodel.model

        self.fusion = FeatureFusionBlock(
            p2_channels=p2_channels,
            p3_channels=p3_channels,
            out_channels=fuse_channels
        )
        # 两次上采样恢复原图尺寸
        self.upSampleLayer1 = UpsampleLayer(fuse_channels, 32, stride=(2, 2))
        self.upSampleLayer2 = UpsampleLayer(32, 8, stride=(2, 2))
        self.upSampleLayer3 = nn.Conv2d(8, out_channel, kernel_size=(1, 1), bias=True)
        # 输出前细化
        self.refine = nn.Sequential(
            ConvBNAct(16, 16, k=3, s=1, p=1),
            ConvBNAct(16, 8, k=3, s=1, p=1)
        )
        self.pred = nn.Conv2d(8, out_channel, kernel_size=1, bias=True)

    def forward(self, data, flag='train'):
        """input size: data-[batch, 2, frequency bins, time points]"""
        # 输入为 STFT 后的时频谱
        # yolomodel
        if flag == 'train':
            p2, p3 = self.yolomodel(data)['feats']
            self.fusion.train()
            self.upSampleLayer1.train()
            self.upSampleLayer2.train()
            self.upSampleLayer3.train()
        else:
            p2, p3 = self.yolomodel.predict(data)['feats']
            self.fusion.eval()
            self.upSampleLayer1.eval()
            self.upSampleLayer2.eval()
            self.upSampleLayer3.eval()
        x = self.fusion(p2, p3)
        print(x.shape)
        # 上采样恢复
        Out2 = self.upSampleLayer1(x)      
        # print("downOut2:", Out2.shape)
        Out3 = self.upSampleLayer2(Out2)
        # print("downOut3:", Out3.shape)
        Out4 = self.upSampleLayer3(Out3)
        if self.out_channel == 1:
            Out4 = Out4.squeeze(1)
        return Out4

# if __name__=='__main__':
#     # inputs = torch.randn([2, 3, 41, 10001])
#     inputs = torch.randn([2, 3, 256, 512]).cuda()
#     # inputs = inputs.to(dtype=torch.float64)
#     model = YoloDenoiser(out_channel=2).cuda()
#     for e, j in enumerate(model.named_parameters()):
#         if e <= 5:
#             print(j[0], 'mean: ', str(j[1].mean()))
#     num_param = sum([param.numel() for param in model.parameters()])
#     print('total params: ', round(num_param / 1e6, 6), 'M')
#     out = model(inputs)
#     print("shape", out.shape)  



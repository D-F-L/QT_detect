import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSampleLayer(nn.Module):
    def __init__(self, in_planes, out_planse, stride=(1, 1)):
        torch.set_default_dtype(torch.float64)
        super(DownSampleLayer, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planse, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_planse)

    def forward(self, data):

        return self.act(self.norm(self.conv(data)))

class UpsampleLayer(nn.Module):
    def __init__(self, in_planes, out_planse, stride=(1, 1), outpadding=(0,0)):
        torch.set_default_dtype(torch.float64)
        super(UpsampleLayer, self).__init__()

        self.conv = nn.ConvTranspose2d(in_planes, out_planse, kernel_size=(3, 3), stride=stride, bias=False, padding=1, output_padding=outpadding)
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_planse)

    def forward(self, data):

        return self.act(self.norm(self.conv(data)))


class UpsampleLayerSC(nn.Module):
    def __init__(self, in_planes, fm_planes, out_planse, stride=(1, 1), outpadding=(0,0)):
        torch.set_default_dtype(torch.float64)
        super(UpsampleLayerSC, self).__init__()

        self.conv = nn.ConvTranspose2d(in_planes, out_planse-fm_planes, kernel_size=(3, 3), stride=stride, bias=False, padding=1, output_padding=outpadding)
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_planse-fm_planes)

    def forward(self, data, fm):
        out = self.conv(data)
        # compute shape
        _,_, out_h, out_w = out.size()
        _,_,fm_h, fm_w = fm.size()
        out = F.pad(out, [0,fm_w-out_w,0,fm_h-out_h])
        out = self.act(self.norm(out))
    
        # print(out.shape, fm.shape)
        return torch.cat((out, fm), dim=1)



class DeepDenoiser(nn.Module):
    def __init__(self, inplaces=1):
        torch.set_default_dtype(torch.float64)
        super(DeepDenoiser, self).__init__()


        # down
        self.downSampleLayer1 = DownSampleLayer(in_planes=inplaces,  out_planse=8)
        self.downSampleLayer2 = DownSampleLayer(in_planes=8,  out_planse=8)     # skip connect
        self.downSampleLayer3 = DownSampleLayer(in_planes=8,  out_planse=8, stride=(2,2))
        self.downSampleLayer4 = DownSampleLayer(in_planes=8,  out_planse=16)    # skip connect
        self.downSampleLayer5 = DownSampleLayer(in_planes=16, out_planse=16,stride=(2,2))
        self.downSampleLayer6 = DownSampleLayer(in_planes=16, out_planse=32)    # skip connect
        self.downSampleLayer7 = DownSampleLayer(in_planes=32, out_planse=32, stride=(2, 2))
        self.downSampleLayer8 = DownSampleLayer(in_planes=32, out_planse=64)    # skip connect
        self.downSampleLayer9 = DownSampleLayer(in_planes=64, out_planse=64, stride=(2, 2))
        self.downSampleLayer10 = DownSampleLayer(in_planes=64, out_planse=128)  # skip connect
        self.downSampleLayer11 = DownSampleLayer(in_planes=128, out_planse=128, stride=(2, 2))
        self.downSampleLayer12 = DownSampleLayer(in_planes=128, out_planse=256)

        # up
        self.upSampleLayer1 = UpsampleLayerSC(in_planes=256, fm_planes=128, out_planse=512, stride=(2, 2))
        self.upSampleLayer2 = DownSampleLayer(in_planes=512, out_planse=128)
        self.upSampleLayer3 = UpsampleLayerSC(in_planes=128, fm_planes=64, out_planse=128, stride=(2, 2))
        self.upSampleLayer4 = DownSampleLayer(in_planes=128, out_planse=64)
        self.upSampleLayer5 = UpsampleLayerSC(in_planes=64, fm_planes=32, out_planse=64, stride=(2, 2))
        self.upSampleLayer6 = DownSampleLayer(in_planes=64, out_planse=32)
        self.upSampleLayer7 = UpsampleLayerSC(in_planes=32, fm_planes=16, out_planse=32, stride=(2, 2))
        self.upSampleLayer8 = DownSampleLayer(in_planes=32, out_planse=16)
        self.upSampleLayer9 = UpsampleLayerSC(in_planes=16, fm_planes=8, out_planse=16, stride=(2, 2))
        self.upSampleLayer10 = DownSampleLayer(in_planes=16, out_planse=8)
        self.upSampleLayer11 = nn.Conv2d(8, 2, kernel_size=(1, 1), bias=True)

    def forward(self, data):
        """input size: data-[batch, 2, frequency bins, time points]"""
        # 输入为 STFT 后的时频谱（虚实部）
        # 下采样层
        downOut1 = self.downSampleLayer1(data)
        # print("downOut1:", downOut1.shape)
        downOut2 = self.downSampleLayer2(downOut1)      # skip connect
        # print("downOut2:", downOut2.shape)
        downOut3 = self.downSampleLayer3(downOut2)
        # print("downOut3:", downOut3.shape)
        downOut4 = self.downSampleLayer4(downOut3)      # skip connect
        # print("downOut4:", downOut4.shape)
        downOut5 = self.downSampleLayer5(downOut4)
        # print("downOut5:", downOut5.shape)
        downOut6 = self.downSampleLayer6(downOut5)      # skip connect
        # print("downOut6:", downOut6.shape)
        downOut7 = self.downSampleLayer7(downOut6)
        # print("downOut7:", downOut7.shape)
        downOut8 = self.downSampleLayer8(downOut7)      # skip connect
        # print("downOut8:", downOut8.shape)
        downOut9 = self.downSampleLayer9(downOut8)
        # print("downOut9:", downOut9.shape)
        downOut10 = self.downSampleLayer10(downOut9)    # skip connect
        # print("downOut10:", downOut10.shape)
        downOut11 = self.downSampleLayer11(downOut10)
        # print("downOut11:", downOut11.shape)
        downOut12 = self.downSampleLayer12(downOut11)
        # print("downOut12:", downOut12.shape)


        # 上采样层
        upOut1 = self.upSampleLayer1(downOut12, downOut10)
        # print("upOut1:", upOut1.shape)
        upOut2 = self.upSampleLayer2(upOut1)
        # print("upOut2:", upOut2.shape)
        upOut3 = self.upSampleLayer3(upOut2, downOut8)
        # print("upOut3:", upOut3.shape)
        upOut4 = self.upSampleLayer4(upOut3)
        # print("upOut4:", upOut4.shape)
        upOut5 = self.upSampleLayer5(upOut4, downOut6)
        # print("upOut5:", upOut5.shape)
        upOut6 = self.upSampleLayer6(upOut5)
        # print("upOut6:", upOut6.shape)
        upOut7 = self.upSampleLayer7(upOut6, downOut4)
        # print("upOut7:", upOut7.shape)
        upOut8 = self.upSampleLayer8(upOut7)
        # print("upOut8:", upOut8.shape)
        upOut9 = self.upSampleLayer9(upOut8, downOut2)
        # print("upOut9:", upOut9.shape)
        upOut10 = self.upSampleLayer10(upOut9)
        # print("upOut10:", upOut10.shape)
        upOut11 = self.upSampleLayer11(upOut10)
        # print("upOut11:", upOut11.shape)

        return upOut11

if __name__=='__main__':
    inputs = torch.randn([2, 1, 41, 10001])
    inputs = inputs.to(dtype=torch.float64)
    model = DeepDenoiser()
    out = model(inputs)
    print("shape", out.shape)  
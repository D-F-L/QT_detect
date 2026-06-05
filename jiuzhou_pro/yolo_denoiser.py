import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
# Load a model

class UpsampleLayer(nn.Module):
    def __init__(self, in_planes, out_planse, stride=(1, 1), outpadding=(1,1)):
        # torch.set_default_dtype(torch.float64)
        super(UpsampleLayer, self).__init__()

        self.conv = nn.ConvTranspose2d(in_planes, out_planse, kernel_size=(3, 3), stride=stride, bias=False, padding=1, output_padding=outpadding)
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_planse)

    def forward(self, data):

        return self.act(self.norm(self.conv(data)))

class YoloDenoiser(nn.Module):
    def __init__(self, out_channel=1):
        # torch.set_default_dtype(torch.float64)
        super(YoloDenoiser, self).__init__()

        self.out_channel = out_channel
        yolomodel = YOLO(model="yolo26s-denoise.yaml")
        yolomodel.load("yolo26_weights/yolo26s.pt")
        self.yolomodel = yolomodel.model

        self.upSampleLayer1 = UpsampleLayer(in_planes=64, out_planse=32, stride=(2, 2))
        self.upSampleLayer2 = UpsampleLayer(in_planes=32, out_planse=8, stride=(2, 2))
        self.upSampleLayer3 = nn.Conv2d(8, out_channel, kernel_size=(1, 1), bias=True)

    def forward(self, data, flag='train'):
        """input size: data-[batch, 2, frequency bins, time points]"""
        if flag == 'train':
            result = self.yolomodel(data)
            if isinstance(result, dict):
                Out1 = result['feats'][0]
            else:
                Out1 = result[0]
            self.upSampleLayer1.train()
            self.upSampleLayer2.train()
            self.upSampleLayer3.train()
        else:
            result = self.yolomodel.predict(data)
            if isinstance(result, dict):
                Out1 = result['feats'][0]
            else:
                Out1 = result[0]
            self.upSampleLayer1.eval()
            self.upSampleLayer2.eval()
            self.upSampleLayer3.eval()
        Out2 = self.upSampleLayer1(Out1)
        Out3 = self.upSampleLayer2(Out2)
        Out4 = self.upSampleLayer3(Out3)
        if self.out_channel == 1:
            Out4 = Out4.squeeze(1)
        return Out4

# if __name__=='__main__':
#     # inputs = torch.randn([2, 3, 41, 10001])
#     inputs = torch.randn([2, 3, 288, 10016]).cuda()
#     # inputs = inputs.to(dtype=torch.float64)
#     model = YoloDenoiser(out_channel=1).cuda()
    
#     out = model(inputs)
#     print("shape", out.shape)  



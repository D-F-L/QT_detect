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
        yolomodel = YOLO(model="yolo26n-denoise.yaml")
        yolomodel.load("yolo26_weights/yolo26n.pt")
        self.yolomodel = yolomodel.model

        self.upSampleLayer1 = UpsampleLayer(in_planes=32, out_planse=16, stride=(2, 2))
        self.upSampleLayer2 = UpsampleLayer(in_planes=16, out_planse=8, stride=(2, 2))
        self.upSampleLayer3 = nn.Conv2d(8, out_channel, kernel_size=(1, 1), bias=True)

    def forward(self, data, flag='train'):
        """input size: data-[batch, 2, frequency bins, time points]"""
        # 输入为 STFT 后的时频谱
        # yolomodel
        Out1 = self.yolomodel(data)
        # print('Out1', Out1.keys())
        # boxes = Out1['boxes']
        # scores = Out1['scores']
        # feats = Out1['feats']
        # print('boxes', boxes.shape)
        # print('scores', scores.shape)
        # print('feats', feats[0].shape)
        if flag == 'train':
            Out1 = self.yolomodel(data)['feats'][0]
            self.upSampleLayer1.train()
            self.upSampleLayer2.train()
            self.upSampleLayer3.train()
        else:
            Out1 = self.yolomodel.predict(data)['feats'][0]
            self.upSampleLayer1.eval()
            self.upSampleLayer2.eval()
            self.upSampleLayer3.eval()
        # print("Out1:", Out1.shape)
        Out2 = self.upSampleLayer1(Out1)      
        # print("downOut2:", Out2.shape)
        Out3 = self.upSampleLayer2(Out2)
        # print("downOut3:", Out3.shape)
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



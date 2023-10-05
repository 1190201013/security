import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import data_comparison2
from torchsummary import summary
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.utils.data import DataLoader
from utils.FID import fid
from utils.loss import PerceptualLoss

gpu_id = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
device = 'cuda:'+gpu_id if torch.cuda.is_available() else 'cpu'

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='xavier', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)
        self.apply(init_func)
'''
class visibnet(BaseNetwork):
    def __init__(self,in_channels):
        super(visibnet, self).__init__()
        #encoder
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        #decoder
        self.layer7=nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer9 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer10 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer11 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer12 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer13 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer14 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer15 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer16 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )
        self.init_weights()
    def forward(self,data):
        layer1 = self.layer1(data)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)
        layer7 = self.layer7(layer6)
        layer8 = self.layer8(torch.cat([layer7, layer5], dim=1))
        layer9 = self.layer9(torch.cat([layer8, layer4], dim=1))
        layer10 = self.layer10(torch.cat([layer9, layer3], dim=1))
        layer11 = self.layer11(torch.cat([layer10, layer2], dim=1))
        layer12 = self.layer12(torch.cat([layer11, layer1], dim=1))
        layer13 = self.layer13(layer12)
        layer14 = self.layer14(layer13)
        layer15=self.layer15(layer14)
        layer16=self.layer16(layer15)
        return layer16
#net=visibnet(in_channels=1)
#summary(net,input_data=torch.rand(1,1,256,256))
'''
class coarsenet(BaseNetwork):
    def __init__(self, in_channels):
        super(coarsenet, self).__init__()
        # encoder
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # decoder
        self.layer7 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer9 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer10 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer11 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer12 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer13 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer14 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer15 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer16 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )
        self.init_weights()

    def forward(self, data):
        layer1 = self.layer1(data)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)
        layer7 = self.layer7(layer6)
        layer8 = self.layer8(torch.cat([layer7, layer5], dim=1))
        layer9 = self.layer9(torch.cat([layer8, layer4], dim=1))
        layer10 = self.layer10(torch.cat([layer9, layer3], dim=1))
        layer11 = self.layer11(torch.cat([layer10, layer2], dim=1))
        layer12 = self.layer12(torch.cat([layer11, layer1], dim=1))
        layer13 = self.layer13(layer12)
        layer14 = self.layer14(layer13)
        layer15 = self.layer15(layer14)
        layer16 = self.layer16(layer15)
        return layer16

class refinenet(BaseNetwork):
    def __init__(self, in_channels):
        super(refinenet, self).__init__()
        # encoder
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # decoder
        self.layer7 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.layer8 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.layer9 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.layer10 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.layer11 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.layer12 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.layer13 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.layer14 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.layer15 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.layer16 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )
        self.init_weights()

    def forward(self, data):
        layer1 = self.layer1(data)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)
        layer7 = self.layer7(layer6)
        layer8 = self.layer8(torch.cat([layer7, layer5], dim=1))
        layer9 = self.layer9(torch.cat([layer8, layer4], dim=1))
        layer10 = self.layer10(torch.cat([layer9, layer3], dim=1))
        layer11 = self.layer11(torch.cat([layer10, layer2], dim=1))
        layer12 = self.layer12(torch.cat([layer11, layer1], dim=1))
        layer13 = self.layer13(layer12)
        layer14 = self.layer14(layer13)
        layer15 = self.layer15(layer14)
        layer16 = self.layer16(layer15)
        return layer16

class discriminator(BaseNetwork):
    def __init__(self,in_channels):
        super(discriminator, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1,padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        self.layer7=nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192,1024),
            nn.LeakyReLU(),
        )
        self.layer8=nn.Sequential(
            nn.Linear(1024,1024),
            nn.LeakyReLU(),
        )
        self.layer9=nn.Sequential(
            nn.Linear(1024,1024),
            nn.LeakyReLU(),
        )
        self.layer10=nn.Sequential(
            nn.Linear(1024,2),
            nn.Softmax()
        )
        self.init_weights()
    def forward(self,data):
        layer1=self.layer1(data)
        layer2=self.layer2(layer1)
        layer3=self.layer3(layer2)
        layer4=self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)
        layer7 = self.layer7(layer6)
        layer8 = self.layer8(layer7)
        layer9 = self.layer9(layer8)
        layer10 = self.layer10(layer9)
        return layer10

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.save_dir = 'compare_weights/'
        
class ImgModel(BaseModel):
    def __init__(self):
        super(ImgModel, self).__init__()
        self.lr = 1e-4
        self.gan_type = 're_avg_gan'
        self.in_channels = 128
        self.coarse= coarsenet(in_channels=self.in_channels).cuda()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.coarse_optimizer=optim.Adam(self.coarse.parameters(), lr=self.lr, betas=(0.9, 0.999),eps=1e-8)
        self.L1_LOSS_WEIGHT = 100
        self.PERC_LOSS_WEIGHT = 1
    def process(self, Si, Ig):
        self.coarse_optimizer.zero_grad()
        Io = self(Si)
        perceptual_loss = self.perceptual_loss(Io, Ig) * self.PERC_LOSS_WEIGHT
        l1_loss = self.l1_loss(Io, Ig) * self.L1_LOSS_WEIGHT
        loss =  l1_loss + perceptual_loss
        return Io, loss

    def forward(self, Si):
        return self.coarse(Si)

    def backward(self, loss=None, retain_graph=True):
        loss.backward(retain_graph=retain_graph)
        self.coarse_optimizer.step()

    def save(self, path):
        if not os.path.exists(self.save_dir + path):
            os.makedirs(self.save_dir + path)
        torch.save(self.coarse.state_dict(), self.save_dir + path + 'img_gen.pth')

    def load(self, path):
        self.coarse.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load(self.save_dir
                                                                + path + 'img_gen.pth').items()})





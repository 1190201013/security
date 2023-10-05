import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import data_comparison
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.utils.data import DataLoader
from utils.FID import fid
from torchsummary import summary
gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
device = 'cuda:' + gpu_id if torch.cuda.is_available() else 'cpu'

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

class ImgGenerator(BaseNetwork):
    def __init__(self, in_channels=133):
        super(ImgGenerator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=5, stride=2,padding=2)
        )
        self.layer2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2,padding=1),
        )

        self.layer3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2,padding=1),
        )
        self.layer4 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=2,padding=1),
        )
        self.layer5 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1,padding=1),
        )
        self.layer6 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1,padding=1),
        )
        self.layer7 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2,padding=1),
        )
        self.layer8 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2,padding=1),
        )

        self.layer9 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2,padding=1),
        )
        self.layer10 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2,padding=1),
        )
        self.layer11 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2,padding=1),
        )
        self.layer12 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2,padding=1),
        )
        self.init_weights()

    def forward(self, feature):
        layer1 = self.layer1(feature)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)
        layer7 = self.layer7(layer6)
        layer8 = self.layer8(layer7)
        layer9 = self.layer9(layer8)
        layer10 = self.layer10(layer9)
        layer11 = self.layer11(layer10)
        layer12 = self.layer12(layer11)
        return layer12

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.save_dir = 'compare_weights/'


class ImgModel(BaseModel):
    def __init__(self):
        super(ImgModel, self).__init__()
        self.lr = 1e-4
        self.in_channels = 133
        self.gen = ImgGenerator(in_channels=self.in_channels).cuda()
        self.loss = 0
        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.weight=0.001
    def Loss(self,fake,real):
        return torch.sum(torch.square(fake-real))/(256*256)
    def process(self, feature, Ig):
        self.gen_optimizer.zero_grad()
        Io = self(feature)
        loss=self.Loss(Io,Ig)*self.weight
        return Io, loss
    def forward(self, feature):
        return self.gen(feature)

    def backward(self, loss=None, retain_graph=True):
        loss.backward(retain_graph=retain_graph)
        self.gen_optimizer.step()

    def save(self, path):
        if not os.path.exists(self.save_dir + path):
            os.makedirs(self.save_dir + path)
        torch.save(self.gen.state_dict(), self.save_dir + path + 'img_gen.pth')

    def load(self, path):
        self.gen.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load(self.save_dir
                                                                + path + 'img_gen.pth').items()})


class SIFTReconstruction():
    def __init__(self):
        self.train_num = 40000
        self.test_num = 5000
        self.batch_size = 32
        self.n_epochs = 35
        train_file = '/home/lixin/data/bird/train/'
        test_file = '/home/lixin/data/bird/test/'
        train_dataset = data_comparison.SIFTDataset(self.train_num, train_file)
        test_dataset = data_comparison.SIFTDataset(self.test_num, test_file)
        self.img_model = ImgModel().cuda()
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)

    def train(self):
        print('\nTrain/Val ' + ' model:')
        for epoch in range(self.n_epochs):
            gen_losses, ssim, psnr = [], [], []
            for cnt, items in enumerate(self.train_loader):
                self.img_model.train()
                Ig, Si, Lg = (item.cuda() for item in items[:-1])
                Io, loss = self.img_model.process(Si, Ig)
                self.img_model.backward(loss)
                s, p = self.metrics(Ig, Io)
                ssim.append(s)
                psnr.append(p)
                gen_losses.append(loss.item())
                if cnt % 20 == 0:
                    print('Tra (%d/%d) Loss:%5.4f, SSIM:%4.4f, PSNR:%4.2f' %
                          (cnt, self.train_num, np.mean(gen_losses), np.mean(ssim), np.mean(psnr)))
        self.img_model.save('image_comparison_bird' + '/')

    def test(self, pretrained=True):
        if pretrained:
            print('\nTest ' + ' model:')
            self.img_model.load('image_comparison_bird' + '/')
        self.img_model.eval()
        ssim, psnr = [], []
        if not os.path.exists('res/image_comparison_bird'):
            os.makedirs('res/image_comparison_bird')
        for cnt, items in enumerate(self.test_loader):
            Ig, Si, Lg = (item.cuda() for item in items[:-1])
            Io = self.img_model(Si)
            s, p = self.metrics(Ig, Io)
            ssim.append(s)
            psnr.append(p)
            if cnt < 500:
                Io = self.postprocess(Io)
                path = 'res/image_comparison_bird/' + 'Io_%06d.jpg' % (cnt + 1)
                cv2.imwrite(path, Io[0])
        path1 = '/home/lixin/data/bird/test/'
        path2 = "res/image_comparison_bird/"
        path = [path1, path2]
        fid_result = fid(path)
        if pretrained:
            print(self.choice + ' Evaluation: SSIM:%4.4f, PSNR:%4.2f, FID:%4.2f' % (
            np.mean(ssim), np.mean(psnr), fid_result))
        return np.mean(ssim), np.mean(psnr), fid_result

    def postprocess(self, img):
        img = img * 127.5 + 127.5
        img = img.permute(0, 2, 3, 1)
        return img.int().cpu().detach().numpy()

    def metrics(self, Ig, Io):
        a = self.postprocess(Ig)
        b = self.postprocess(Io)
        ssim, psnr = [], []
        for i in range(len(a)):
            ssim.append(compare_ssim(a[i], b[i], win_size=11, data_range=255.0, channel_axis=2))
            psnr.append(compare_psnr(a[i], b[i], data_range=255))
        return np.mean(ssim), np.mean(psnr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, help='train or test the model', choices=['train', 'test'])
    args = parser.parse_args()
    model = SIFTReconstruction()
    if args.type == 'train':
        model.train()
    elif args.type == 'test':
        model.test(True)
    print('End.')


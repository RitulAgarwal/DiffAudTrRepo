import torch.nn as nn 

class UNet(nn.Module):
    def __init__(self):
        super(UNet ,self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=8, out_channels= 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128,3)
        self.conv3 = nn.Conv2d(128,256,3)
        self.conv4 = nn.Conv2d(256,512,3)
        self.latent_rep = nn.Conv2d(512,1,1)
        
        self.df = nn.ConvTranspose2d(1,512,1)
        self.deconv1 = nn.ConvTranspose2d(512, 256,3)
        self.deconv2 = nn.ConvTranspose2d(256, 128,3)
        self.deconv3 = nn.ConvTranspose2d(128,64,3)
        self.deconv4 = nn.ConvTranspose2d(64,8,3)
        
    def forward(self, x):
        print('INSIDE UNET')
        print(x.shape)
        e1 = self.relu(self.conv1(x))
        print(e1.shape)
        
        e2 = self.relu(self.conv2(e1))
        print(e2.shape)
        e3 = self.relu(self.conv3(e2))
        print(e3.shape)
        e4 = self.relu(self.conv4(e3))
        print(e4.shape)
        e5 = self.latent_rep(e4)##laent rep as only 1 channel
        print(e5.shape)
        # return e5.squeeze(1).flatten()
        ##if u want to have the latent represenstation accessed then do this 
        # and do this with statspool or maxpool as ye wali lr toh 61504 ki hai isko 1024 banna hoga 
        d0 = self.df(e5)
        print(d0.shape)
        
        d1 = self.relu(self.deconv1(d0+e4))
        print(d1.shape)
        
        d2 = self.relu(self.deconv2(d1+e3))
        print(d2.shape)
        
        d3 = self.relu(self.deconv3(d2+e2))
        print(d3.shape)
        
        d4 = self.relu(self.deconv4(d3+e1))
        print(d4.shape)

        return d4
    
import numpy as np
import torch
import torch.utils.data as torch_data
from torchvision.utils import save_image
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import random


'''
Generator and discriminator models
'''

class Reshape(nn.Module):
    def __init__(self, channels, shape):
        super(Reshape, self).__init__()
        self.channels = channels
        self.shape = shape

    def forward(self, x):
        return x.reshape(-1, self.channels, self.shape, self.shape, self.shape)

class Flatten(torch.nn.Module):

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class Generator3D_Adaptive(nn.Module):
    def __init__(self, latent_size=100, n_features=32, output_shape=(64, 64, 64)):
        super(Generator3D_Adaptive, self).__init__()
        self.latent_size = latent_size
        self.output_shape = output_shape
        self.init_shape = tuple(np.array(output_shape) // 16)
        self.main = nn.ModuleList([
            # input is Z, going into a convolution
            nn.Linear(self.latent_size, n_features * 8 * np.prod(self.init_shape)),

            Reshape(n_features * 8, self.init_shape[0]),

            nn.BatchNorm3d(n_features * 8),
            nn.ReLU(True),

            nn.ConvTranspose3d(n_features * 8, n_features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(n_features * 4),
            nn.ReLU(True),

            nn.ConvTranspose3d(n_features * 4, n_features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(n_features * 2),
            nn.ReLU(True),

            nn.ConvTranspose3d(n_features * 2, n_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(n_features),
            nn.ReLU(True),

            nn.ConvTranspose3d(n_features, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid() # -> normalize images to 

        ])

    def forward(self, x):
        for i in range(len(self.main)):
            x = self.main[i](x)
        return x
    
    
class Discriminator3D_Adaptive(nn.Module):
    def __init__(self, n_features=32, n_outputs=1, input_shape=(64, 64, 64)):
        super(Discriminator3D_Adaptive, self).__init__()
        self.input_shape = input_shape
        self.final_shape = tuple(np.array(input_shape) // 16)
        self.main = nn.ModuleList([

            nn.Conv3d(1, n_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(n_features, n_features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(n_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(n_features * 2, n_features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(n_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(n_features * 4, n_features * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(n_features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            Flatten(),
            nn.Linear(n_features * 8 * np.prod(self.final_shape), n_outputs), # outputs logits

            nn.Sigmoid()
        ])

    def forward(self, x):
        for i in range(len(self.main)):
            x = self.main[i](x)
        return x
    
 
'''
Encoder model
'''

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_bn, padding=1):
        super(ResBlock, self).__init__()

        if use_bn:
            self.bn1 = nn.BatchNorm3d(in_channels)
            self.bn2 = nn.BatchNorm3d(in_channels // 2)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
        
        self.conv0 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels // 2, out_channels, kernel_size=3, padding=padding)


    def forward(self, x):
        identity = self.conv0(x)

        out = self.bn1(x)
        out = self.relu(x)
        out = self.conv1(x)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity

        return out
    
class Encoder(nn.Module):

    def __init__(self, in_channels=1, latent_size=100, use_bn=True):

        super(Encoder, self).__init__()

        n = 16
        self.conv1 = nn.Conv3d(in_channels, n, kernel_size=1)
        self.resblock1 = ResBlock(n, 2 * n, use_bn)
        self.resblock2 = ResBlock(2 * n, 4 * n, use_bn)
        self.resblock3 = ResBlock(4 * n, 8 * n, use_bn)

        self.pool = nn.MaxPool3d(kernel_size=2)
        self.act = nn.ReLU()
        self.flatten = Flatten()
        self.fc = nn.Linear((8 ** 3) * 8*n, latent_size)

    def forward(self, x):

        x = self.act(self.conv1(x))
        x = self.resblock1(x)
        x = self.pool(x)
        x = self.resblock2(x)
        x = self.pool(x)
        x = self.resblock3(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x
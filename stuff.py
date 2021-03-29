import pytorch_lightning as pl

import torch
import torchvision
from torchvision import datasets
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

magic = 267648

magical_features = 128

class AEDecoder(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.decoder_step_1 = nn.Linear(magical_features, magic)
        self.decoder_step_2 = nn.Sequential(
            nn.AdaptiveMaxPool2d((53, 43)),
            nn.ConvTranspose2d(128, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
            nn.AdaptiveMaxPool2d((218, 178)),
        )
    
    def forward(self, x):
        res = self.decoder_step_1(x)
        res = res.view(-1, 128, 51, 41)
        return self.decoder_step_2(res)
    
class Varience(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.mn = 0
        self.std = 0
        self.mn_layer = nn.Linear(magical_features, magical_features)
        self.std_layer = nn.Linear(magical_features, magical_features)
    
    def forward(self, x):
        self.mn = self.mn_layer(x) + 1
        self.std = self.std_layer(x) + 1
        point = torch.normal(self.mn, self.std)
        return (self.mn + torch.exp(self.std/2) * point)

class AE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.epoch = 0
        self.varience = Varience()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(magic, magical_features),
            self.varience,
        )
        self.decoder = AEDecoder()
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        #loss = F.mse_loss(decoded, x)
        loss = F.binary_cross_entropy(decoded, x)
        pred = torch.cat((self.varience.mn, self.varience.std), 1).cuda()
        loss += F.kl_div(pred, torch.zeros(pred.shape).cuda())
        self.log('loss', loss, on_epoch=True, logger=True)
        return loss
    
    def on_train_epoch_end(self, outputs):
        self.epoch +=1
        d = MNIST()
        d.setup(stage='')
        start = d.test_d[1][0].cuda()
        image = self.decoder(torch.normal(self.varience.mn, self.varience.std))
        self.logger.experiment.add_image('result bce: epoch %d' % self.epoch, image[0])

    
class MNIST(pl.LightningDataModule):
    def __init__(self):
        super().__init__(self)
    
    def setup(self, stage):
        self.train_d = torchvision.datasets.MNIST(root="G:\MLData", train=True, download=False, transform=torchvision.transforms.ToTensor())
        self.test_d = torchvision.datasets.MNIST(root="G:\MLData", train=False, download=False, transform=torchvision.transforms.ToTensor())
        
    def train_dataloader(self):
        return DataLoader(self.train_d, shuffle=True, batch_size=128)
    
    def test_dataloader(self):
        return DataLoader(self.test_d, shuffle=True, batch_size=128)
    
class Celeb(pl.LightningDataModule):
    def __init__(self):
        super().__init__(self)
        
    def setup(self, stage):
        dataset=datasets.ImageFolder('G:\\MLData\\Celeba', transform=torchvision.transforms.ToTensor())
        train, test = train_test_split(list(range(100)), test_size=0.30)
        self.train = Subset(dataset, train)
        self.test = Subset(dataset, test)
        
    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=10)
    
    def test_dataloader(self):
        return DataLoader(self.test, shuffle=True, batch_size=10)
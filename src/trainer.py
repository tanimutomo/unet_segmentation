import time
from datetime import timedelta
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from src.model import UNet
from src.utils import init_weights


class Trainer(object):

    def __init__(self, model, criterion, optim, train_loader, val_loader, device, conf):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optim = optim
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = conf['epochs']
        self.save_name = conf['save_name']

    def train(self):
        self.model.apply(init_weights)
        self.model.train()
        for epoch in range(self.epochs):
            train_start_time = time.time()
            for i, (input, target) in enumerate(self.train_loader):
                input = input.to(self.device)
                target = target.to(self.device, dtype=torch.float32)
                self.optim.zero_grad()
                output = self.model(input)
                target = F.upsample(torch.unsqueeze(target, 0), output.size()[2:], mode='nearest')
                target = torch.squeeze(target, 0).to(torch.int64)
                # print(output.shape, output.dtype)
                # print(target.shape, target.dtype)
                loss = self.criterion(output, target)
                loss.backward()
                self.optim.step()

                if i == 1:
                    print('training is starting')

            seconds = time.time() - train_start_time
            elapsed = str(timedelta(seconds=seconds))
            print('Epoch : {epoch}\t'
                    'Time : {time}\t'
                    'Loss : {loss:.4f}\t'.format(
                        epoch=epoch+1,
                        time=elapsed, loss=loss.item()))

        if os.path.exists('./model'):
            pass
        else:
            os.mkdir('./model')
        torch.save(self.model.state_dict(), './src/model/u_net_{}.pth'.format(self.save_name))



#!/Users/tanimu/.pyenv/shims/python
# encoding=utf8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
import torchvision.transforms.functional as F

from src.dutils import transforms as extended_transforms
from src.dutils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d
from src.dutils import transforms as extended_transforms

import numpy as np
from src.dataset import VOC
from src.loader import get_loader
from src.trainer import Trainer
from src.model import UNet
from src.utils import *

conf = {
        'epochs': 3,
        'bs': 4,
        'lr': 1e-3,
        'momentum': 0.9,
        'root': './data/VOC/',
        'save_name': 0,
        'init_size': (256, 256)
        }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet()
criterion = nn.CrossEntropyLoss()
optim = optim.SGD(model.parameters(), lr=conf['lr'], momentum=conf['momentum'])
train_loader, val_loader = get_loader(conf['root'], conf['bs'], conf['init_size'])

trainer = Trainer(model, criterion, optim, train_loader, val_loader, device, conf)
trainer.train()


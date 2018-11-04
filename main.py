#!/Users/tanimu/.pyenv/shims/python
# encoding=utf8

import os
import argparse
import numpy as np
from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
import torchvision.transforms.functional as F

from src.dataset import VOC
from src.loader import get_loader
from src.trainer import Trainer
from src.model import UNet
from src.utils import *


params = {
        'epochs': 1,
        'bs': 4,
        'lr': 1e-4,
        'momentum': 0.95,
        'init_size': (256, 256),
        'bn': True,
        'visualize': True,
        'save_name': 100,
        'num_classes': 21,
        'cml': False
        }

if params['cml']:
    experiment = Experiment(api_key="xK18bJy5xiPuPf9Dptr43ZuMk",
            project_name="U-Net_VOC", workspace="tanimutomo")
else:
    experiment = None

if params['cml']:
    experiment.log_multiple_params(params)

data_root = './data/VOC'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = UNet(params['bn'])
criterion = nn.CrossEntropyLoss(ignore_index=255)
optim = optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'])
train_loader, val_loader = get_loader(data_root, params['bs'], params['init_size'])

if os.path.exists('./model') == 0:
    os.mkdir('./model')

trainer = Trainer(device, model, criterion, optim, train_loader, val_loader, params, experiment)
trainer.iteration()


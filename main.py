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

import os
import argparse
import numpy as np
from comet_ml import Experiment

from src.dataset import VOC
from src.loader import get_loader
from src.trainer import Trainer
from src.model import UNet
from src.utils import *


parser = argparse.ArgumentParser(description='Params')
parser.add_argument('--cml', action='store_true')
args = parser.parse_args()

if args.cml:
    experiment = Experiment(api_key="xK18bJy5xiPuPf9Dptr43ZuMk",
            project_name="U-Net_VOC", workspace="tanimutomo")
else:
    experiment = None

hyper_params = {
        'epochs': 1000,
        'bs': 4,
        'lr': 1e-2,
        'momentum': 0.9,
        'init_size': (256, 256),
        'bn': True,
        'visualize': True,
        'save_name': 2,
        'num_classes': 21,
        }

if args.cml:
    experiment.log_multiple_params(hyper_params)

data_root = './data/VOC'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(hyper_params['bn'])
criterion = nn.CrossEntropyLoss(ignore_index=255)
optim = optim.SGD(model.parameters(), lr=hyper_params['lr'], momentum=hyper_params['momentum'])
train_loader, val_loader = get_loader(data_root, hyper_params['bs'], hyper_params['init_size'])

if os.path.exists('./model') == 0:
    os.mkdir('./model')
if os.path.exists('./image') == 0:
    os.mkdir('./image')

trainer = Trainer(device, model, criterion, optim, train_loader, val_loader, hyper_params, experiment)
trainer.iteration()


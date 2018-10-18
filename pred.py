#!/Users/tanimu/.pyenv/shims/python
# encoding=utf8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torchvision.transforms import functional as F

import os
import argparse
import numpy as np
from comet_ml import Experiment

from src.dataset import VOC
from src.loader import get_loader
from src.model import UNet
from src.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_root = './data/VOC'
to_save_dir = './image'
model_path = './model/u_net_6.pth'
model_num = '6'

if os.path.exists('./image') == 0:
    os.mkdir('./image')
if os.path.exists('./image/train') == 0:
    os.mkdir('./image/train')
if os.path.exists('./image/val') == 0:
    os.mkdir('./image/val')

model = UNet(bn=True)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

train_loader, val_loader = get_loader(data_root, 1, (256, 256))

rand = torch.randint(0, 100, (3,))

'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

restore_transform = transforms.Compose([
        DeNormalize(*mean_std),
        transforms.ToPILImage(),
    ])

visualize = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(400),
        transforms.ToTensor()
    ])

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = F.to_pil_image(mask.to(dtype=torch.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

for phase, loader in zip(['train', 'val'], [train_loader, val_loader]):
    for i, (input, gt) in enumerate(loader):
        if i in rand:
            val_visual = []
            input = input.to(device)
            gt = gt.to(device)
            pred = torch.argmax(model(input), dim=1)
            input = input.squeeze(0)
            
            input_pil = restore_transform(input.detach())
            gt_pil = colorize_mask(gt.detach())
            pred_pil = colorize_mask(pred.detach())

            # input_pil.save(os.path.join(to_save_dir, '{}_{}_input.png'.format(phase, i)))
            # pred_pil.save(os.path.join(to_save_dir, '{}_{}_pred.png'.format(phase, i)))
            # gt_pil.save(os.path.join(to_save_dir, '{}_{}_gt.png'.format(phase, i)))

            val_visual.extend([visualize(input_pil.convert('RGB')),
                visualize(pred_pil.convert('RGB')),
                visualize(gt_pil.convert('RGB'))])

            val_visual = torch.stack(val_visual, 0)
            val_visual = save_image(val_visual, 
                    filename=os.path.join(to_save_dir, phase, '{}_{}_stack.png'.format(model_num, i)),
                    nrow=3, padding=5)
        else:
            pass

        if i >= 100:
            break
            


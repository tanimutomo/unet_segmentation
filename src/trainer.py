import os
import time
import random
from datetime import timedelta
from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from src.model import UNet
from src.utils import init_weights, evaluate, AverageMeter, colorize_mask, DeNormalize
from src.loader import visualize, restore_transform



class Trainer(object):

    def __init__(self, device, model, criterion, optim, train_loader, val_loader, params):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optim = optim
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = params['epochs']
        self.save_name = params['save_name']
        self.num_classes = params['num_classes']
        self.visualize = params['visualize']


    def iteration(self):
        self.model.apply(init_weights)
        for epoch in range(self.epochs):
            train_loss = self.train(epoch)
            val_loss, acc, acc_cls, mean_iu, fwavacc = self.validate(epoch)

            metrics = {
                    'train_loss': train_loss.avg,
                    'val_loss': val_loss.avg,
                    'accuracy': acc,
                    'accuracy_class': acc_cls,
                    'mean_iu': mean_iu,
                    'fwavacc': fwavacc
                    }
            experiment.log_multiple_metrics(metrics, step=epoch)

            self.report(metrics, epoch)

    def train(self, epoch):
        self.model.train()

        train_loss = AverageMeter()

        for i, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device, dtype=torch.float32)
            self.optim.zero_grad()
            output = self.model(inputs)
            targets = F.upsample(torch.unsqueeze(targets, 0), output.size()[2:], mode='nearest')
            targets = torch.squeeze(targets, 0).to(torch.int64)
            # print(output.shape, output.dtype)
            # print(targets.shape, targets.dtype)
            loss = self.criterion(output, targets)
            loss.backward()
            self.optim.step()

            train_loss.update(loss.item(), inputs.size(0))

            if epoch == 0 and i == 1:
                print('training is starting on {}'.format(self.device))

            return train_loss


    def validate(self, epoch):
        self.model.eval()

        val_loss = AverageMeter()
        inputs_all, gts_all, predictions_all = [], [], []

        for i, (inputs, gts) in enumerate(self.val_loader):
            N = inputs.size(0)
            inputs = inputs.to(self.device)
            gts = gts.to(self.device, dtype=torch.float32)

            outputs = self.model(inputs)
            predictions = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
            # print(predictions.shape, gts.shape)
            # print(torch.min(gts))
            # print(torch.max(torch.where(gts>=255, torch.zeros(gts.shape), gts)))

            gts = F.upsample(torch.unsqueeze(gts, 0), outputs.size()[2:], mode='nearest')
            gts = torch.squeeze(gts, 0).to(torch.int64)
            val_loss.update(self.criterion(outputs, gts).item(), N)

            if random.random() > 0.1:
                inputs_all.append(None)
            else:
                inputs_all.append(inputs.data.squeeze_(0).cpu())
            gts_all.append(gts.data.squeeze_(0).cpu().numpy())
            predictions_all.append(predictions)

        acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, self.num_classes)

        if self.visualize and epoch % 20 == 0:
            val_visual = []
            for i, data in enumerate(zip(inputs_all, gts_all, predictions_all)):
                if data[0] is None:
                    continue
                print(data[0].shape, type(data[0]), data[0].dtype)
                input_pil = restore_transform(data[0])
                gt_pil = colorize_mask(data[1])
                predictions_pil = colorize_mask(data[2])
                if train_args['val_save_to_img_file']:
                    input_pil.save('./image/{}/e_{}/{}_input.png'.format(self.save_name, epoch, i))
                    predictions_pil.save('./image/{}/e_{}/{}_prediction.png'.format(self.save_name, epoch, i))
                    gt_pil.save('./image/{}/e_{}/{}_gt.png'.format(self.save_name, epoch, i))
                val_visual.extend([visualize(input_pil.convert('RGB')), visualize(gt_pil.convert('RGB')),
                                   visualize(predictions_pil.convert('RGB'))])
            val_visual = torch.stack(val_visual, 0)
            val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)
            val_visual.save('./image/{}/sum_e_{}.png'.format(self.save_name, epoch))

        return val_loss, acc, acc_cls, mean_iu, fwavacc


    def report(self, metrics, epoch):
        print('[epoch %d], [train loss %.5f], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
            epoch, i + 1, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5]))

        if epoch % 20 == 0:
            torch.save(self.model.state_dict(), './model/u_net_{}.pth'.format(self.save_name))

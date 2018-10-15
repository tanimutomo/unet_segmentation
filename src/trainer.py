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

    def __init__(self, device, model, criterion, optim, train_loader, val_loader, params, experiment=None):
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
        self.experiment = experiment

    def iteration(self):
        self.model.apply(init_weights)
        for epoch in range(self.epochs):
            train_loss = self.train(epoch)
            # val_loss, acc, acc_cls, mean_iu, fwavacc = self.validate(epoch)
            val_loss, acc, acc_cls, mean_iu = self.validate(epoch)

            metrics = {
                    'train_loss': train_loss.avg,
                    'val_loss': val_loss.avg,
                    'acc': acc,
                    'acc_cls': acc_cls,
                    'mean_iu': mean_iu
                    # 'fwavacc': fwavacc
                    }

            print(metrics)

            if self.experiment is not None:
                self.experiment.log_multiple_metrics(metrics, step=epoch)

            self.report(metrics, epoch)

    def train(self, epoch):
        self.model.train()

        train_loss = AverageMeter()

        for i, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device, dtype=torch.float32)
            self.optim.zero_grad()
            outputs = self.model(inputs)

            # experiment for calcurate accuracy and visualize outputs
            # print(targets.shape, type(targets), targets.dtype)
            # out = outputs.clone().detach().cpu()
            # print(out.shape, type(out), out.dtype)
            # print(torch.min(out))
            # print(torch.max(out))

            targets = F.upsample(torch.unsqueeze(targets, 0), outputs.size()[2:], mode='nearest')
            targets = torch.squeeze(targets, 0).to(torch.int64)
            # print(output.shape, output.dtype)
            loss = self.criterion(outputs, targets)
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
            preds = torch.argmax(outputs, dim=1)
            gts = F.upsample(torch.unsqueeze(gts, 0), outputs.size()[2:], mode='nearest')
            gts = torch.squeeze(gts, 0).to(torch.int64)
            val_loss.update(self.criterion(outputs, gts).item(), N)
            acc, acc_cls, mean_iu = evaluate(preds.detach(), gts.detach(), self.num_classes)

        return val_loss, acc, acc_cls, mean_iu

        # if self.visualize and epoch % 20 == 0:
        #     val_visual = []
        #     for i, data in enumerate(zip(inputs_all, gts_all, predictions_all)):
        #         if data[0] is None:
        #             continue
        #         input_pil = restore_transform(data[0].detach())
        #         gt_pil = colorize_mask(data[1])
        #         predictions_pil = colorize_mask(data[2])

        #         input_pil.save('./image/{}/e_{}/{}_input.png'.format(self.save_name, epoch, i))
        #         predictions_pil.save('./image/{}/e_{}/{}_prediction.png'.format(self.save_name, epoch, i))
        #         gt_pil.save('./image/{}/e_{}/{}_gt.png'.format(self.save_name, epoch, i))
        #         val_visual.extend([visualize(input_pil.convert('RGB')), visualize(gt_pil.convert('RGB')),
        #                            visualize(predictions_pil.convert('RGB'))])
        #     val_visual = torch.stack(val_visual, 0)
        #     val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)
        #     val_visual.save('./image/{}/sum_e_{}.png'.format(self.save_name, epoch))

        # return val_loss, acc, acc_cls, mean_iu, fwavacc


    def report(self, metrics, epoch):
        print('[epoch %d], [train loss %.5f], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f]' % (
            epoch, metrics['train_loss'], metrics['val_loss'], metrics['acc'], metrics['acc_cls'], metrics['mean_iu']))
        print('[epoch %d]\t[train loss %.5f]\t[val loss %.5f]' % (epoch, metrics['train_loss'], metrics['val_loss']))

        if epoch % 20 == 0 or epoch == (self.epochs - 1):
            torch.save(self.model.state_dict(), './model/u_net_{}.pth'.format(self.save_name))
            print('model saved')

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



class Trainer(object):

    def __init__(self, device, model, criterion, optim, train_loader, val_loader, params, experiment=None):
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optim = optim
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = params['epochs']
        self.save_name = params['save_name']
        self.num_classes = params['num_classes']
        self.experiment = experiment

    def iteration(self):
        self.model.to(self.device)
        self.model.apply(init_weights.to(self.device))
        for epoch in range(self.epochs):
            train_loss, train_acc, train_acc_cls, train_mean_iu = self.train(epoch)
            # val_loss, acc, acc_cls, mean_iu, fwavacc = self.validate(epoch)
            val_loss, val_acc, val_acc_cls, val_mean_iu = self.validate(epoch)

            metrics = {
                    'train_loss': train_loss.avg,
                    'train_acc': train_acc.avg,
                    'train_acc_cls': train_acc_cls.avg,
                    'train_mean_iu': train_mean_iu.avg,
                    'val_loss': val_loss.avg,
                    'val_acc': val_acc.avg,
                    'val_acc_cls': val_acc_cls.avg,
                    'val_mean_iu': val_mean_iu.avg
                    # 'fwavacc': fwavacc
                    }

            if self.experiment is not None:
                self.experiment.log_multiple_metrics(metrics, step=epoch)

            self.report(metrics, epoch)

    def train(self, epoch):
        self.model.train()

        train_loss = AverageMeter()
        train_acc = AverageMeter()
        train_acc_cls = AverageMeter()
        train_mean_iu = AverageMeter()

        for i, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            print(type(intpus))
            # targets = targets.to(self.device, dtype=torch.float32)
            self.optim.zero_grad()
            outputs = self.model(inputs)
            preds = torch.argmax(outputs, dim=1)

            # targets = F.upsample(torch.unsqueeze(targets, 0), outputs.size()[2:], mode='nearest')
            # targets = torch.squeeze(targets, 0).to(torch.int64)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optim.step()

            train_loss.update(loss.item(), inputs.size(0))
            train_metric = evaluate(preds.detach(), targets.detach(), self.num_classes)
            train_acc.update(train_metric[0])
            train_acc_cls.update(train_metric[1])
            train_mean_iu.update(train_metric[2])

            if epoch == 0 and i == 1:
                print('iteration is started on {}'.format(self.device))

        return train_loss, train_acc, train_acc_cls, train_mean_iu


    def validate(self, epoch):
        self.model.eval()

        val_loss = AverageMeter()
        val_acc = AverageMeter()
        val_acc_cls = AverageMeter()
        val_mean_iu = AverageMeter()
        # inputs_all, gts_all, predictions_all = [], [], []

        for i, (inputs, gts) in enumerate(self.val_loader):
            N = inputs.size(0)
            inputs = inputs.to(self.device)
            gts = gts.to(self.device)
            # gts = gts.to(self.device, dtype=torch.float32)

            outputs = self.model(inputs)
            preds = torch.argmax(outputs, dim=1)
            # gts = F.upsample(torch.unsqueeze(gts, 0), outputs.size()[2:], mode='nearest')
            # gts = torch.squeeze(gts, 0).to(torch.int64)
            val_loss.update(self.criterion(outputs, gts).item(), N)
            val_metric = evaluate(preds.detach(), gts.detach(), self.num_classes)
            val_acc.update(val_metric[0])
            val_acc_cls.update(val_metric[1])
            val_mean_iu.update(val_metric[2])

        return val_loss, val_acc, val_acc_cls, val_mean_iu


    def report(self, metrics, epoch):
        print('Train: [epoch %d], [loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f]' % (
            epoch, metrics['train_loss'], metrics['train_acc'], metrics['train_acc_cls'], metrics['train_mean_iu']))
        print('Val  : [epoch %d], [loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f]' % (
            epoch, metrics['val_loss'], metrics['val_acc'], metrics['val_acc_cls'], metrics['val_mean_iu']))
        # print('[epoch %d]\t[train loss %.5f]\t[val loss %.5f]' % (epoch, metrics['train_loss'], metrics['val_loss']))

        if epoch % 20 == 0 or epoch == (self.epochs - 1):
            dir = './model/{}/'.format(self.save_name)
            if os.path.exists(dir) == 0:
                os.mkdir(dir)
            torch.save(self.model.state_dict(), '{}u_net_{}.pth'.format(dir, epoch))
            print('model is saved at {}'.format(dir))

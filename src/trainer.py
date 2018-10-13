import time
from datetime import timedelta
import os
import torch
import torch.nn as nn
from torch import optim
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

        # self.timestamp_s = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        # self.epochs = 0
        # self.iteration = 0
        # self.max_iter = max_iter

    def train(self):
        self.model.apply(init_weights)
        self.model.train()
        iters = 10000
        epochs = 1
        for epoch in range(epochs):
            for i, (input, target) in enumerate(self.train_loader):
                input = input.to(self.device)
                target = target.to(self.device)
                train_start_time = time.time()
                self.optim.zero_grad()
                output = self.model(input)
                print(output.shape, target.shape)
                loss = self.criterion(output, target)
                loss.backward()
                self.optim.step()

                seconds = time.time() - train_start_time
                elapsed = str(timedelta(seconds=seconds))
                print('Iteration : [{iter}/{iters}]\t'
                      'Time : {time}\t'
                      'Loss : {loss:.4f}\t'.format(
                      iter=i+1, iters=iters,
                      time=elapsed, loss=loss.item()))

        if os.path.exists('./src/model'):
            os.mkdir('./src/model')
        torch.save(self.model.state_dict(), './src/model/u_net_{}.pth'.format(self.save_name))


        # iters_per_epoch = len(self.train_data_loader.dataset) // self.cfg.train_batch_size
        # if len(self.train_data_loader.dataset) % self.cfg.train_batch_size != 0:
        #     iters_per_epoch += 1

        # epoch = 1

        # # torch.cuda.synchronize()  # parallel mode
        # self.model.train()

        # train_start_time = time.time()
        # data_iter = iter(self.train_data_loader)

        # for n_iter in range(self.cfg.n_iters):
        #     self.scheduler.step()
        #     try:
        #         input, target = next(data_iter)
        #     except:
        #         data_iter = iter(self.train_data_loader)
        #         input, target = next(data_iter)

        #     input_var = input.clone().to(self.device)
        #     target_var = target.to(self.device)
        #     output = self.model(input_var)
        #     # output = output.view(output.size(0), output.size(1), -1)
        #     # target_var = target_var.view(target_var.size(0), -1)
        #     loss = self.c_loss(output, target_var)

        #     self.reset_grad()
        #     loss.backward()
        #     self.optim.step()
        #     # print('Done')

        #     # output_label = torch.argmax(_output, dim=1)

        #     if (n_iter + 1) % self.cfg.log_step == 0:
        #         seconds = time.time() - train_start_time
        #         elapsed = str(timedelta(seconds=seconds))
        #         print('Iteration : [{iter}/{iters}]\t'
        #               'Time : {time}\t'
        #               'Loss : {loss:.4f}\t'.format(
        #               iter=n_iter+1, iters=self.cfg.n_iters,
        #               time=elapsed, loss=loss.item()))
        #         try:
        #             nsml.report(
        #                     train__loss=loss.item(),
        #                     step=n_iter+1)
        #         except ImportError:
        #             pass

        #     if (n_iter + 1) % iters_per_epoch == 0:
        #         self.validate(epoch)
        #         epoch += 1



#     def validate(self, epoch):
# 
#         self.model.eval()
#         val_start_time = time.time()
#         data_iter = iter(self.val_data_loader)
#         max_iter = len(self.val_data_loader)
#         # for n_iter in range(max_iter): #FIXME
#         n_iter =  0
#         input, target = next(data_iter)
# 
#         input_var = input.clone().to(self.device)
#         target_var = target.to(self.device)
# 
#         output = self.model(input_var)
#         _output = output.clone()
#         # output = output.view(output.size(0), output.size(1), -1)
#         # target_var = target_var.view(target_var.size(0), -1)
#         loss = self.c_loss(output, target_var)
# 
#         output_label = torch.argmax(_output, dim=1)
# 
#         if (n_iter + 1) % self.cfg.log_step == 0:
#             seconds = time.time() - val_start_time
#             elapsed = str(timedelta(seconds=seconds))
#             print('### Validation\t'
#                   'Iteration : [{iter}/{iters}]\t'
#                   'Time : {time:}\t'
#                   'Loss : {loss:.4f}\t'.format(
#                   iter=n_iter+1, iters=max_iter,
#                   time=elapsed, loss=loss.item()))
#             try:
#                 nsml.report(
#                         val__loss=loss.item(),
#                         step=epoch)
#             except ImportError:
# pass






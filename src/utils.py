import torch.nn as nn

def init_weights(m):
    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')



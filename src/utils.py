import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
    # elif type(m) == nn.BatchNorm2d:
    #     nn.init.uniform_(m, a=0, b=1)


def crop_to_square(image):
    size = min(image.size)
    left, upper = (image.width - size) // 2, (image.height - size) // 2
    right, bottom = (image.width + size) // 2, (image.height + size) // 2
    return image.crop((left, upper, right, bottom))
    

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def make_confmat(preds, gts, num_cls):
    preds_flat = torch.flatten(preds)
    gts_flat = torch.flatten(gts)
    mask = (gts_flat >= 0) & (gts_flat < num_cls)
    confmat = torch.bincount(num_cls * gts_flat[mask] + preds_flat[mask], minlength=num_cls ** 2)
    confmat = confmat.reshape(num_cls, num_cls)
    return confmat


def evaluate(preds, gts, num_cls):
    confmat = make_confmat(preds, gts, num_cls)
    acc = torch.diag(confmat).sum() / torch.sum(confmat)
    acc_cls, iu = 0, 0
    for i in range(num_cls):
        correct = torch.diag(confmat)[i]
        sum_pred = torch.sum(confmat, dim=1)[i]
        sum_gt = torch.sum(confmat, dim=0)[i]
        sum_ = sum_pred + sum_gt
        if sum_pred != 0:
            acc_cls += correct / sum_pred
        if sum_ != 0:
            iu += correct / (sum_ - correct)
    acc_cls = torch.mean(acc_cls.float())
    mean_iu = torch.mean(iu.float())

    return acc, acc_cls, mean_iu


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

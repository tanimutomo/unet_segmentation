import torch
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.transforms.functional as F

from PIL import Image

# from src.dutils import transforms as extended_transforms
# from src.dutils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d

from src.dataset import VOC
from src.utils import MaskToTensor, DeNormalize


mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def get_loader(root, batch_size, init_size):

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ])

    target_transform = MaskToTensor()

    augmentation = transforms.Compose([
        transforms.RandomAffine(degrees=45, translate=(0.3, 0.3)),
        transforms.RandomHorizontalFlip(p=0.5)
        ])

    train_set = VOC('train', root, init_size, transform=input_transform, target_transform=target_transform, augmentation=augmentation)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True)
    val_set = VOC('val', root, init_size, transform=input_transform, target_transform=target_transform, augmentation=augmentation)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=0, shuffle=False)

    return train_loader, val_loader

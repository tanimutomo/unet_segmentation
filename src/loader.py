import torch
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
import torchvision.transforms.functional as F

from src.dutils import transforms as extended_transforms
from src.dutils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d
from src.dutils import transforms as extended_transforms

from src.dataset import VOC

def get_loader(root, batch_size):
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()
    restore_transform = standard_transforms.Compose([
        extended_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage(),
    ])
    visualize = standard_transforms.Compose([
        standard_transforms.Resize(400),
        standard_transforms.CenterCrop(400),
        standard_transforms.ToTensor()
    ])

    train_set = VOC('train', root, transform=input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, shuffle=True)
    val_set = VOC('val', root, transform=input_transform, target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4, shuffle=False)

    return train_loader, val_loader

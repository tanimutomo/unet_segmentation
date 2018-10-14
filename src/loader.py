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

visualize = transforms.Compose([
    transforms.Resize(400),
    transforms.CenterCrop(400),
    transforms.ToTensor()
])

restore_transform = transforms.Compose([
    DeNormalize(*mean_std),
    transforms.ToPILImage(),
])

def get_loader(root, batch_size, init_size):

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ])

    target_transform = MaskToTensor()


    train_set = VOC('train', root, init_size, transform=input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True)
    val_set = VOC('val', root, init_size, transform=input_transform, target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=0, shuffle=False)

    return train_loader, val_loader

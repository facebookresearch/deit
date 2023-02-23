import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torchvision.models as tmodels
from functools import partial
from tools.models import *
from tools.pruners import prune_weights_reparam

def model_and_opt_loader(model_string,DEVICE):
    if DEVICE == None:
        raise ValueError('No cuda device!')
    if model_string == 'vgg16':
        model = VGG16().to(DEVICE)
        amount = 0.20
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 50000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 40000,
            "scheduler": None
        }
    elif model_string == 'resnet18':
        model = ResNet18().to(DEVICE)
        amount = 0.20
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 50000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 40000,
            "scheduler": None
        }
    elif model_string == 'densenet':
        model = DenseNet121().to(DEVICE)
        amount = 0.20
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 80000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 60000,
            "scheduler": None
        }
    elif model_string == 'effnet':
        model = EfficientNetB0().to(DEVICE)
        amount = 0.20
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 50000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 40000,
            "scheduler": None
        }
    else:
        raise ValueError('Unknown model')
    prune_weights_reparam(model)
    return model,amount,batch_size,opt_pre,opt_post
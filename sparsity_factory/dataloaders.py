import torch
from tools.datasets import *

data_route      = './data/'
cifar10_strings = ['vgg16','resnet18','densenet','effnet']

def dataset_loader(model,batch_size=100,num_workers=5):
    if model in cifar10_strings:
        print("Loading CIFAR-10 with batch size "+str(batch_size))
        train_loader,test_loader = get_cifar10_loaders(data_route,batch_size,num_workers)
    else:
        raise ValueError('Model not implemented :P')
    return train_loader,test_loader

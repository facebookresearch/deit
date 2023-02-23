import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dts
import torchvision.transforms as T

cifar_nm    = T.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))

def get_cifar10_loaders(data_route,batch_size,num_workers):
    tfm_train = T.Compose([T.RandomCrop(32, padding=4),T.RandomHorizontalFlip(),T.ToTensor(),cifar_nm])
    tfm_test = T.Compose([T.ToTensor(),cifar_nm])
    
    train_set = dts.CIFAR10(data_route,train=True,download=True,transform=tfm_train)
    test_set = dts.CIFAR10(data_route,train=False,download=False,transform=tfm_test)
    
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=num_workers)
    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,drop_last=False,num_workers=num_workers)
    
    return train_loader,test_loader
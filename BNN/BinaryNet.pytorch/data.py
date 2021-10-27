import torch
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


import numpy as np
import _pickle as cPickle


def output_label(yy):

    data = list(yy)
    yy1 = np.zeros([len(data), max(data)+1])
    yy1[np.arange(len(data)), data] = 1
    return yy1

def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True):
    train = (split == 'train')
    if name == 'cifar10':
        return datasets.CIFAR10(root='./dataset/',
                                train=train,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)
    elif name == 'cifar100':
        return datasets.CIFAR100(root='./dataset/',
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'imagenet':
        path = './dataset/'
        return datasets.ImageFolder(root=path,
                                    transform=transform,
                                    target_transform=target_transform)

    elif name == 'rml':

        X_train_tensor = torch.load("/home/nitin/Research/RML_Research/Dataset/2018.01/X_train.pt")
        X_test_tensor = torch.load("/home/nitin/Research/RML_Research/Dataset/2018.01/X_test_10.pt")
        Y_train_tensor = torch.load("/home/nitin/Research/RML_Research/Dataset/2018.01/Y_train.pt")
        Y_test_tensor = torch.load("/home/nitin/Research/RML_Research/Dataset/2018.01/Y_test_10.pt") 

        # X_train_tensor = torch.load("/home/nitin/Research/RML_Research/RBENN/Dataset/X_train.pt")
        # X_test_tensor = torch.load("/home/nitin/Research/RML_Research/RBENN/Dataset/X_test.pt")
        # Y_train_tensor = torch.load("/home/nitin/Research/RML_Research/RBENN/Dataset/Y_train.pt")
        # Y_test_tensor = torch.load("/home/nitin/Research/RML_Research/RBENN/Dataset/Y_test.pt")
            
        # print(indices)
        
        # X_train = X_train_tensor[(indices,)]
        # Y_train = Y_train_tensor[(indices,)]
        
        X_train = X_train_tensor
        Y_train = Y_train_tensor
        # print(X_train_tensor.shape)
        # print(Y_train_tensor.shape)
        trainset = TensorDataset(X_train,Y_train)

            
        testset = TensorDataset(X_test_tensor,Y_test_tensor)
 

        if split == 'train':
            return trainset

        elif split == 'val':       
            return testset

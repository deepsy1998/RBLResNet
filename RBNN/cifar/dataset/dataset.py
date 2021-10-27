from datetime import datetime
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import _pickle as cPickle


# Defining the output labels for the train and test data sets


def output_label(yy):

    data = list(yy)
    yy1 = np.zeros([len(data), max(data)+1])
    yy1[np.arange(len(data)), data] = 1
    return yy1

def load_data(type='both',dataset='cifar10',data_path='/data',batch_size = 256,batch_size_test=256,num_workers=0):
    # load data
    
    if dataset == 'rml':
        
        X_train_tensor = torch.load("/home/nitin/Research/RML_Research/Dataset/2018.01/X_train.pt")
        X_test_tensor = torch.load("/home/nitin/Research/RML_Research/Dataset/2018.01/X_test_4.pt")
        Y_train_tensor = torch.load("/home/nitin/Research/RML_Research/Dataset/2018.01/Y_train.pt")
        Y_test_tensor = torch.load("/home/nitin/Research/RML_Research/Dataset/2018.01/Y_test_4.pt") 
            
        # X_train_tensor = torch.load("/home/nitin/RML_Research/RBENN/Dataset/X_train.pt")
        # X_test_tensor = torch.load("/home/nitin/RML_Research/RBENN/Dataset/X_test.pt")
        # Y_train_tensor = torch.load("/home/nitin/RML_Research/RBENN/Dataset/Y_train.pt")
        # Y_test_tensor = torch.load("/home/nitin/RML_Research/RBENN/Dataset/Y_test.pt")

        # print(indices)
       
        # X_train = X_train_tensor[(indices,)]
        # Y_train = Y_train_tensor[(indices,)]

        X_train = X_train_tensor
        # print(X_train.shape)
        # exit()
        Y_train = Y_train_tensor
        # print(Y_train.shape)
        # exit()

        # print(Y_train_tensor.shape)
        trainset = TensorDataset(X_train,Y_train)
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True)
            
        testset = TensorDataset(X_test_tensor,Y_test_tensor)
        testloader = DataLoader(
            testset,
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True)

        if type=='both':
                return trainloader, testloader
        elif type=='train':
                return trainloader
        elif type=='val':
                return testloader

    # elif dataset == 'rml':

    #     X_train_tensor = torch.load("/home/nitin/RML_Research/Dataset/2018.01/X_train.pt")
    #     X_test_tensor = torch.load("/home/nitin/RML_Research/Dataset/2018.01/X_test.pt")
    #     Y_train_tensor = torch.load("/home/nitin/RML_Research/Dataset/2018.01/Y_train.pt")
    #     Y_test_tensor = torch.load("/home/nitin/RML_Research/Dataset/2018.01/Y_test.pt")

    #     X_train = X_train_tensor
    #     Y_train = Y_train_tensor

    #     # print(Y_train_tensor.shape)
    #     trainset = TensorDataset(X_train,Y_train)
    #     trainloader = DataLoader(
    #         trainset,
    #         batch_size=batch_size,
    #         shuffle=True,
    #         num_workers=num_workers,
    #         pin_memory=True)
            
    #     testset = TensorDataset(X_test_tensor,Y_test_tensor)
    #     testloader = DataLoader(
    #         testset,
    #         batch_size=batch_size_test,
    #         shuffle=False,
    #         num_workers=num_workers,
    #         pin_memory=True)

    #     if type=='both':
    #             return trainloader, testloader
    #     elif type=='train':
    #             return trainloader
    #     elif type=='val':
    #             return testloader
        
    # elif dataset == 'rml':
    #     # Load the dataset downloaded from https://www.deepsig.io/datasets

    #     with open("/content/drive/MyDrive/RML_Research/Dataset/RML2016.10a_dict.pkl", 'rb') as f:
    #         Xd = cPickle.load(f, encoding="latin1")

    #     # Preprocessing the data
    #     # Separate the SNR and Modulation into snrs and mods lists from the key of Xd

    #     snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])

    #     X = []
    #     lbl = []
    #     for mod in mods:
    #         # mod is the label. mod = modulation scheme
    #         for snr in snrs:
    #             X.append(Xd[(mod, snr)])
    #     #         #snr = signal to noise ratio
    #             for i in range(Xd[(mod, snr)].shape[0]):
    #                 lbl.append((mod, snr))
    #     X = np.vstack(X)

    #     X = (X - np.average(X))/np.std(X)

    #     X = np.expand_dims(X, axis=1)
    #     # Partition the data into training and test sets
    #     # Taking 75% of the samples for the train set & 25% for the test set


    #     np.random.seed(2016)
    #     n_examples = X.shape[0]

    #     n_train = int(n_examples * 0.75)
    #     train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
    #     test_idx = list(set(range(0, n_examples))-set(train_idx))
    #     X_train = X[train_idx]
    #     X_test =  X[test_idx]


    #     Y_train = output_label(map(lambda x: mods.index(lbl[x][0]), train_idx))
    #     Y_train = np.argmax(Y_train, axis=1)

    #     Y_test = output_label(map(lambda x: mods.index(lbl[x][0]), test_idx))
    #     Y_test = np.argmax(Y_test, axis=1)


    #     trainset = TensorDataset(torch.Tensor(X_train),torch.Tensor(Y_train))
    #     trainloader = DataLoader(
    #         trainset,
    #         batch_size=batch_size,
    #         shuffle=True,
    #         num_workers=num_workers,
    #         pin_memory=True)

    #     testset = TensorDataset(torch.Tensor(X_test),torch.Tensor(Y_test))
    #     testloader = DataLoader(
    #         testset,
    #         batch_size=batch_size_test,
    #         shuffle=False,
    #         num_workers=num_workers,
    #         pin_memory=True)

    #     if type=='both':
    #             return trainloader, testloader
    #     elif type=='train':
    #             return trainloader
    #     elif type=='val':
    #             return testloader

    # else:
        
    #     param = {'cifar10':{'name':datasets.CIFAR10,'size':32,'normalize':[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]},
    #             'cifar100':{'name':datasets.CIFAR100,'size':32,'normalize':[(0.507, 0.487, 0.441), (0.267, 0.256, 0.276)]},
    #             'mnist':{'name':datasets.MNIST,'size':32,'normalize':[(0.5,0.5,0.5),(0.5,0.5,0.5)]},
    #             'tinyimagenet':{'name':datasets.ImageFolder,'size':224,'normalize':[(0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)]}}

        
    #     data = param[dataset]

    #     if data['name']==datasets.ImageFolder:
    #         data_transforms = {
    #             'train': transforms.Compose([
    #                 transforms.Resize(data['size']),
    #                 transforms.RandomRotation(20),
    #                 transforms.RandomHorizontalFlip(0.5),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(*data['normalize']),
    #             ]),
    #             'val': transforms.Compose([
    #                 transforms.Resize(data['size']),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(*data['normalize']),
    #             ]),
    #             'test': transforms.Compose([
    #                 transforms.Resize(data['size']),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(*data['normalize']),
    #             ])
    #         }
    #         data_dir = os.path.join(data_path,'tiny-imagenet-200')
    #         image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
    #                         for x in ['train', 'val']}
    #         dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=num_workers)
    #                         for x in ['train', 'val']}
    #         return dataloaders.values()
        
    #     else:

    #         transform1 = transforms.Compose([
    #             transforms.RandomCrop(data['size'],padding=4),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize(*data['normalize']),
    #         ])
    #         transform2 = transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize(*data['normalize']),
    #         ])

    #         trainset = data['name'](root=data_path,
    #                                     train=True,
    #                                     download=True,
    #                                     transform=transform1);
    #         trainloader = DataLoader(
    #             trainset,
    #             batch_size=batch_size,
    #             shuffle=True,
    #             num_workers=num_workers,
    #             pin_memory=True)

    #         testset = data['name'](root=data_path,
    #                                 train=False,
    #                                 download=True,
    #                                 transform=transform2);
    #         testloader = DataLoader(
    #             testset,
    #             batch_size=batch_size_test,
    #             shuffle=False,
    #             num_workers=num_workers,
    #             pin_memory=True)

    #         if type=='both':
    #             return trainloader, testloader
    #         elif type=='train':
    #             return trainloader
    #         elif type=='val':
    #             return testloader
        


    

def delete_module_fromdict(statedict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k,v in statedict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict

def add_module_fromdict(statedict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k,v in statedict.items():
        name = 'module.'+k
        new_state_dict[name] = v
    return new_state_dict

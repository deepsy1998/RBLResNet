'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import *


__all__ =['resnet18A_LSTM','resnet18A_1w1a_LSTM','light','resnet18A_compare','MC_net_10blocks_binary','MC_net_10blocksv2','MC_net_6blocks','rml_resnet_1w1av2','rml_resnet','resnet18Av2','resnet18A_1w1a','resnet18B_1w1a','resnet18C_1w1a','resnet18_1w1a','resnet18A_1w1a_U','resnet18A_1w1a_K']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock_compare(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_compare, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (planes - in_planes)//2, (planes - in_planes)//2 ), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        # print('3', x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.bn2(self.conv2(out))
        # print('1', out.shape)
        # print('2',  self.shortcut(x).shape)
        out += self.shortcut(x)
        out = self.relu2(out)
        return out





class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding=1):
        super(BasicBlock, self).__init__()
        self.conv1 = BinarizeConv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.Hardtanh(inplace=True)
        if stride == 2: 
            self.conv2 = BinarizeConv2d(planes, planes, kernel_size=(1,3), stride=1, padding=(0,1), bias=False)
        else:
            self.conv2 = BinarizeConv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        # self.conv2 = BinarizeConv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.tanh2 = nn.Hardtanh(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                BinarizeConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # print(x.shape)
        out = self.tanh1(self.bn1(self.conv1(x)))
        # print(out.shape)
        out = self.bn2(self.conv2(out))
        # print(out.shape)
        # print(self.shortcut(x).shape)
        out += self.shortcut(x)
        out = self.tanh2(out)
        return out

class BasicBlock_real(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding=1):
        super(BasicBlock_real, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.Hardtanh(inplace=True)
        
        if stride == 2: 
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=(1,3), stride=1, padding=(0,1), bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.tanh2 = nn.Hardtanh(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

    def forward(self, x):
        out = self.tanh1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.tanh2(out)
        return out




class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = BinarizeConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = BinarizeConv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                BinarizeConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_paper(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_paper, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(2,3), stride=stride, padding=0, bias=False)

        self.tanh1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(2,3), stride=1, padding=0, bias=False)

        self.tanh2 = nn.ReLU(inplace=False)

        self.shortcut = nn.Sequential()
        

    def forward(self, x):
        # print(x.shape)
        out = F.pad(x,(1,1,0,1))
        out = self.conv1(out)
        # print(out.shape)
        out = self.tanh1(out)
        # print(out.shape)
        out = F.pad(out,(1,1,0,1))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.tanh2(out)
        return out

class BasicBlock_ppr_1w1a(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_ppr_1w1a, self).__init__()
        self.conv1 = BinarizeConv2d(in_planes, planes, kernel_size=(2,3), stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.ReLU(inplace=False)

        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=(2,3), stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.tanh2 = nn.ReLU(inplace=False)

        self.shortcut = nn.Sequential()
        

    def forward(self, x):
        # print(x.shape)
        out = F.pad(x,(1,1,0,1))
        out = self.conv1(out)
        out = self.bn1(out)
        # print(out.shape)
        out = self.tanh1(out)
        # print(out.shape)
        out = F.pad(out,(1,1,0,1))
        out = self.conv2(out)
        out = self.bn2(out)
        # print(out.shape)
        out += self.shortcut(x)
        out = self.tanh2(out)
        return out



class ResNet_rml_binary(nn.Module):
    def __init__(self, block, num_blocks, num_channel, num_classes=10):
        super(ResNet_rml_binary, self).__init__()
        self.in_planes = num_channel[0]

        self.conv1   = nn.Conv2d(1, num_channel[0], kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channel[0])
        self.layer1  = self._make_layer(block, num_channel[0], num_blocks[0], stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size= (2,2), stride=(1,2), padding=0)

        self.conv2   = BinarizeConv2d(32, num_channel[0], kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channel[0])
        self.layer2  = self._make_layer(block, num_channel[1], num_blocks[1],  stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size= (1,2), stride=(1,2), padding=0)

        self.conv3   = BinarizeConv2d(32, num_channel[0], kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(num_channel[0])
        self.layer3  = self._make_layer(block, num_channel[2], num_blocks[2], stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size= (1,2), stride=(1,2), padding=0)

        self.conv4   = BinarizeConv2d(32, num_channel[0], kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(num_channel[0])
        self.layer4  = self._make_layer(block, num_channel[3], num_blocks[3], stride=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size= (1,2), stride=(1,2), padding=0)

        self.conv5   = BinarizeConv2d(32, num_channel[0], kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(num_channel[0])
        self.layer5  = self._make_layer(block, num_channel[3], num_blocks[3], stride=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size= (1,2), stride=(1,2), padding=0)

        self.conv6   = BinarizeConv2d(32, num_channel[0], kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(num_channel[0])
        self.layer6  = self._make_layer(block, num_channel[3], num_blocks[3], stride=1)
        self.maxpool6 = nn.MaxPool2d(kernel_size= (1,2), stride=(1,2), padding=0)

        self.flat = nn.Flatten()
        self.bn7 = nn.BatchNorm1d(64)
        self.linear1  = nn.Linear(64, 128)
        self.act1 = nn.SELU(inplace=False)

        self.bn8 = nn.BatchNorm1d(128)
        self.linear2  = nn.Linear(128, 128)
        self.act2 = nn.SELU(inplace=False)

        self.bn9 = nn.BatchNorm1d(128)
        self.linear3  = nn.Linear(128, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.layer1(out)
        out = self.maxpool1(out)
        # print(out.shape)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.layer2(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.layer3(out)
        out = self.maxpool3(out)
    
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.layer4(out)
        out = self.maxpool4(out)
        # print(out.shape)

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.layer5(out)
        out = self.maxpool5(out)

        out = self.conv6(out)
        out = self.bn6(out)
        out = self.layer6(out)
        out = self.maxpool6(out)


        out = self.flat(out)
        # print(out.shape)
        out = self.bn7(out)
        out = self.linear1(out)
        # print(out.shape)
        out = self.act1(out)

        # print(out.shape)
        out = self.bn8(out)
        out = self.linear2(out)
        # print(out.shape)
        out = self.act2(out)
        
        # print(out.shape)
        out = self.bn9(out)
        out = self.linear3(out)
        # print(out.shape)


        return out 

class ResNet_rml(nn.Module):
    def __init__(self, block, num_blocks, num_channel, num_classes=10):
        super(ResNet_rml, self).__init__()
        self.in_planes = num_channel[0]

        self.conv1   = nn.Conv2d(1, num_channel[0], kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.layer1  = self._make_layer(block, num_channel[0], num_blocks[0], stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size= (2,2), stride=(1,2), padding=0)

        self.conv2   = nn.Conv2d(32, num_channel[0], kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.layer2  = self._make_layer(block, num_channel[1], num_blocks[1], stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size= (1,2), stride=(1,2), padding=0)

        self.conv3   = nn.Conv2d(32, num_channel[0], kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.layer3  = self._make_layer(block, num_channel[2], num_blocks[2], stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size= (1,2), stride=(1,2), padding=0)

        self.conv4   = nn.Conv2d(32, num_channel[0], kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.layer4  = self._make_layer(block, num_channel[3], num_blocks[3], stride=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size= (1,2), stride=(1,2), padding=0)

        self.conv5   = nn.Conv2d(32, num_channel[0], kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.layer5  = self._make_layer(block, num_channel[3], num_blocks[3], stride=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size= (1,2), stride=(1,2), padding=0)

        self.conv6   = nn.Conv2d(32, num_channel[0], kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.layer6  = self._make_layer(block, num_channel[3], num_blocks[3], stride=1)
        self.maxpool6 = nn.MaxPool2d(kernel_size= (1,2), stride=(1,2), padding=0)

        self.flat = nn.Flatten()

        self.linear1  = nn.Linear(512, 128)
        self.act1 = nn.SELU(inplace=False)

        self.linear2  = nn.Linear(128, 128)
        self.act2 = nn.SELU(inplace=False)

        self.linear3  = nn.Linear(128, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        out = self.conv1(x)
        # print(x.shape)
        out = self.layer1(out)
        out = self.maxpool1(out)
        # print(out.shape)

        out = self.conv2(out)
        out = self.layer2(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        out = self.layer3(out)
        out = self.maxpool3(out)
    
        out = self.conv4(out)
        out = self.layer4(out)
        out = self.maxpool4(out)
        # print(out.shape)

        out = self.conv5(out)
        out = self.layer5(out)
        out = self.maxpool5(out)

        out = self.conv6(out)
        out = self.layer6(out)
        out = self.maxpool6(out)


        out = self.flat(out)
        # print(out.shape)
        out = self.linear1(out)
        # print(out.shape)
        out = self.act1(out)

        # print(out.shape)
        out = self.linear2(out)
        # print(out.shape)
        out = self.act2(out)
        
        # print(out.shape)
        out = self.linear3(out)
        # print(out.shape)


        return out 

class ResNet_compare(nn.Module):
    def __init__(self, block, num_blocks, num_channel, num_classes=10):
        super(ResNet_compare, self).__init__()

        self.in_planes = num_channel[0]
        self.conv1 = nn.Conv2d(1, num_channel[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channel[0])
        self.layer1 = self._make_layer(block, num_channel[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, num_channel[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, num_channel[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, num_channel[3], num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool1d(128)
        self.bn2     = nn.BatchNorm1d(num_channel[3]*block.expansion)
        self.linear  = nn.Linear(num_channel[3]*block.expansion, num_classes)
               
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        print(x.shape)
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.squeeze(out,dim=2)
        # print(out.shape)
        out = self.avgpool(out)
        # print(out.shape)
        out = torch.squeeze(out) 
        out = self.bn2(out)
        # print(out.shape)
        out = self.linear(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channel, num_classes=10):
        super(ResNet, self).__init__()

        self.in_planes = num_channel[0]
        self.conv1   = nn.Conv2d(1, num_channel[0], kernel_size=(3,3), stride=1, padding="same", bias=False)
        self.bn1     = nn.BatchNorm2d(num_channel[0])
        self.layer1  = self._make_layer(block, num_channel[0], num_blocks[0], kernel_size=(3,3), stride=1, padding=1)
        self.layer2  = self._make_layer(block, num_channel[1], num_blocks[1], kernel_size=(3,3), stride=2, padding=1)
        self.layer3  = self._make_layer(block, num_channel[2], num_blocks[2], kernel_size=(1,3), stride=2, padding=(0,1))
        self.layer4  = self._make_layer(block, num_channel[3], num_blocks[3], kernel_size=(1,3), stride=2, padding=(0,1))
        # self.layer3  = self._make_layer(block, num_channel[2], num_blocks[2], kernel_size=(3,3), stride=2, padding=1)
        # self.layer4  = self._make_layer(block, num_channel[3], num_blocks[3], kernel_size=(3,3), stride=2, padding=1)
        self.avgpool = nn.AvgPool1d(128)
        self.bn2     = nn.BatchNorm1d(num_channel[3]*block.expansion)
        self.linear  = nn.Linear(num_channel[3]*block.expansion, num_classes)
        # self.dropout = nn.Dropout(p=0.5, inplace=False)
       
    def _make_layer(self, block, planes, num_blocks, kernel_size, stride, padding):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride, padding))
            self.in_planes = planes * block.expansion
            if stride == 2: 
                kernel_size = (1,3)
                padding = (0,1)
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        out = self.bn1(self.conv1(x))
        # print(out.shape)
        out = self.layer1(out)
        # print("hi")
        out = self.layer2(out)
        # print("hi")
        out = self.layer3(out)
        # print("hi")
        out = self.layer4(out)
        # print("hi")
        # print(out.shape)
        out = torch.squeeze(out,dim=2)
        # print(out.shape)
        out = self.avgpool(out)
        # print(out.shape)
        out = torch.squeeze(out,dim=2)
        # print(out.shape)
        out = self.bn2(out)
        # print(out.shape)
        out = self.linear(out)
        # out = self.dropout(out)
        
        return out 

class ResNet_LSTM(nn.Module):
    def __init__(self, block, num_blocks, num_channel, num_classes=10):
        super(ResNet_LSTM, self).__init__()

        self.hidden_size = 64

        self.in_planes = num_channel[0]
        self.conv1   = nn.Conv2d(1, num_channel[0], kernel_size=(3,3), stride=1, padding="same", bias=False)
        self.bn1     = nn.BatchNorm2d(num_channel[0])
        self.layer1  = self._make_layer(block, num_channel[0], num_blocks[0], kernel_size=(3,3), stride=1, padding=1)
        self.layer2  = self._make_layer(block, num_channel[1], num_blocks[1], kernel_size=(3,3), stride=2, padding=1)
        self.layer3  = self._make_layer(block, num_channel[2], num_blocks[2], kernel_size=(1,3), stride=2, padding=(0,1))
        self.layer4  = self._make_layer(block, num_channel[3], num_blocks[3], kernel_size=(1,3), stride=2, padding=(0,1))
        self.avgpool = nn.AvgPool1d(2)
        self.lstm    = nn.LSTM(input_size=64, hidden_size=self.hidden_size, batch_first=True) #lstm
        self.bn2     = nn.BatchNorm1d(num_channel[3]*block.expansion)
        self.linear  = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.relu = nn.ReLU()
       
    def _make_layer(self, block, planes, num_blocks, kernel_size, stride, padding):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride, padding))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        out = self.bn1(self.conv1(x))
        # print(out.shape)
        out = self.layer1(out)
        # print("hi")
        out = self.layer2(out)
        # print("hi")
        out = self.layer3(out)
        # print("hi")
        out = self.layer4(out)
        # print("hi")
        # print(out.shape)
        out = torch.squeeze(out,dim=2)
        # print(out.shape)
        out = self.avgpool(out)
        # print(out.shape)
        # out = torch.squeeze(out,dim=2) 
        # print(out.shape)
        out = self.bn2(out)
        # print(out.shape)
        out = torch.swapaxes(out, 1, 2)
        # print(out.shape)
        out = self.avgpool(out)
        output, (hn, cn) = self.lstm(out)
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        # print(out.shape)
        out = self.linear(out)
        out = self.dropout(out)
        
        return out 

class m_block(nn.Module):
    
    def __init__(self, in_planes, mode=1, stride=1):
        super(m_block, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, 32, kernel_size=(1,1), stride=stride, padding=0, bias=True)
        self.relu1 = nn.ReLU(inplace=False)

        
        self.conv2 = nn.Conv2d(32, 48*mode, kernel_size=(3,1), stride=stride, padding=(1,0), bias=True)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv3 = nn.Conv2d(32, 48*mode, kernel_size=(1,3), stride=stride, padding=(0,1), bias=True)
        self.relu3 = nn.ReLU(inplace=False)

        self.conv4 = nn.Conv2d(32, 32*mode, kernel_size=(1,1), stride=stride, padding=0, bias=True)
        self.relu4 = nn.ReLU(inplace=False)
        

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu1(x)

        out1 = self.conv2(x)
        out1 = self.relu2(out1)

        out2 = self.conv3(x)
        out2 = self.relu3(out2)

        out3 = self.conv4(x)
        out3 = self.relu4(out3)
        # print(out.shape)
        out = torch.cat((out1,out2,out3),dim=1)
        return out

class m_block_p(nn.Module):
    
    def __init__(self, in_planes, stride=1):
        super(m_block_p, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, 32, kernel_size=(1,1), stride=stride, padding=0, bias=True)
        self.relu1 = nn.ReLU(inplace=False)

        
        self.conv2 = nn.Conv2d(32, 48, kernel_size=(3,1), stride=stride, padding=(1,0), bias=True)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,3),stride=(1,2),padding=(0,1))

        self.conv3 = nn.Conv2d(32, 48, kernel_size=(1,3), stride=(1,2), padding=(0,1), bias=True)
        self.relu3 = nn.ReLU(inplace=False)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=(1,1), stride=(1,2), padding=0, bias=True)
        self.relu4 = nn.ReLU(inplace=False)
   
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu1(x)

        out1 = self.conv2(x)
        out1 = self.relu2(out1)
        out1 = self.pool1(out1)

        out2 = self.conv3(x)
        out2 = self.relu3(out2)

        out3 = self.conv4(x)
        out3 = self.relu4(out3)
        # print(out.shape)
        out = torch.cat((out1,out2,out3),dim=1)
        return out        

class m_block_1w1a(nn.Module):
    
    def __init__(self, in_planes, mode=1, stride=1):
        super(m_block_1w1a, self).__init__()

        self.conv1 = BinarizeConv2d(in_planes, 32, kernel_size=(1,1), stride=stride, padding=0, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=False)

        
        self.conv2 = BinarizeConv2d(32, 48*mode, kernel_size=(3,1), stride=stride, padding=(1,0), bias=False)
        self.bn2   = nn.BatchNorm2d(48*mode)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv3 = BinarizeConv2d(32, 48*mode, kernel_size=(1,3), stride=stride, padding=(0,1), bias=False)
        self.bn3   = nn.BatchNorm2d(48*mode)
        self.relu3 = nn.ReLU(inplace=False)

        self.conv4 = BinarizeConv2d(32, 32*mode, kernel_size=(1,1), stride=stride, padding=0, bias=False)
        self.bn4   = nn.BatchNorm2d(32*mode)
        self.relu4 = nn.ReLU(inplace=False)
        

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out1 = self.relu2(out1)

        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out2 = self.relu3(out2)

        out3 = self.conv4(x)
        out3 = self.bn4(out3)
        out3 = self.relu4(out3)
        # print(out.shape)
        out = torch.cat((out1,out2,out3),dim=1)
        return out

class m_block_p_1w1a(nn.Module):
    
    def __init__(self, in_planes, stride=1):
        super(m_block_p_1w1a, self).__init__()

        self.conv1 = BinarizeConv2d(in_planes, 32, kernel_size=(1,1), stride=stride, padding=0, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=False)

        
        self.conv2 = BinarizeConv2d(32, 48, kernel_size=(3,1), stride=stride, padding=(1,0), bias=False)
        self.bn2   = nn.BatchNorm2d(48)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,3),stride=(1,2),padding=(0,1))

        self.conv3 = BinarizeConv2d(32, 48, kernel_size=(1,3), stride=(1,2), padding=(0,1), bias=False)
        self.bn3   = nn.BatchNorm2d(48)
        self.relu3 = nn.ReLU(inplace=False)

        self.conv4 = BinarizeConv2d(32, 32, kernel_size=(1,1), stride=(1,2), padding=0, bias=False)
        self.bn4   = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU(inplace=False)
    
        

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out1 = self.relu2(out1)
        out1 = self.pool1(out1)

        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out2 = self.relu3(out2)

        out3 = self.conv4(x)
        out3 = self.bn4(out3)
        out3 = self.relu4(out3)
        # print(out.shape)
        out = torch.cat((out1,out2,out3),dim=1)
        return out    

class MC_net_6(nn.Module):
    def __init__(self,  num_classes=24):
        super(MC_net_6, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3,7), stride=(1,2), padding=(1,3), bias=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,3),stride=(1,2),padding=(0,1))

        self.pre_conv1 = nn.Conv2d(64, 32, kernel_size=(3,1), stride=(1,1), padding=(1,0), bias=False)
        self.pre_relu1 = nn.ReLU(inplace=False)
        self.pre_pool1 = nn.AvgPool2d(kernel_size=(1,3),stride=(1,2),padding=(0,1))

        self.pre_conv2 = nn.Conv2d(64, 32, kernel_size=(1,3), stride=(1,2), padding=(0,1), bias=False)
        self.pre_relu2 = nn.ReLU(inplace=False)

        self.jump_conv1x1 = nn.Conv2d(64, 128, kernel_size=(1,1), stride=(1,2), bias=False)
        self.jump_relu1 = nn.ReLU(inplace=False)
        self.jump_pool1 = nn.MaxPool2d(kernel_size=(1,3),stride=(1,2),padding=(0,1))

        self.post_pool  = nn.MaxPool2d(kernel_size=(1,3),stride=(1,2),padding=(0,1))
        self.m_block1 = m_block_p(64)

        self.m_block2 = m_block(128,1)

        self.m_block3 = m_block_p(128)
        self.jump_pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=(1,2))

        self.m_block4 = m_block(128,1)

        self.m_block5 = m_block_p(128)
        self.jump_pool5 = nn.MaxPool2d(kernel_size=(2,2),stride=(1,2))

        self.m_block6 = m_block(128,2)

        self.global_pool = nn.AvgPool2d(kernel_size=(2,8))

        self.flat = nn.Flatten()
        self.linear1  = nn.Linear(384, num_classes)
        self.dropout = nn.Dropout(p=0.5, inplace=False)



    def forward(self, x):

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)

        out1 = self.pre_conv1(out)
        out1 = self.pre_relu1(out1)
        out1 = self.pre_pool1(out1)

        out2 = self.pre_conv2(out)
        out2 = self.pre_relu2(out2)

        out = torch.cat((out1,out2),dim=1)

        out1 = self.jump_conv1x1(out)
        out1 = self.jump_relu1(out1)
        out1 = self.jump_pool1(out1)

        out2 = self.post_pool(out)
        out2 = self.m_block1(out2)

        out = out1 + out2

        out = self.m_block2(out) + out

        out = self.m_block3(out) + self.jump_pool3(F.pad(out,(1,0,0,0))) 
        out = self.m_block4(out) + out 
        out = self.m_block5(out) + self.jump_pool5(F.pad(out,(1,0,0,0)))

        out1 = self.m_block6(out)

        out = torch.cat((out,out1),dim=1)
        out = self.global_pool(out)
        out = self.flat(out)

        out = self.linear1(out) 
        out = self.dropout(out)

        return out 

class MC_net_10(nn.Module):
    def __init__(self,  num_classes=24):
        super(MC_net_10, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3,7), stride=(1,2), padding=(1,3), bias=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,3),stride=(1,2),padding=(0,1))

        self.pre_conv1 = nn.Conv2d(64, 32, kernel_size=(3,1), stride=(1,1), padding=(1,0), bias=False)
        self.pre_relu1 = nn.ReLU(inplace=False)
        self.pre_pool1 = nn.AvgPool2d(kernel_size=(1,3),stride=(1,2),padding=(0,1))

        self.pre_conv2 = nn.Conv2d(64, 32, kernel_size=(1,3), stride=(1,2), padding=(0,1), bias=False)
        self.pre_relu2 = nn.ReLU(inplace=False)

        self.jump_conv1x1 = nn.Conv2d(64, 128, kernel_size=(1,1), stride=(1,2), bias=False)
        self.jump_relu1 = nn.ReLU(inplace=False)
        self.jump_pool1 = nn.MaxPool2d(kernel_size=(1,3),stride=(1,2),padding=(0,1))

        self.post_pool  = nn.MaxPool2d(kernel_size=(1,3),stride=(1,2),padding=(0,1))
        self.m_block1 = m_block_p(64)

        self.m_block2 = m_block(128,1)

        self.m_block3 = m_block_p(128)
        self.jump_pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=(1,2))

        self.m_block4 = m_block(128,1)

        self.m_block5 = m_block_p(128)
        self.jump_pool5 = nn.MaxPool2d(kernel_size=(2,2),stride=(1,2))

        self.m_block6 = m_block(128,1)

        self.m_block7 = m_block_p(128)
        self.jump_pool7 = nn.MaxPool2d(kernel_size=(2,2),stride=(1,2))

        self.m_block8 = m_block(128,1)

        self.m_block9 = m_block_p(128)
        self.jump_pool9 = nn.MaxPool2d(kernel_size=(2,2),stride=(1,2))

        self.m_block10 = m_block(128,2)


        self.global_pool = nn.AvgPool2d(kernel_size=(2,2))

        self.flat = nn.Flatten()
        self.linear1  = nn.Linear(384, num_classes)
        self.dropout = nn.Dropout(p=0.5, inplace=False)



    def forward(self, x):

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)

        out1 = self.pre_conv1(out)
        out1 = self.pre_relu1(out1)
        out1 = self.pre_pool1(out1)

        out2 = self.pre_conv2(out)
        out2 = self.pre_relu2(out2)

        out = torch.cat((out1,out2),dim=1)

        out1 = self.jump_conv1x1(out)
        out1 = self.jump_relu1(out1)
        out1 = self.jump_pool1(out1)

        out2 = self.post_pool(out)
        out2 = self.m_block1(out2)

        out = out1 + out2

        out = self.m_block2(out) + out
        out = self.m_block3(out) + self.jump_pool3(F.pad(out,(1,0,0,0))) 

        out = self.m_block4(out) + out 
        out = self.m_block5(out) + self.jump_pool5(F.pad(out,(1,0,0,0)))

        out = self.m_block6(out) + out 
        out = self.m_block7(out) + self.jump_pool7(F.pad(out,(1,0,0,0)))

        out = self.m_block8(out) + out 
        out = self.m_block9(out) + self.jump_pool9(F.pad(out,(1,0,0,0)))

        out1 = self.m_block10(out)

        out = torch.cat((out,out1),dim=1)
        out = self.global_pool(out)
        out = self.flat(out)
        
        out = self.linear1(out) 
        out = self.dropout(out)

        return out

class MC_net_10_binary(nn.Module):
    def __init__(self,  num_classes=24):
        super(MC_net_10_binary, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3,7), stride=(1,2), padding=(1,3), bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,3),stride=(1,2),padding=(0,1))

        self.pre_conv1 = BinarizeConv2d(64, 32, kernel_size=(3,1), stride=(1,1), padding=(1,0), bias=False)
        self.bn2   = nn.BatchNorm2d(32)
        self.pre_relu1 = nn.ReLU(inplace=False)
        self.pre_pool1 = nn.AvgPool2d(kernel_size=(1,3),stride=(1,2),padding=(0,1))

        self.pre_conv2 = BinarizeConv2d(64, 32, kernel_size=(1,3), stride=(1,2), padding=(0,1), bias=False)
        self.bn3   = nn.BatchNorm2d(32)
        self.pre_relu2 = nn.ReLU(inplace=False)

        self.jump_conv1x1 = BinarizeConv2d(64, 128, kernel_size=(1,1), stride=(1,2), bias=False)
        self.jump_bn   = nn.BatchNorm2d(128)
        self.jump_relu1 = nn.ReLU(inplace=False)
        self.jump_pool1 = nn.MaxPool2d(kernel_size=(1,3),stride=(1,2),padding=(0,1))

        self.post_pool  = nn.MaxPool2d(kernel_size=(1,3),stride=(1,2),padding=(0,1))
        self.m_block1 = m_block_p_1w1a(64)

        self.m_block2 = m_block_1w1a(128,1)

        self.m_block3 = m_block_p_1w1a(128)
        self.jump_pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=(1,2))

        self.m_block4 = m_block_1w1a(128,1)

        self.m_block5 = m_block_p_1w1a(128)
        self.jump_pool5 = nn.MaxPool2d(kernel_size=(2,2),stride=(1,2))

        self.m_block6 = m_block_1w1a(128,1)

        self.m_block7 = m_block_p_1w1a(128)
        self.jump_pool7 = nn.MaxPool2d(kernel_size=(2,2),stride=(1,2))

        self.m_block8 = m_block_1w1a(128,1)

        self.m_block9 = m_block_p_1w1a(128)
        self.jump_pool9 = nn.MaxPool2d(kernel_size=(2,2),stride=(1,2))

        self.m_block10 = m_block_1w1a(128,2)


        self.global_pool = nn.AvgPool2d(kernel_size=(2,2))

        self.flat = nn.Flatten()
        self.linear1  = nn.Linear(384, num_classes)
        self.dropout = nn.Dropout(p=0.5, inplace=False)



    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out1 = self.pre_conv1(out)
        out1 = self.bn2(out1)
        out1 = self.pre_relu1(out1)
        out1 = self.pre_pool1(out1)

        out2 = self.pre_conv2(out)
        out2 = self.bn3(out2)
        out2 = self.pre_relu2(out2)

        out = torch.cat((out1,out2),dim=1)

        out1 = self.jump_conv1x1(out)
        out1 = self.jump_bn(out1)
        out1 = self.jump_relu1(out1)
        out1 = self.jump_pool1(out1)

        out2 = self.post_pool(out)
        out2 = self.m_block1(out2)

        out = out1 + out2

        out = self.m_block2(out) + out
        out = self.m_block3(out) + self.jump_pool3(F.pad(out,(1,0,0,0))) 

        out = self.m_block4(out) + out 
        out = self.m_block5(out) + self.jump_pool5(F.pad(out,(1,0,0,0)))

        out = self.m_block6(out) + out 
        out = self.m_block7(out) + self.jump_pool7(F.pad(out,(1,0,0,0)))

        out = self.m_block8(out) + out 
        out = self.m_block9(out) + self.jump_pool9(F.pad(out,(1,0,0,0)))

        out1 = self.m_block10(out)

        out = torch.cat((out,out1),dim=1)
        out = self.global_pool(out)
        out = self.flat(out)
        
        out = self.linear1(out) 
        out = self.dropout(out)

        return out

class Lightweight(nn.Module):
    def __init__(self,   num_classes=24):
        super(Lightweight, self).__init__()

        self.conv1    = nn.Conv2d(1, 16, kernel_size=(3,3), stride=1, padding=(3,1), bias=False)
        self.bn1      = nn.BatchNorm2d(16)
        self.relu1    = nn.ReLU(inplace=False)
        self.avgpool1 = nn.AvgPool2d(kernel_size= (3,2), stride=(3,2), padding=0)

        self.conv2    = nn.Conv2d(16, 32, kernel_size=(3,3), stride=1, padding=(3,1), bias=False)
        self.bn2      = nn.BatchNorm2d(32)
        self.relu2    = nn.ReLU(inplace=False)
        self.avgpool2 = nn.AvgPool2d(kernel_size= (2,2), stride=(2,2), padding=0)

        self.block1_c1  = nn.Conv2d(32, 32, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn_b1c1    = nn.BatchNorm2d(32)
        self.relu_b1c1  = nn.ReLU(inplace=False)

        self.block1_c2  = nn.Conv2d(32, 8, kernel_size=(3,1), stride=1, padding=(1,0), bias=False)
        self.bn_b1c2    = nn.BatchNorm2d(8)
        self.relu_b1c2  = nn.ReLU(inplace=False)

        self.block1_c3  = nn.Conv2d(8, 8, kernel_size=(1,3), stride=1, padding=(0,1), bias=False)
        self.bn_b1c3    = nn.BatchNorm2d(8)
        self.relu_b1c3  = nn.ReLU(inplace=False)

        self.block1_c4  = nn.Conv2d(8, 32, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn_b1c4    = nn.BatchNorm2d(32)
        self.relu_b1c4  = nn.ReLU(inplace=False)

        self.block2_c1  = nn.Conv2d(32, 64, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn_b2c1    = nn.BatchNorm2d(64)
        self.relu_b2c1  = nn.ReLU(inplace=False)

        self.block2_c2  = nn.Conv2d(64, 16, kernel_size=(3,1), stride=1, padding=(1,0), bias=False)
        self.bn_b2c2    = nn.BatchNorm2d(16)
        self.relu_b2c2  = nn.ReLU(inplace=False)

        self.block2_c3  = nn.Conv2d(16, 16, kernel_size=(1,3), stride=1, padding=(0,1), bias=False)
        self.bn_b2c3    = nn.BatchNorm2d(16)
        self.relu_b2c3  = nn.ReLU(inplace=False)

        self.block2_c4  = nn.Conv2d(16, 64, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn_b2c4    = nn.BatchNorm2d(64)
        self.relu_b2c4  = nn.ReLU(inplace=False)

        self.block3_c1  = nn.Conv2d(64, 128, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn_b3c1    = nn.BatchNorm2d(128)
        self.relu_b3c1  = nn.ReLU(inplace=False)

        self.block3_c2  = nn.Conv2d(128, 32, kernel_size=(3,1), stride=1, padding=(1,0), bias=False)
        self.bn_b3c2    = nn.BatchNorm2d(32)
        self.relu_b3c2  = nn.ReLU(inplace=False)

        self.block3_c3  = nn.Conv2d(32, 32, kernel_size=(1,3), stride=1, padding=(0,1), bias=False)
        self.bn_b3c3    = nn.BatchNorm2d(32)
        self.relu_b3c3  = nn.ReLU(inplace=False)

        self.block3_c4  = nn.Conv2d(32, 128, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn_b3c4    = nn.BatchNorm2d(128)
        self.relu_b3c4  = nn.ReLU(inplace=False)

        self.avgpool3 = nn.AvgPool2d(kernel_size= (3,256), stride=1, padding=0)
        self.flat = nn.Flatten()
        self.linear1  = nn.Linear(128, num_classes)

    def forward(self, x):
        
        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu1(out)
        # print(out.shape)
        out = self.avgpool1(out)

        # print(out.shape)
        out = self.conv2(out)
        # print(out.shape)
        out = self.bn2(out)
        out = self.relu2(out)
        

        out = self.avgpool2(out)
        # print(out.shape)

        out = self.block1_c1(out)
        # print(out.shape)
        out = self.bn_b1c1(out)
        out = self.relu_b1c1(out)

        out1 = self.block1_c2(out)
        out1 = self.bn_b1c2(out1)
        out1 = self.relu_b1c2(out1)
        # print(out1.shape)

        out1 = self.block1_c3(out1)
        out1 = self.bn_b1c3(out1)
        out1 = self.relu_b1c3(out1)

        # print(out1.shape)
        out1 = self.block1_c4(out1)
        out1 = self.bn_b1c4(out1)
        out1 = self.relu_b1c4(out1)


        out = self.block2_c1(out + out1)
        out = self.bn_b2c1(out)
        out = self.relu_b2c1(out)

        out1 = self.block2_c2(out)
        out1 = self.bn_b2c2(out1)
        out1 = self.relu_b2c2(out1)

        out1 = self.block2_c3(out1)
        out1 = self.bn_b2c3(out1)
        out1 = self.relu_b2c3(out1)

        out1 = self.block2_c4(out1)
        out1 = self.bn_b2c4(out1)
        out1 = self.relu_b2c4(out1)


        out = self.block3_c1(out + out1)
        out = self.bn_b3c1(out)
        out = self.relu_b3c1(out)

        out1 = self.block3_c2(out)
        out1 = self.bn_b3c2(out1)
        out1 = self.relu_b3c2(out1)

        out1 = self.block3_c3(out1)
        out1 = self.bn_b3c3(out1)
        out1 = self.relu_b3c3(out1)

        out1 = self.block3_c4(out1)
        out1 = self.bn_b3c4(out1)
        out1 = self.relu_b3c4(out1)


        out = self.avgpool3(out + out1)
        out = self.flat(out)
        out = self.linear1(out)

        return out 



def MC_net_10blocks_binary(**kwargs):
    return MC_net_10_binary(24)

def MC_net_10blocksv2(**kwargs):
    return MC_net_10(24)

def MC_net_6blocks(**kwargs):
    return MC_net_6(24)

def rml_resnet_1w1av2(**kwargs):
    return ResNet_rml_binary(BasicBlock_ppr_1w1a, [2,2,2,2],[32,32,32,32],**kwargs)

def rml_resnet(**kwargs):
    return ResNet_rml(BasicBlock_paper, [2,2,2,2],[32,32,32,32],**kwargs)

def resnet18Av2(**kwargs):
    return ResNet(BasicBlock_real, [2,2,2,2],[32,32,64,128],**kwargs)

def resnet18A_LSTM(**kwargs):
    return ResNet_LSTM(BasicBlock_real, [2,2,2,2],[32,32,64,128],**kwargs)

def resnet18A_compare(**kwargs):
    return ResNet_compare(BasicBlock_compare, [2,2,2,2],[32,32,64,128],**kwargs)

def resnet18A_1w1a(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2],[32,32,64,128],**kwargs)

def resnet18A_1w1a_LSTM(**kwargs):
    return ResNet_LSTM(BasicBlock, [2,2,2,2],[32,32,64,128],**kwargs)

def resnet18A_1w1a_U(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2],[16,16,32,64],**kwargs)

# def resnet18A_1w1a(**kwargs):
#     return ResNet(BasicBlock, [3,4,6,3],[32,32,64,128],**kwargs)

def resnet18A_1w1a_K(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2],[64,64,128,256],**kwargs)

def resnet18B_1w1a(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2],[32,64,128,256],**kwargs)

def resnet18C_1w1a(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2],[64,64,128,256],**kwargs)

def resnet18_1w1a(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2],[64,128,256,512],**kwargs)

def light(**kwargs):
    return Lightweight(**kwargs)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = resnet18_1w1a()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()

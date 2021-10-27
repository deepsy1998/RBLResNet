import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
from .binarized_modules import  BinarizeLinear,BinarizeConv2d
import torch.nn.functional as F

__all__ = ['resnet_binary']

def Binaryconv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def init_model(model):
    for m in model.modules():
        if isinstance(m, BinarizeConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,do_bntan=True):
        super(BasicBlock, self).__init__()

        self.conv1 = Binaryconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.conv2 = Binaryconv3x3(planes, planes)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.do_bntan=do_bntan;
        self.stride = stride

    def forward(self, x):

        residual = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh1(out)

        out = self.conv2(out)


        if self.downsample is not None:
            if residual.data.max()>1:
                import pdb; pdb.set_trace()
            residual = self.downsample(residual)

        out += residual
        if self.do_bntan:
            out = self.bn2(out)
            out = self.tanh2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = BinarizeConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = BinarizeConv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.tanh = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        import pdb; pdb.set_trace()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.tanh(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.do_bntan:
            out = self.bn2(out)
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
        # print("nasama pochu")
        # print(out.shape)
        out += self.shortcut(x)
        out = self.tanh2(out)
        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1,do_bntan=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                BinarizeConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes,do_bntan=do_bntan))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)
        x = torch.squeeze(x)
        x = self.avgpool(x) 
        # print(x.shape)
        x = torch.squeeze(x) 
        # print(x.shape)
        x = self.bn2(x)
        x = self.fc(x)
        return x

class ResNet_ppr(nn.Module):

    def __init__(self):
        super(ResNet_ppr, self).__init__()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.size(1))
        out = self.conv1(x)
        # print(out.size(1))
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
        out = self.alpha1(out)

        # print(out.shape)
        out = self.bn8(out)
        out = self.linear2(out)
        # print(out.shape)
        out = self.act2(out)
        out = self.alpha2(out)
        
        # print(out.shape)
        out = self.bn9(out)
        out = self.linear3(out)
        # print(out.shape)


        return out

class ResNet_rml_ppr(ResNet_ppr):

    def __init__(self, num_classes=11, block=BasicBlock_ppr_1w1a, depth=18):
        super(ResNet_rml_ppr, self).__init__()
        self.in_planes = 32

        self.conv1   = BinarizeConv2d(1, 32, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1  = self._make_layer(block, 32, 2, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size= (2,2), stride=(1,2), padding=0)

        self.conv2   = BinarizeConv2d(32, 32, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.layer2  = self._make_layer(block, 32, 2, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size= (1,2), stride=(1,2), padding=0)

        self.conv3   = BinarizeConv2d(32, 32, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.layer3  = self._make_layer(block, 32, 2, stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size= (1,2), stride=(1,2), padding=0)

        self.conv4   = BinarizeConv2d(32, 32, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.layer4  = self._make_layer(block, 32, 2, stride=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size= (1,2), stride=(1,2), padding=0)

        self.conv5   = BinarizeConv2d(32, 32, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.layer5  = self._make_layer(block, 32, 2, stride=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size= (1,2), stride=(1,2), padding=0)

        self.conv6   = BinarizeConv2d(32, 32, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.layer6  = self._make_layer(block, 32, 2, stride=1)
        self.maxpool6 = nn.MaxPool2d(kernel_size= (1,2), stride=(1,2), padding=0)

        self.flat = nn.Flatten()
        self.bn7 = nn.BatchNorm1d(512)
        self.linear1  = nn.Linear(512, 128)
        self.act1 = nn.SELU(inplace=False)
        self.alpha1 = nn.AlphaDropout(p=0.3, inplace=False)

        self.bn8 = nn.BatchNorm1d(128)
        self.linear2  = nn.Linear(128, 128)
        self.act2 = nn.SELU(inplace=False)
        self.alpha2 = nn.AlphaDropout(p=0.3, inplace=False)

        self.bn9 = nn.BatchNorm1d(128)
        self.linear3  = nn.Linear(128, num_classes)

        init_model(self)
        # self.regime = {
        #   0: {'optimizer': 'SGD', 'lr': 1e-1,
        #       'weight_decay': 1e-4, 'momentum': 0.9},
        #   81: {'lr': 1e-4},
        #   122: {'lr': 1e-5, 'weight_decay': 0},
        #   164: {'lr': 1e-6}
        # }
        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-2},
            21: {'lr': 1e-3},
            41: {'lr': 5e-4},
            61: {'lr': 1e-4},
            81: {'lr': 1e-5}
        }

class ResNet_rml(ResNet):

    def __init__(self, num_classes=24, block=BasicBlock, depth=18):
        super(ResNet_rml, self).__init__()
        self.inflate = 5
        self.inplanes = 32
        n = 2
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.layer1 = self._make_layer(block, 32, n, stride=1)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.layer4 = self._make_layer(block, 128, n, stride=2)
        self.avgpool = nn.AvgPool1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, num_classes)

        init_model(self)
        # self.regime = {
        #   0: {'optimizer': 'SGD', 'lr': 1e-1,
        #       'weight_decay': 1e-4, 'momentum': 0.9},
        #   81: {'lr': 1e-4},
        #   122: {'lr': 1e-5, 'weight_decay': 0},
        #   164: {'lr': 1e-6}
        # }
        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 5e-2},
            101: {'lr': 1e-3},
            142: {'lr': 5e-4},
            184: {'lr': 1e-4},
            220: {'lr': 1e-5}
        }

class ResNet_imagenet(ResNet):

    def __init__(self, num_classes=1000,
                 block=Bottleneck, layers=[3, 4, 23, 3]):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = 64
        self.conv1 = BinarizeConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.tanh = nn.Hardtanh(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = BinarizeLinear(512 * block.expansion, num_classes)

        init_model(self)
        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-1,
                'weight_decay': 1e-4, 'momentum': 0.9},
            30: {'lr': 1e-2},
            60: {'lr': 1e-3, 'weight_decay': 0},
            90: {'lr': 1e-4}
        }


class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=10, block=BasicBlock, depth=18):
        super(ResNet_cifar10, self).__init__()
        self.inflate = 5
        self.inplanes = 16*self.inflate
        n = int((depth - 2) / 6)
        self.conv1 = BinarizeConv2d(3, 16*self.inflate, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.maxpool = lambda x: x
        self.bn1 = nn.BatchNorm2d(16*self.inflate)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.layer1 = self._make_layer(block, 16*self.inflate, n)
        self.layer2 = self._make_layer(block, 32*self.inflate, n, stride=2)
        self.layer3 = self._make_layer(block, 64*self.inflate, n, stride=2,do_bntan=False)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.bn2 = nn.BatchNorm1d(64*self.inflate)
        self.bn3 = nn.BatchNorm1d(10)
        self.logsoftmax = nn.LogSoftmax()
        self.fc = BinarizeLinear(64*self.inflate, num_classes)

        init_model(self)
        #self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-1,
        #        'weight_decay': 1e-4, 'momentum': 0.9},
        #    81: {'lr': 1e-4},
        #    122: {'lr': 1e-5, 'weight_decay': 0},
        #    164: {'lr': 1e-6}
        #}
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            101: {'lr': 1e-3},
            142: {'lr': 5e-4},
            184: {'lr': 1e-4},
            220: {'lr': 1e-5}
        }


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

class MC_net_10_forward(nn.Module):

    def __init__(self):
        super(MC_net_10_forward, self).__init__()


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

class MC_net_10(MC_net_10_forward):

    def __init__(self, num_classes=24):
        super(MC_net_10, self).__init__()
        self.conv1 = BinarizeConv2d(1, 64, kernel_size=(3,7), stride=(1,2), padding=(1,3), bias=False)
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

        init_model(self)
        # self.regime = {
        #   0: {'optimizer': 'SGD', 'lr': 1e-1,
        #       'weight_decay': 1e-4, 'momentum': 0.9},
        #   81: {'lr': 1e-4},
        #   122: {'lr': 1e-5, 'weight_decay': 0},
        #   164: {'lr': 1e-6}
        # }
        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 5e-3},
            101: {'lr': 1e-3},
            142: {'lr': 5e-4},
            184: {'lr': 1e-4},
            220: {'lr': 1e-5}
        }

def resnet_binary(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        depth = depth or 50
        if depth == 18:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers = [2, 2, 2, 2])
        if depth == 34:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers = [3, 4, 6, 3])
        if depth == 50:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers = [3, 4, 6, 3])
        if depth == 101:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers = [3, 4, 23, 3])
        if depth == 152:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers = [3, 8, 36, 3])

    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        depth = depth or 18
        return ResNet_cifar10(num_classes=num_classes,
                              block=BasicBlock, depth=depth)

    elif dataset == 'rml':
        num_classes = num_classes or 11
        depth = depth or 18
        # return ResNet_rml_ppr(num_classes=24, block=BasicBlock_ppr_1w1a, depth=depth)
        # return MC_net_10(num_classes=24)
        return ResNet_rml(num_classes=24, block=BasicBlock, depth=depth)
                              

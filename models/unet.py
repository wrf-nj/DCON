import torch
import torch.nn as nn
import torch.nn.functional as F

def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(1, planes)
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    else:
        raise ValueError('Normalization type {} is not supporter'.format(norm))
    return m


#### Note: All are functional units except the norms, which are sequential
class ConvD(nn.Module):
    def __init__(self, inplanes, planes, norm='bn', first=False, activation='relu'):
        super(ConvD, self).__init__()

        self.first = first
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=True)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm)
        self.maxpool2D = nn.MaxPool2d(kernel_size=2)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):


        if not self.first:
            x = self.maxpool2D(x)

        #layer 1 conv, bn
        x = self.conv1(x)
        x = self.bn1(x)

        #layer 2 conv, bn, relu
        y = self.conv2(x)
        y = self.bn2(y)
        y = self.activation(y)

        #layer 3 conv, bn
        z = self.conv3(y)
        z = self.bn3(z)
        z = self.activation(z)

        return z


class ConvU(nn.Module):
    def __init__(self, planes, norm='bn', first=False, activation='relu'):
        super(ConvU, self).__init__()

        self.first = first
        if not self.first:
            self.conv1 = nn.Conv2d(2*planes, planes, 3, 1, 1, bias=True)
            self.bn1   = normalization(planes, norm)

        self.pool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(planes, planes//2, 1, 1, 0, bias=True)
        self.bn2   = normalization(planes//2, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x, prev):
        #layer 1 conv, bn, relu
        if not self.first:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.activation(x)

        #upsample, layer 2 conv, bn, relu
        y = self.pool(x)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.activation(y)

        #concatenation of two layers
        y = torch.cat([prev, y], 1)

        #layer 3 conv, bn
        y = self.conv3(y)
        y = self.bn3(y)
        y = self.activation(y)

        return y

class Unet1(nn.Module):
    def __init__(self, c=3, n=16, norm='bn', num_classes=2, activation='relu'):
        super(Unet1, self).__init__()

        self.convd1 = ConvD(c,     n, norm, first=True, activation=activation)
        self.convd2 = ConvD(n,   2*n, norm, activation=activation)
        self.convd3 = ConvD(2*n, 4*n, norm, activation=activation)
        self.convd4 = ConvD(4*n, 8*n, norm, activation=activation)
        self.convd5 = ConvD(8*n,16*n, norm, activation=activation)
        
        self.convu4 = ConvU(16*n, norm, first=True, activation=activation)
        self.convu3 = ConvU(8*n, norm, activation=activation)
        self.convu2 = ConvU(4*n, norm, activation=activation)
        self.convu1 = ConvU(2*n, norm, activation=activation)

        self.seg1 = nn.Conv2d(2*n, num_classes, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x ,dropout_rate,fdropout):
        self.dropout_rate=dropout_rate
        self.f_dropout=fdropout
        # print("dropout_rate:", dropout_rate)
        # print("f_dropout:",fdropout)

        x1 = self.convd1(x)
        if fdropout==1:
            x1 = nn.Dropout2d(p=self.dropout_rate)(x1)
        x2 = self.convd2(x1)
        if fdropout==1:
            x2 = nn.Dropout2d(p=self.dropout_rate)(x2)
        x3 = self.convd3(x2)
        if fdropout==1:
            x3 = nn.Dropout2d(p=self.dropout_rate)(x3)
        x4 = self.convd4(x3)
        if fdropout==1:
            x4 = nn.Dropout2d(p=self.dropout_rate)(x4)
        x5 = self.convd5(x4)

        y4 = self.convu4(x5, x4)
        y3 = self.convu3(y4, x3)
        y2 = self.convu2(y3, x2)
        y1 = self.convu1(y2, x1)
        y1_pred = self.seg1(y1)

        return y1_pred,x5

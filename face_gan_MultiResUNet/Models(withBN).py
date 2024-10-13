import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvBlock2d(nn.Module):
    def __init__(self, in_feature, out_feature, kernel_size, stride=(1, 1), activation='relu'):
        super(ConvBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=kernel_size, stride=stride, padding='same')
        self.batchNorm = nn.BatchNorm2d(out_feature)
        self.activation = activation
        
    def forward(self, x):
        x = self.batchNorm(self.conv(x))
        if self.activation == 'relu':
            return F.relu(x)
        else:
            return x


class MultiResBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(MultiResBlock, self).__init__()
        feature_3x3 = out_feature // 6
        feature_5x5 = out_feature // 3
        feature_7x7 = out_feature - feature_3x3 - feature_5x5
        self.conv_3x3 = ConvBlock2d(in_feature, feature_3x3, kernel_size=3)
        self.conv_5x5 = ConvBlock2d(feature_3x3, feature_5x5, kernel_size=3)
        self.conv_7x7 = ConvBlock2d(feature_5x5, feature_7x7, kernel_size=3)

        self.conv_1x1 = ConvBlock2d(in_feature, out_feature, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm2d(out_feature)
        self.batch_norm2 = nn.BatchNorm2d(out_feature)

    def forward(self, x):
        o_3x3 = self.conv_3x3(x)
        o_5x5 = self.conv_5x5(o_3x3)
        o_7x7 = self.conv_7x7(o_5x5)
        o = self.batch_norm1(torch.cat([o_3x3, o_5x5, o_7x7], axis=1))

        o_1x1 = self.conv_1x1(x)

        o = self.batch_norm1(o + o_1x1)

        return F.relu(o)
    
class ResPath(nn.Module):
    def __init__(self, in_feature, out_feature, length):
        super(ResPath, self).__init__()
        self.respath_length = length
        self.residuals = torch.nn.ModuleList([])
        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])

        for i in range(self.respath_length):
            if(i==0):
                self.residuals.append(ConvBlock2d(in_feature, out_feature, kernel_size = (1,1), activation='None'))
                self.convs.append(ConvBlock2d(in_feature, out_feature, kernel_size = (3,3),activation='relu'))

            	
            else:
                self.residuals.append(ConvBlock2d(out_feature, out_feature, kernel_size = (1,1), activation='None'))
                self.convs.append(ConvBlock2d(out_feature, out_feature, kernel_size = (3,3), activation='relu'))

            self.bns.append(torch.nn.BatchNorm2d(out_feature))

    def forward(self, x):
        
        for i in range(self.respath_length):
            res = self.residuals[i](x)

            x = self.convs[i](x)
            # x = self.bns[i](x)
            # x = torch.nn.functional.relu(x)

            x = x + res
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)
        
        return x



class MultiResUNet(nn.Module):
    def __init__(self, in_feature, out_feature, alpha=1.667, ngf = 32):
        super(MultiResUNet, self).__init__()
        #encoder
        feature1 = int(ngf * alpha)
        self.multi1 = MultiResBlock(in_feature, feature1)
        self.pool1 = nn.MaxPool2d(2)
        self.respath1 = ResPath(feature1, ngf, length=4)

        feature2 = int(ngf * 2 * alpha)
        self.multi2 = MultiResBlock(feature1, feature2)
        self.pool2 = nn.MaxPool2d(2)
        self.respath2 = ResPath(feature2, ngf * 2, length=3)

        feature3 = int(ngf * 4 * alpha)
        self.multi3 = MultiResBlock(feature2, feature3)
        self.pool3 = nn.MaxPool2d(2)
        self.respath3 = ResPath(feature3, ngf * 4, length=2)

        feature4 = int(ngf * 8 * alpha)
        self.multi4 = MultiResBlock(feature3, feature4)
        self.pool4 = nn.MaxPool2d(2)
        self.respath4 = ResPath(feature4, ngf * 8, length=1)

        feature5 = int(ngf * 16 * alpha)
        self.multi5 = MultiResBlock(feature4, feature5)

        #decoder
        out_feature5 = feature5
        self.upsample1 = nn.ConvTranspose2d(out_feature5, ngf * 8, kernel_size = (2, 2), stride = (2, 2))  
        out_feature4 = int(ngf * 8 * alpha)
        self.multi6 = MultiResBlock(ngf * 8 * 2, out_feature4)
        
        self.upsample2 = nn.ConvTranspose2d(out_feature4, ngf * 4, kernel_size = (2, 2), stride = (2, 2))
        out_feature3 = int(ngf * 4 * alpha)  
        self.multi7 = MultiResBlock(ngf * 4 * 2, out_feature3)
	
        self.upsample3 = nn.ConvTranspose2d(out_feature3, ngf * 2, kernel_size = (2, 2), stride = (2, 2))
        out_feature2 = int(ngf * 2 * alpha)
        self.multi8 = MultiResBlock(ngf * 2 * 2, out_feature2)
		
        self.upsample4 = nn.ConvTranspose2d(out_feature2, ngf, kernel_size = (2, 2), stride = (2, 2))
        out_feature1 = int(ngf * alpha)
        self.multi9 = MultiResBlock(ngf * 2, out_feature1)

        self.conv_final = ConvBlock2d(out_feature1, out_feature, kernel_size = (1,1), activation='None')

    def forward(self, x):
        #encoder
        layer1 = self.multi1(x)        
        layer2 = self.multi2(self.pool1(layer1))
        layer3 = self.multi3(self.pool2(layer2))
        layer4 = self.multi4(self.pool3(layer3))
        layer5 = self.multi5(self.pool4(layer4))
        #decoder
        layer4 = self.multi6(torch.cat([self.upsample1(layer5), self.respath4(layer4)], axis=1))
        layer3 = self.multi7(torch.cat([self.upsample2(layer4), self.respath3(layer3)], axis=1))
        layer2 = self.multi8(torch.cat([self.upsample3(layer3), self.respath2(layer2)], axis=1))
        layer1 = self.multi9(torch.cat([self.upsample4(layer2), self.respath1(layer1)], axis=1))

        return self.conv_final(layer1)




class Discriminator(nn.Module):
    def __init__(self, in_feature, alpha=0.667, ndf = 32):
        super(Discriminator, self).__init__()
        feature1 = int(ndf * alpha)
        self.multi1 = MultiResBlock(in_feature, feature1)
        self.pool1 = nn.MaxPool2d(2)

        feature2 = int(ndf * 2 * alpha)
        self.multi2 = MultiResBlock(feature1, feature2)
        self.pool2 = nn.MaxPool2d(2)

        feature3 = int(ndf * 4 * alpha)
        self.multi3 = MultiResBlock(feature2, feature3)
        self.pool3 = nn.MaxPool2d(2)

        feature4 = int(ndf * 8 * alpha)
        self.multi4 = MultiResBlock(feature3, feature4)
        self.pool4 = nn.MaxPool2d(2)

        feature5 = int(ndf * 16 * alpha)
        self.multi5 = MultiResBlock(feature4, feature5)
        self.pool5 = nn.MaxPool2d(2)

        self.FC = nn.Linear(feature5 * 8 * 8, 1)

    def forward(self, x):
        #encoder
        layer = self.multi1(x)        
        layer = self.multi2(self.pool1(layer))
        layer = self.multi3(self.pool2(layer))
        layer = self.multi4(self.pool3(layer))
        layer = self.multi5(self.pool4(layer))
        layer = self.FC(self.pool5(layer).view(x.shape[0], -1))
        return F.sigmoid(layer)


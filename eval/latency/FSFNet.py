# FSFNet full model definition for Pytorch
# August 2020
# MINJONG KIM
#######################


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Stem(nn.Module):
    def __init__(self):
        super().__init__()


        self.conv_1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.bn_1 = nn.BatchNorm2d(32, eps=1e-3)

        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.bn_2 = nn.BatchNorm2d(64, eps=1e-3)

        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.bn_3 = nn.BatchNorm2d(64, eps=1e-3)

        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.bn_4 = nn.BatchNorm2d(64, eps=1e-3)

        self.conv_5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.bn_5 = nn.BatchNorm2d(64, eps=1e-3)

    def forward(self, input):

        output = self.conv_1(input)
        output = self.bn_1(output)
        output = F.relu(output)

        output = self.conv_2(output)
        output = self.bn_2(output)
        output = F.relu(output)

        output = self.conv_3(output)
        output = self.bn_3(output)
        output = F.relu(output)

        output = self.conv_4(output)
        output = self.bn_4(output)
        output = F.relu(output)

        output = self.conv_5(output)
        output = self.bn_5(output)
        output = F.relu(output)

        return output

class FSF_8(nn.Module):
    def __init__(self, input, output, dropprob, dilated, kernel_size):
        super().__init__()

        self.dropout = nn.Dropout2d(dropprob)
        self.conv3x3_1 = nn.Conv2d(input, output, (kernel_size, kernel_size), stride=1, bias=True,padding=(1, 1))
        self.conv3x3_2 = nn.Conv2d(output, output, (kernel_size, kernel_size), stride=1, bias=True, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(output, eps=1e-03)
        self.bn2 = nn.BatchNorm2d(output, eps=1e-03)

    def forward(self, input):

        output = F.interpolate(input, size=(int(input.size(2)) // 2, int(input.size(3)) // 2),  mode='nearest')
        output = self.conv3x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.conv3x3_2(output)
        output = self.bn2(output)
        output = F.interpolate(output, size=(int(input.size(2) ), int(input.size(3) )), mode='nearest')

        if (self.dropout.p != 0):
            output = self.dropout(output)


        return F.relu(output + input)  #



class FSF_16(nn.Module):
    def __init__(self, input, output, dropprob, dilated, kernel_size):
        super().__init__()

        self.dropout = nn.Dropout2d(dropprob)
        self.conv3x1_0 = nn.Conv2d(input, output, (kernel_size, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(output, output, (1, kernel_size), stride=1, padding=(0, 1), bias=True)
        self.conv3x3_2 = nn.Conv2d(output, output, (kernel_size, kernel_size), stride=1, padding=(dilated, dilated), bias=True, dilation=(dilated, dilated), groups=output)
        self.bn1 = nn.BatchNorm2d(output, eps=1e-03)
        self.bn2 = nn.BatchNorm2d(output, eps=1e-03)

    def forward(self, input):

        output = F.interpolate(input, size=(int(input.size(2)) // 2, int(input.size(3)) // 2), mode='nearest')
        output = self.conv3x1_0(output)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.conv3x3_2(output)
        output = self.bn2(output)
        output = F.interpolate(output, size=(int(input.size(2)), int(input.size(3))), mode='nearest')

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + input)



class FSF_32(nn.Module):
    def __init__(self, input, output, dropprob, dilated, kernel_size):
        super().__init__()

        self.dropout = nn.Dropout2d(dropprob)
        self.conv3x1_0 = nn.Conv2d(input, output, (1, 1), stride=1, bias=True)
        self.conv1x3_1 = nn.Conv2d(output, output, (1, 1), stride=1, bias=True)
        self.conv3x3_2 = nn.Conv2d(output, output, (kernel_size, kernel_size), stride=1, padding=(dilated, dilated), bias=True, dilation=(dilated, dilated), groups=output)
        self.bn1 = nn.BatchNorm2d(output, eps=1e-03)
        self.bn2 = nn.BatchNorm2d(output, eps=1e-03)

    def forward(self, input):

        output = F.interpolate(input, size=(int(input.size(2)) // 2, int(input.size(3)) // 2), mode='nearest')
        output = self.conv3x1_0(output)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.conv3x3_2(output)
        output = self.bn2(output)
        output = F.interpolate(output, size=(int(input.size(2)), int(input.size(3) )), mode='nearest')

        if (self.dropout.p != 0):
            output = self.dropout(output)


        return F.relu(output + input)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.initial_block = Stem()

        # 8-downsample rate
        self.conv1_8 = FSF_8(64, 64, 0.01, 1, 3)
        self.conv2_8 = FSF_8(64, 64, 0.02, 2, 3)
        self.conv3_8 = FSF_8(64, 64, 0.03, 4, 3)
        self.conv4_8 = FSF_8(64, 64, 0.04, 8, 3)

        # connection conv
        self.conv1x1_8_16_channel = nn.Conv2d(64, 96, 1, stride=1, padding=0)
        self.bn_conv1x1_8_16_channel = nn.BatchNorm2d(96, eps=1e-03)

        # 16-downsample rate
        self.conv1_16 = FSF_16(96, 96, 0.05, 1, 3)
        self.conv2_16 = FSF_16(96, 96, 0.06, 2, 3)
        self.conv3_16 = FSF_16(96, 96, 0.07, 4, 3)
        self.conv4_16 = FSF_16(96, 96, 0.08, 8, 3)

        # connection conv
        self.conv1x1_32_64_channel = nn.Conv2d(96, 128, 1, stride=1, padding=0)
        self.bn_conv1x1_32_64_channel = nn.BatchNorm2d(128, eps=1e-03)

        # 32-downsample rate
        self.conv1_32 = FSF_32(128, 128, 0.05, 1, 3)
        self.conv2_32 = FSF_32(128, 128, 0.06, 2, 3)
        self.conv3_32 = FSF_32(128, 128, 0.07, 4, 3)
        self.conv4_32 = FSF_32(128, 128, 0.08, 8, 3)

        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):

        output_8 = self.initial_block(input)

        output_8 = self.conv1_8(output_8)
        output_8 = self.conv2_8(output_8)
        output_8 = self.conv3_8(output_8)
        output_8 = self.conv4_8(output_8)


        output_16 = F.interpolate(output_8, size=(int(output_8.size(2)) // 2, int(output_8.size(3)) // 2), mode='nearest')
        output_16 = self.conv1x1_8_16_channel(output_16)
        output_16 = self.bn_conv1x1_8_16_channel(output_16)
        output_16 = F.relu(output_16)

        output_16 = self.conv1_16(output_16)
        output_16 = self.conv2_16(output_16)
        output_16 = self.conv3_16(output_16)
        output_16 = self.conv4_16(output_16)


        output_32 = F.interpolate(output_16, size=(int(output_16.size(2)) //2, int(output_16.size(3)) // 2), mode='nearest')
        output_32 = self.conv1x1_32_64_channel(output_32)
        output_32 = self.bn_conv1x1_32_64_channel(output_32)
        output_32 = F.relu(output_32)

        output_32 = self.conv1_32(output_32)
        output_32 = self.conv2_32(output_32)
        output_32 = self.conv3_32(output_32)
        output_32 = self.conv4_32(output_32)

        if predict:
            output_32 = self.output_conv(output_32)


        return output_8, output_16, output_32


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1x1_merge_32_16 = nn.Conv2d(128, 32, 1, stride=1, padding=0)
        self.bn_merge1x1_32_16 = nn.BatchNorm2d(32, eps=1e-03)

        self.conv3x3_merge_32_16 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                                             bias=True)
        self.bn_merge3x3_32_16 = nn.BatchNorm2d(32, eps=1e-3)

        self.conv1x1_merge_16_8 = nn.Conv2d(32, 48, 1, stride=1, padding=0)
        self.bn_merge1x1_16_8 = nn.BatchNorm2d(48, eps=1e-03)

        self.conv3x3_merge_16_8 = nn.Conv2d(112, 32, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                                            bias=True)
        self.bn_merge3x3_16_8 = nn.BatchNorm2d(32, eps=1e-3)

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(32, num_classes, 1, stride=1, padding=0, bias=True)


    def forward(self, output_8, output_16, output_32):

        output = self.conv1x1_merge_32_16(output_32)
        output = self.bn_merge1x1_32_16(output)
        output = F.relu(output)

        # 32 to 16
        output = F.interpolate(output, size=(int(output.size(2) * 2), int(output.size(3) * 2)), mode='nearest')
        output = torch.cat([output, output_16], 1)# 16 to 8
        output = self.conv3x3_merge_32_16(output)
        output = self.bn_merge3x3_32_16(output)
        output = F.relu(output)
        output = self.conv1x1_merge_16_8(output)
        output = self.bn_merge1x1_16_8(output)
        output = F.relu(output)

        # 16 to 8
        output = F.interpolate(output, size=(int(output_8.size(2)), int(output_8.size(3) )), mode='nearest')
        output = torch.cat([output, output_8], 1)
        output = self.conv3x3_merge_16_8(output)
        output = self.bn_merge3x3_16_8(output)
        output = F.relu(output)
        output = self.output_conv(output)

        # 8 to ori
        output = F.interpolate(output, size=(int(output_8.size(2) * 8), int(output_8.size(3) * 8)), mode='nearest')

        return output


class FSFNet(nn.Module):
    def __init__(self, num_classes, encoder=None):  # use encoder to pass pretrained encoder
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            output_8, output_16, output_32= self.encoder.forward(input, predict=True)
            return output_32
        else:
            output_8, output_16, output_32 = self.encoder(input)  # predict=False by default
            return self.decoder.forward(output_8, output_16, output_32)

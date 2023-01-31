import torch
from torch import nn

__all__ = ['UNet', 'SRNN']


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.dropout = nn.Dropout(p = 0.2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = self.relu(out)

        return out



class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=1, **kwargs):
        print("updated_1")
        super().__init__()

        nb_filter = [16, 32, 64, 128, 256]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.pos_branch = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=15 * 15 * 256,out_features=256),
                                        nn.Linear(in_features=256,out_features=1),
                                        nn.Sigmoid()
                                        )


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)

        pos = self.pos_branch(x4_0)
        return output, pos

class SRNN(nn.Module):

    def __init__(self,num_classes):
        super().__init__()

        self.block1 = SRNN._block(num_classes,16,16)
        self.block2 = SRNN._block(16,32,32)
        self.block3 = SRNN._block(32,64,64)
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.Linear1 = nn.Linear(100,64)
        self.Linear2 = nn.Linear(64,100)

        self.block5 = SRNN._block_2(1,64,64,7,3)
        self.block6 = SRNN._block_2(64,32,32,4,2)
        self.block7 = SRNN._block_2(32, 16, 16, 4, 2)
        self.block8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4,
                               stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=num_classes, kernel_size=3, stride=1, padding=1),
        )


    def forward(self,input):
        output = self.block1(input)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.flatten(output)
        output = self.Linear1(output)
        output = self.Linear2(output)
        output = output.reshape(-1,1,10,10)
        output = self.block5(output)
        output = self.block6(output)
        output = self.block7(output)
        output = self.block8(output)

        return output

    @staticmethod
    def _block(in_channel,middle_channel,out_channel):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=middle_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=middle_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        return block

    @staticmethod
    def _block_2(in_channel,middle_channel,out_channel,kernel_size,stride):
        p = int((kernel_size-stride)/2)
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channel, out_channels=middle_channel, kernel_size=kernel_size, stride=stride, padding=p),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=middle_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        return block


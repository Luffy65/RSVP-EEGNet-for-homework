import torch
import torch.nn as nn

class EEGNet(nn.Module):
    """
    four block:
    1. conv2d
    2. depthwiseconv2d
    3. separableconv2d
    4. classify
    """
    def __init__(self, batch_size=4, num_class=2):
        super(EEGNet, self).__init__()
        self.batch_size = batch_size
        # 1. conv2d
        self.block1 = nn.Sequential()
        self.block1_conv = nn.Conv2d(in_channels=1,
                                     out_channels=8,
                                     kernel_size=(1, 128),
                                     padding=(0, 64),
                                     bias=False
                                     )
        self.block1.add_module('conv1', self.block1_conv)
        self.block1.add_module('norm1', nn.BatchNorm2d(8))

        # 2. depthwiseconv2d
        self.block2 = nn.Sequential()
        # [N, 8, 32, 1152] -> [N, 16, 1, 1152]
        self.block2.add_module('conv2', nn.Conv2d(in_channels=8,
                                                  out_channels=16,
                                                  kernel_size=(32, 1),
                                                  groups=2,
                                                  bias=False))
        self.block2.add_module('act1', nn.ELU())
        # [N, 16, 1, 1152] -> [N, 16, 1, 288]
        self.block2.add_module('pool1', nn.AvgPool2d(kernel_size=(1, 4)))
        self.block2.add_module('drop1', nn.Dropout(p=0.5))

        # 3. separableconv2d
        self.block3 = nn.Sequential()
        self.block3.add_module('conv3', nn.Conv2d(in_channels=16,
                                                  out_channels=16,
                                                  kernel_size=(1, 64),
                                                  padding=(0, 32),
                                                  groups=16,
                                                  bias=False
                                                  ))
        self.block3.add_module('conv4', nn.Conv2d(in_channels=16,
                                                  out_channels=16,
                                                  kernel_size=(1, 1),
                                                  bias=False))
        self.block3.add_module('norm2', nn.BatchNorm2d(16))
        self.block3.add_module('act2', nn.ELU())
        self.block3.add_module('pool2', nn.AvgPool2d(kernel_size=(1, 8)))
        self.block3.add_module('drop2', nn.Dropout(p=0.5))

        # 4. classify
        self.classify = nn.Sequential(nn.Linear(576, num_class))

    def forward(self, x):
        # [B, 32, 1152] -> [B, 1, 32, 1152]
        x = torch.reshape(x, (self.batch_size, 1, 32, 1152))

        # [B, 1, 32, 1152] -> [B, 1, 32, 1151]
        x = x[:, :, :, range(1151)]

        # [B, 1, 32, 1151] -> [B, 8, 32, 1152]
        x = self.block1(x)

        # [B, 8, 32, 1152] -> [B, 16, 1, 1152] -> [B, 16, 1, 288]
        x = self.block2(x)

        # [B, 16, 1, 288] -> [B, 16, 1, 287]
        x = x[:, :, :, range(287)]

        # [B, 16, 1, 287] -> [B, 16, 1, 36]
        x = self.block3(x)

        # [B, 16, 1, 36] -> [B, 576]
        x = x.view(x.size(0), -1)

        # [B, 576] -> [B, num_class]
        x = self.classify(x)

        # x = nn.functional.softmax(x, dim=1)

        return x

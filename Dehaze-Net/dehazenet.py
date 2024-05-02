import torch
import torch.nn as nn
import torch.nn.functional as F

class DehazeNet(nn.Module):
    def __init__(self, input=16, groups=4):
        super(DehazeNet, self).__init__()
        self.input = input
        self.groups = groups
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.input, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=self.input, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=self.input, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=self.input, kernel_size=7, padding=3)
        
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=6, padding=2)

        # Used ChatGPT to solve this bug in the output channel while implementing the paper
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    #local extremum
    def Maxout(self, x, groups):
        x = x.reshape(x.shape[0], groups, x.shape[1]//groups, x.shape[2], x.shape[3])
        x, _ = torch.max(x, dim=2, keepdim=True)
        out = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        return out

    #a bounded relu function suggested in the original paper
    def BRelu(self, x):
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        x = torch.max(x, zeros)
        x = torch.min(x, ones)
        return x

    def forward(self, x):
        expected_height = x.size(2)
        expected_width = x.size(3)
        out = self.conv1(x)
        out = self.Maxout(out, self.groups)
        out1 = self.conv2(out)
        out2 = self.conv3(out)
        out3 = self.conv4(out)
        y = torch.cat((out1, out2, out3), dim=1)
        y = self.maxpool(y)
        y = self.conv5(y)
        if y.size(2) != expected_height:
            diff = expected_height - y.size(2)
            y = F.pad(y, (0, 0, diff // 2, diff - diff // 2))  # Padding bottom and top
        if y.size(3) != expected_width:
            diff = expected_width - y.size(3)
            y = F.pad(y, (diff // 2, diff - diff // 2, 0, 0))
        y = self.BRelu(y)
        return y

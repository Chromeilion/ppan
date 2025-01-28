'''
Sight-to-Sound Network (S2SNet) - basic modules and arch of the custom ResNet;
implementation follows the architecture described in the paper:
https://www.robots.ox.ac.uk/~vgg/publications/2020/Koepke20/koepke20.pdf
'''

import torch
import torch.nn as nn

from torchvision.models.resnet import BasicBlock, ResNet


class AggregationModule(nn.Module):
    '''
    Aggregation module - allows the network to make use of temporal information
        by performing channel-wise temporal weighted average of the input frames.
    params:
        k: int, number of frames to aggregate (default=5)
    '''
    def __init__(self, k=5):
        super(AggregationModule, self).__init__()
        self.k = k
        self.conv = nn.Conv3d(64, 64, kernel_size=(k, 1, 1), stride=1, padding=0, bias=False)
        
    def forward(self, x):
        n_frames = x.shape[2]
        assert n_frames == self.k, 'k must be equal to the number of frames in the input tensor'

        out = self.conv(x)
        out = out.squeeze(2)
        
        return out
  

class SlopeModule(nn.Module):
    '''
    Slope module - allows the network to preserve spatial information,
    by use of a slope vector representanting the positions of the 88 piano keys.
    '''
    
    def __init__(self):
        super(SlopeModule, self).__init__()

        self.conv11 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv12 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(320, 256, kernel_size=3, stride=1, padding=1, bias=False)

        # expand tensor to [64, 10, 50] to concat to output of previous layer
        # TODO: a bit unclear in the paper, so this is a reasonable guess
        self.expand = nn.Sequential(
            nn.Conv1d(88, 10 * 50, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Unflatten(1, (10, 50))
        )

    def forward(self, x, s):
        out = self.conv11(s)
        out = self.conv12(out)
        
        # rearrange dimensions to performorm the expand operation
        out = torch.movedim(out, -1, 1)
        out = self.expand(out)
        out = torch.movedim(out, -1, 1)

        out = torch.concat([x, out], 1)
        out = self.conv2(out)
        
        return out


class S2SNet(ResNet):
    '''
    S2SNet - full model with additional modules
    '''
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=88):
        super(S2SNet, self).__init__(block, layers, num_classes=num_classes)
        
        self.aggregation = AggregationModule()
        self.slope = SlopeModule()

        # number of input channels equals 1 for grayscale frames
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # first block where we deal with the separate frames 
        # before aggregation and passing through the rest of the network
        self.block0 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool
        )        

    @staticmethod
    def _make_slope_vector(batch_size):
        s = torch.arange(1, 89).float()
        s = s / 88
        s = s.view(1, 1, 88)
        s = s.repeat(batch_size, 1, 1)
        return s

    def forward(self, x):
        # TODO: is this necessary to do at each forward pass?
        batch_size = x.shape[0]
        s = self._make_slope_vector(batch_size)

        # pass each frame through block0 and stack them before aggregation
        # (we are initially treating each frame as a separate channel)
        # TODO: can this be parallelized?
        frame_stack = []
        for i in range(x.shape[1]):
            frame = x[:, i].unsqueeze(1)
            frame_stack.append(self.block0(frame))
        x = torch.stack(frame_stack, dim=2)
        x = self.aggregation(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.slope(x, s)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
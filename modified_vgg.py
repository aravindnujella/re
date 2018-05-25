# Load modified vgg, return last three layers just before max pool

# cfg = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }
import torch.nn as nn
import torchvision.models as models


class vgg11_features(nn.Module):

    def __init__(self, vgg_weights=True):
        super(vgg11_features, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        # cfg = [4,72,144,288,288,576,576,576,576]
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 72, (3, 3), (1, 1)),self.relu,
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(72, 144, (3, 3), (1, 1)),self.relu,
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(144, 288, (3, 3), (1, 1)),self.relu,
            nn.Conv2d(288, 288, (3, 3), (1, 1)),self.relu,
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(288, 576, (3, 3), (1, 1)),self.relu,
            nn.Conv2d(576, 576, (3, 3), (1, 1)),self.relu,
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(576, 576, (3, 3), (1, 1)),self.relu,
            nn.Conv2d(576, 576, (3, 3), (1, 1)),self.relu,
        )

        self.wing_conv5 = nn.Conv2d(576, 64, (3, 3), (1, 1))
        self.wing_conv4 = nn.Conv2d(576, 64, (3, 3), (1, 1))
        self.wing_conv3 = nn.Conv2d(288, 32, (3, 3), (1, 1))

        # initialize with vgg weights
        if vgg_weights == True:
            self.init_weights()

    def forward(self, x):
        outs = []
        x = self.layer1(x); x = self.pool(x)
        x = self.layer2(x); x = self.pool(x)
        x = self.layer3(x); outs.append(self.wing_conv3(x)); x = self.pool(x)
        x = self.layer4(x); outs.append(self.wing_conv4(x)); x= self.pool(x)
        x = self.layer5(x); outs.append(self.wing_conv5(x)); x = self.pool(x)
        return x, outs

    def init_weights(self):
        vgg = models.vgg11(pretrained=True)
        _shapes = []
        for child in vgg.features.children():
            if isinstance(child, nn.Conv2d):
                _shapes.append(child.weight.shape)
        d_in = 1
        i = 0
        new_filters = []
        for child in vgg.features.children():
            if isinstance(child, nn.Conv2d):
                cur_in = _shapes[i][1]
                cur_out = _shapes[i][0]
                kernel_shape = _shapes[i][2:]
                d_out = cur_out // 8
                fan_in = kernel_shape[0] * kernel_shape[1]
                # ignore_filters: cur_out, cur_in + d_in, kernel_shape
                c = torch.zeros((cur_out, d_in) + kernel_shape, requires_grad=True)
                ignore_filters = torch.cat([child.weight, c], 1)
                # copy_filters: d_out, cur_in+d_in, kernel_shape
                a = torch.zeros((d_out, cur_in,) + kernel_shape, requires_grad=True)
                b = torch.ones((d_out, d_in,) + kernel_shape, requires_grad=True) / fan_in
                copy_filters = torch.cat([a, b], 1)
                new_conv = torch.cat([ignore_filters, copy_filters], 0)
                new_filters.append(new_conv)
                d_in = d_out
                i += 1
        for

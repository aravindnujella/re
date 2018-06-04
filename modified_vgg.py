# Load modified vgg, return last three layers just before max pool

# cfg = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class vgg11_features(nn.Module):

    def __init__(self, pre_trained_weights=True):
        super(vgg11_features, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # cfg = [4,72,144,288,288,576,576,576,576]
        self.layer1 = nn.Sequential(
            nn.Conv2d(3+4, 72, (3, 3), padding=(1, 1)), nn.BatchNorm2d(72), self.relu,
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(72, 144, (3, 3), padding=(1, 1)), nn.BatchNorm2d(144), self.relu,
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(144, 288, (3, 3), padding=(1, 1)), nn.BatchNorm2d(288), self.relu,
            nn.Conv2d(288, 288, (3, 3), padding=(1, 1)), nn.BatchNorm2d(288), self.relu,
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(288, 576, (3, 3), padding=(1, 1)), nn.BatchNorm2d(576), self.relu,
            nn.Conv2d(576, 576, (3, 3), padding=(1, 1)), nn.BatchNorm2d(576), self.relu,
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(576, 576, (3, 3), padding=(1, 1)), nn.BatchNorm2d(576), self.relu,
            nn.Conv2d(576, 576, (3, 3), padding=(1, 1)), nn.BatchNorm2d(576), self.relu,
        )

        self.wing_conv5 = nn.Conv2d(576, 64, (3, 3), padding=(1, 1))
        self.wing_conv4 = nn.Conv2d(576, 64, (3, 3), padding=(1, 1))
        self.wing_conv3 = nn.Conv2d(288, 32, (3, 3), padding=(1, 1))

        # initialize with vgg weights
        if pre_trained_weights == True:
            self.init_weights()

    def forward(self, x):
        outs = []
        x = self.layer1(x); x = self.pool(x);
        x = self.layer2(x); x = self.pool(x);
        x = self.layer3(x); outs.append(self.wing_conv3(x)); x = self.pool(x);
        x = self.layer4(x); outs.append(self.wing_conv4(x)); x = self.pool(x);
        x = self.layer5(x); outs.append(self.wing_conv5(x)); x = self.pool(x);
        return x, outs

    def init_weights(self):
        _shapes = [[] for i in range(5)]
        l = 0
        vgg = models.vgg11(pretrained=True)
        for child in vgg.features.children():
            if isinstance(child, nn.Conv2d):
                _shapes[l].append(child.weight.shape)
            elif isinstance(child, nn.MaxPool2d):
                l += 1
        d_in = 4
        new_filters = [[] for l in range(5)]
        i = 0; l = 0
        for child in vgg.features.children():
            if isinstance(child, nn.Conv2d):
                cur_in = _shapes[l][i][1]
                cur_out = _shapes[l][i][0]
                kernel_shape = _shapes[l][i][2:]
                d_out = cur_out // 8
                fan_in = kernel_shape[0] * kernel_shape[1]
                # ignore_filters: cur_out, cur_in + d_in, kernel_shape
                c = torch.zeros((cur_out, d_in) + kernel_shape)
                ignore_filters = torch.cat([child.weight, c], 1)
                a = torch.zeros((d_out, cur_in,) + kernel_shape)
                idx = np.array([i % d_in for i in range(d_out)])
                b = np.zeros((d_out, d_in))
                b[range(d_out), idx] = 1
                b = torch.from_numpy(b).unsqueeze(-1).unsqueeze(-1).float()
                b = b.repeat([1, 1, kernel_shape[0], kernel_shape[1]]) / fan_in
                # print(type(a),type(b))
                copy_filters = torch.cat([a, b], 1)
                new_conv = torch.cat([ignore_filters, copy_filters], 0)
                new_filters[l].append(new_conv)
                d_in = d_out
                i += 1
            elif isinstance(child, nn.MaxPool2d):
                l += 1
                i = 0
        l = 0
        for name, child in self.named_children():
            if name[:-1] == "layer":
                k = 0
                for gc in child.children():
                    if isinstance(gc, nn.Conv2d):
                        gc.weight = nn.Parameter(new_filters[l][k])
                        k += 1
                    elif isinstance(gc, nn.BatchNorm2d):
                        nn.init.constant_(gc.weight, 1)
                        nn.init.constant_(gc.bias, 0)
                l += 1
            elif name[:-1] == "wing_conv":
                nn.init.xavier_uniform_(child.weight)


class vgg16_features(nn.Module):

    def __init__(self, pre_trained_weights=True):
        super(vgg11_features, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # cfg = [4,72,144,288,288,576,576,576,576]
        self.layer1 = nn.Sequential(
            nn.Conv2d(3+4, 72, (3, 3), padding=(1, 1)), nn.BatchNorm2d(72), self.relu,
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(72, 144, (3, 3), padding=(1, 1)), nn.BatchNorm2d(144), self.relu,
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(144, 288, (3, 3), padding=(1, 1)), nn.BatchNorm2d(288), self.relu,
            nn.Conv2d(288, 288, (3, 3), padding=(1, 1)), nn.BatchNorm2d(288), self.relu,
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(288, 576, (3, 3), padding=(1, 1)), nn.BatchNorm2d(576), self.relu,
            nn.Conv2d(576, 576, (3, 3), padding=(1, 1)), nn.BatchNorm2d(576), self.relu,
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(576, 576, (3, 3), padding=(1, 1)), nn.BatchNorm2d(576), self.relu,
            nn.Conv2d(576, 576, (3, 3), padding=(1, 1)), nn.BatchNorm2d(576), self.relu,
        )

        self.wing_conv5 = nn.Conv2d(576, 64, (3, 3), padding=(1, 1))
        self.wing_conv4 = nn.Conv2d(576, 64, (3, 3), padding=(1, 1))
        self.wing_conv3 = nn.Conv2d(288, 32, (3, 3), padding=(1, 1))

        # initialize with vgg weights
        if pre_trained_weights == True:
            self.init_weights()

    def forward(self, x):
        outs = []
        x = self.layer1(x); x = self.pool(x);
        x = self.layer2(x); x = self.pool(x);
        x = self.layer3(x); outs.append(self.wing_conv3(x)); x = self.pool(x);
        x = self.layer4(x); outs.append(self.wing_conv4(x)); x = self.pool(x);
        x = self.layer5(x); outs.append(self.wing_conv5(x)); x = self.pool(x);
        return x, outs

    def init_weights(self):
        _shapes = [[] for i in range(5)]
        l = 0
        vgg = models.vgg11(pretrained=True)
        for child in vgg.features.children():
            if isinstance(child, nn.Conv2d):
                _shapes[l].append(child.weight.shape)
            elif isinstance(child, nn.MaxPool2d):
                l += 1
        d_in = 4
        new_filters = [[] for l in range(5)]
        i = 0; l = 0
        for child in vgg.features.children():
            if isinstance(child, nn.Conv2d):
                cur_in = _shapes[l][i][1]
                cur_out = _shapes[l][i][0]
                kernel_shape = _shapes[l][i][2:]
                d_out = cur_out // 8
                fan_in = kernel_shape[0] * kernel_shape[1]
                # ignore_filters: cur_out, cur_in + d_in, kernel_shape
                c = torch.zeros((cur_out, d_in) + kernel_shape)
                ignore_filters = torch.cat([child.weight, c], 1)
                a = torch.zeros((d_out, cur_in,) + kernel_shape)
                idx = np.array([i % d_in for i in range(d_out)])
                b = np.zeros((d_out, d_in))
                b[range(d_out), idx] = 1
                b = torch.from_numpy(b).unsqueeze(-1).unsqueeze(-1).float()
                b = b.repeat([1, 1, kernel_shape[0], kernel_shape[1]]) / fan_in
                # print(type(a),type(b))
                copy_filters = torch.cat([a, b], 1)
                new_conv = torch.cat([ignore_filters, copy_filters], 0)
                new_filters[l].append(new_conv)
                d_in = d_out
                i += 1
            elif isinstance(child, nn.MaxPool2d):
                l += 1
                i = 0
        l = 0
        for name, child in self.named_children():
            if name[:-1] == "layer":
                k = 0
                for gc in child.children():
                    if isinstance(gc, nn.Conv2d):
                        gc.weight = nn.Parameter(new_filters[l][k])
                        k += 1
                    elif isinstance(gc, nn.BatchNorm2d):
                        nn.init.constant_(gc.weight, 1)
                        nn.init.constant_(gc.bias, 0)
                l += 1
            elif name[:-1] == "wing_conv":
                nn.init.xavier_uniform_(child.weight)


if __name__ == '__main__':
    import numpy as np
    net = vgg11_features(pre_trained_weights=True)
    torch.save(net.state_dict(), "./models/vgg11_features_7.pt")
    # net.load_state_dict(torch.load("./models/vgg11_features.pt"))
    # net_parameters = filter(lambda p: p.requires_grad, net.parameters())
    # params = sum([np.prod(p.size()) for p in net_parameters])
    # print(params)

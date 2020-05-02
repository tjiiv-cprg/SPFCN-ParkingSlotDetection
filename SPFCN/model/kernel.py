import torch
from torch import nn


class BasicModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=(kernel - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, feature):
        return self.activate(self.bn(self.conv(feature)))

    def rebuild_conv(self, weight, bias, device):
        out_channels, in_channels, _, _ = weight.shape
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              self.conv.kernel_size,
                              self.conv.stride,
                              self.conv.padding,
                              bias=True).to(device)
        self.conv.weight = nn.Parameter(weight)
        self.conv.bias = nn.Parameter(bias)

    def get_alpha(self):
        return nn.Softmax(dim=0)(torch.cat([torch.norm(weight, p=2).view(1) for weight in self.conv.weight], dim=0))

    def prune_in_channels(self, index, device):
        w = torch.cat((self.conv.weight[:, 0:index], self.conv.weight[:, index + 1:]), dim=1)
        b = self.conv.bias
        self.rebuild_conv(w, b, device)

    def prune_out_channels(self, index, device):
        w = torch.cat((self.conv.weight[0:index], self.conv.weight[index + 1:]), dim=0)
        b = torch.cat((self.conv.bias[0:index], self.conv.bias[index + 1:]), dim=0)
        self.rebuild_conv(w, b, device)

        w = torch.cat((self.bn.weight[0:index], self.bn.weight[index + 1:]), dim=0)
        b = torch.cat((self.bn.bias[0:index], self.bn.bias[index + 1:]), dim=0)
        self.bn = nn.BatchNorm2d(self.bn.num_features - 1).to(device)
        self.bn.weight = nn.Parameter(w)
        self.bn.bias = nn.Parameter(b)

    def merge(self, device):
        if not hasattr(self, 'bn'):
            return
        mean = self.bn.running_mean
        var_sqrt = torch.sqrt(self.bn.running_var + self.bn.eps)
        beta = self.bn.weight
        gamma = self.bn.bias
        del self.bn

        w = self.conv.weight * (beta / var_sqrt).reshape([self.conv.out_channels, 1, 1, 1])
        b = (self.conv.bias - mean) / var_sqrt * beta + gamma
        self.rebuild_conv(w, b, device)

        self.forward = lambda feature: self.activate(self.conv(feature))


class SelectKernel(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        nn.Module.__init__(self)
        self.conv3 = BasicModule(in_channels, out_channels, 3, stride=stride)
        self.conv5 = BasicModule(in_channels, out_channels, 5, stride=stride)
        self.cen = nn.Sequential(nn.Linear(8, 8), nn.ReLU(inplace=True), nn.Linear(8, 2), nn.Softmax(dim=0))

        self.alpha = torch.Tensor([1., 0.])

    def forward(self, feature):
        vector = torch.cat(
            [torch.mean(self.conv3.bn.running_mean).view(1), torch.std(self.conv3.bn.running_mean).view(1),
             torch.mean(self.conv3.bn.running_var).view(1), torch.std(self.conv3.bn.running_var).view(1),
             torch.mean(self.conv5.bn.running_mean).view(1), torch.std(self.conv5.bn.running_mean).view(1),
             torch.mean(self.conv5.bn.running_var).view(1), torch.std(self.conv5.bn.running_var).view(1)], dim=0)
        self.alpha = self.cen(vector)
        return self.alpha[0] * self.conv3(feature) + self.alpha[1] * self.conv5(feature)

    def auto_select(self, threshold=0.9):
        return self.conv3 if self.alpha[0] > threshold else self.conv5 if self.alpha[1] > threshold else None

    def enforce_select(self):
        return self.conv3 if self.alpha[0] > self.alpha[1] else self.conv5


class SpModule(nn.Module):
    def __init__(self, in_channels_id, out_channels_id, device,
                 in_channels, out_channels, stride=1):
        nn.Module.__init__(self)
        self.in_channels_id = in_channels_id
        self.out_channels_id = out_channels_id
        self.device = device
        self.conv_module = SelectKernel(in_channels, out_channels, stride).to(device)

    def forward(self, feature):
        return self.conv_module(feature)

    def get_regularization(self):
        if isinstance(self.conv_module, SelectKernel):
            return -10 * torch.log(self.conv_module.alpha[0] ** 2 + self.conv_module.alpha[1] ** 2)
        elif isinstance(self.conv_module, BasicModule):
            return 0.01 * torch.norm(self.conv_module.conv.weight, p=1)

    def prune(self, group_id, prune_index):
        if self.in_channels_id == group_id:
            self.conv_module.prune_in_channels(prune_index, self.device)
        elif self.out_channels_id == group_id:
            self.conv_module.prune_out_channels(prune_index, self.device)

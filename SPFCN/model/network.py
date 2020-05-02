import torch
import torch.nn.functional as f
from torch import nn

from .kernel import BasicModule, SelectKernel, SpModule


class Hourglass(nn.Module):
    def __init__(self, dim_encoder, group_id, device):
        nn.Module.__init__(self)
        curr_dim = dim_encoder[0]
        next_dim = dim_encoder[1]
        self.front = SpModule(group_id - 1, group_id, device, curr_dim, next_dim, stride=2)

        self.middle = nn.Sequential(
            SpModule(group_id, group_id + 1, device, next_dim, next_dim),
            SpModule(group_id + 1, group_id + 2, device, next_dim, next_dim),
            SpModule(group_id + 2, group_id, device, next_dim, next_dim),
        ) if len(dim_encoder) <= 2 else Hourglass(dim_encoder[1:], group_id + 1, device)

        self.rear = SpModule(group_id, group_id - 1, device, next_dim, curr_dim)

    def forward(self, feature):
        front = self.front(feature)
        middle = self.middle(front)
        rear = f.interpolate(self.rear(middle), scale_factor=2)
        return feature + rear


class SEBlock(nn.Module):
    def __init__(self, ipc, sqz):
        nn.Module.__init__(self)
        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.att = nn.Sequential(nn.Linear(ipc, sqz), nn.ReLU(inplace=True), nn.Linear(sqz, ipc), nn.Sigmoid())

    def forward(self, feature):
        attention = self.pooling(feature).view(feature.size(0), -1)
        attention = self.att(attention).view(feature.size(0), -1, 1, 1)
        return attention * feature


class MarkHeading(nn.Module):
    def __init__(self, ipc, sqz):
        nn.Module.__init__(self)
        self.att = SEBlock(ipc, sqz)
        self.conv = nn.Conv2d(ipc, 1, kernel_size=1)
        self.activate = nn.Sigmoid()

    def forward(self, feature):
        return self.activate(self.conv(self.att(feature)))


class DirectionHeading(nn.Module):
    def __init__(self, ipc, sqz):
        nn.Module.__init__(self)
        self.att = SEBlock(ipc, sqz)
        self.conv = nn.Conv2d(ipc, 2, kernel_size=1)
        self.activate = nn.Tanh()

    def forward(self, feature):
        return self.activate(self.conv(self.att(feature)))


class SlotNetwork(nn.Module):
    def __init__(self, channel_encoder, device_id: int = 0):
        nn.Module.__init__(self)
        device = torch.device('cpu' if device_id < 0 else 'cuda:%d' % device_id)

        self.norm = nn.LayerNorm([3, 224, 224])
        self.conv1 = SpModule(-1, 0, device, 3, channel_encoder[0])
        self.backbone = Hourglass(channel_encoder, 1, device)
        self.conv2 = SpModule(0, -1, device, channel_encoder[0], channel_encoder[0])
        self.mark_heading = MarkHeading(channel_encoder[0], 16).to(device)
        self.direction_heading = DirectionHeading(channel_encoder[0], 16).to(device)

    def forward(self, feature):
        feature = self.conv1(feature)
        feature = self.backbone(feature)
        feature = self.conv2(feature)
        mark = self.mark_heading(feature)
        direction = self.direction_heading(feature)
        return mark, direction

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')
                module.bias.data.fill_(0)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def get_regularization(self):
        return sum(module.get_regularization() for module in self.modules() if isinstance(module, SpModule))

    def auto_select(self):
        count = 0
        for module in self.modules():
            if isinstance(module, SpModule) and isinstance(module.conv_module, SelectKernel):
                select_check = module.conv_module.auto_select()
                if select_check is not None:
                    module.conv_module = select_check
                    count += 1
        print("\tAuto Selected %d module(s)" % count)

    def enforce_select(self):
        count = 0
        for module in self.modules():
            if isinstance(module, SpModule) and isinstance(module.conv_module, SelectKernel):
                module.conv_module = module.conv_module.enforce_select()
                count += 1
        print("\tEnforce Selected %d module(s)" % count)

    def prune(self, group_id, prune_indices):
        for module in self.modules():
            if isinstance(module, SpModule) and isinstance(module.conv_module, BasicModule):
                module.prune(group_id, prune_indices)

    def prune_channel(self):
        prune_dict = dict()
        for module in self.modules():
            if isinstance(module, SpModule) and isinstance(module.conv_module, BasicModule):
                if module.out_channels_id in prune_dict.keys():
                    prune_dict[module.out_channels_id] += module.conv_module.get_alpha()
                else:
                    prune_dict[module.out_channels_id] = module.conv_module.get_alpha()
        prune_list = ((key, torch.min(value, dim=0)) for key, value in prune_dict.items() if key >= 0)
        min_group, min_value, min_index = 0, 1, 0
        for group_id, prune_info in prune_list:
            if prune_info.values < 0.02:
                print("\tAuto Pruned: Group {}, Channel {}, Contribution {:.3f}"
                      .format(group_id, prune_info.indices.item(), prune_info.values.item()))
                for module in self.modules():
                    if isinstance(module, SpModule) and isinstance(module.conv_module, BasicModule):
                        module.prune(group_id, prune_info.indices)
            elif prune_info.values < min_value:
                min_group, min_value, min_index = group_id, prune_info.values, prune_info.indices
        if min_value < 0.05:
            print("\tEnforce Pruned: Group {}, Channel {}, Contribution {:.3f}"
                  .format(min_group, min_index, min_value))
            for module in self.modules():
                if isinstance(module, SpModule) and isinstance(module.conv_module, BasicModule):
                    module.prune(min_group, min_index)

    def merge(self):
        for module in self.modules():
            if isinstance(module, SpModule) and isinstance(module.conv_module, BasicModule):
                module.conv_module.merge(module.device)

    def get_encoder(self):
        select_encoder = []
        channel_encoder = []
        for module in self.modules():
            if isinstance(module, SpModule) and isinstance(module.conv_module, BasicModule):
                select_encoder.append(module.conv_module.conv.kernel_size[0])
                channel_encoder.append(module.conv_module.conv.in_channels)
        channel_encoder.append(channel_encoder[-1])
        return select_encoder, channel_encoder

    def rebuild_network(self, select_encoder, channel_encoder):
        for module in self.modules():
            if isinstance(module, SpModule):
                module.conv_module = BasicModule(channel_encoder.pop(0),
                                                 channel_encoder[0],
                                                 select_encoder.pop(0),
                                                 module.conv_module.conv3.conv.stride).to(module.device)
                module.conv_module.merge(module.device)
                # del module.conv_module.bn
                # module.conv_module.forward = lambda x: module.conv_module.activate(module.conv_module.conv(x))

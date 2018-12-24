# -*- coding: utf-8 -*-
'''
网络模型的类
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import *
from operation_factory import *


class Model(nn.Module):
    def __init__(self, num_classes=20, decoder=True):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.decoder = decoder
        self.connection_information = {'start_end_node': [],
                                       'connection_mode': [],
                                       'start_operation_list': [],
                                       }

        self.output_name = ['input',
                            'downsample_1', 'regular_1',
                            'downsample_2', 'regular_2',
                            'downsample_3', 'regular_3',
                            'upsample_4', 'regular_4',
                            'upsample_5', 'regular_5',
                            'upsample_6', ]

        self.output_channel = [[1, 3],
                               [2, 16], [2, 16],
                               [4, 64], [4, 64],
                               [8, 128], [8, 128],
                               [4, 64], [4, 64],
                               [2, 2 * num_classes], [2, 2 * num_classes],
                               [1, num_classes]]
        # downsample block 1 /2
        self.downsample_1_config = Downsample_config(3, 16, 3, 2)
        self.downsample_1 = get_Downsample_block(self.downsample_1_config)  # /2

        # regular convolution
        block_num = 0
        in_channel = [16] * block_num
        out_channel = [16] * block_num
        self.regular_1_config = Regular_config(in_channel, out_channel, 3, 1, block_num)
        self.regular_1 = get_Regular_block(self.regular_1_config)

        # /4
        self.downsample_2_config = Downsample_config(16, 64, 3, 2)
        self.downsample_2 = get_Downsample_block(self.downsample_2_config)

        # regular2
        block_num = 5
        in_channel = [64] * block_num
        out_channel = [64] * block_num
        self.regular_2_config = Regular_config(in_channel, out_channel, 3, 1, 5)
        self.regular_2 = get_Regular_block(self.regular_2_config)

        # /8
        self.downsample_3_config = Downsample_config(64, 128, 3, 2)  # /8
        self.downsample_3 = get_Downsample_block(self.downsample_3_config)

        # regular3
        block_num = 8
        in_channel = [128] * block_num
        out_channel = [128] * block_num
        self.regular_3_config = Regular_config(in_channel, out_channel, 3, 1, 8)
        self.regular_3 = get_Regular_block(self.regular_3_config)

        if self.decoder:
            # 4
            self.upsample_4_config = Upsample_config(128, 64, 3, 2)
            self.upsample_4 = get_Upsample_block(self.upsample_4_config)

            # regular 4
            block_num = 2
            in_channel = [64] * block_num
            out_channel = [64] * block_num
            self.regular_4_config = Regular_config(in_channel, out_channel, 3, 1, 2)
            self.regular_4 = get_Regular_block(self.regular_4_config)

            # 2
            self.upsample_5_config = Upsample_config(64, 2 * num_classes, 3, 2)
            self.upsample_5 = get_Upsample_block(self.upsample_5_config)

            # regular_5
            block_num = 2
            in_channel = [2 * num_classes] * block_num
            out_channel = [2 * num_classes] * block_num
            self.regular_5_config = Regular_config(in_channel, out_channel, 3, 1, 2)
            self.regular_5 = get_Regular_block(self.regular_5_config)

            # 1
            self.upsample_6_config = Upsample_config(2 * num_classes, num_classes, 3, 2)
            self.upsample_6 = get_Upsample_block(self.upsample_6_config)

        else:
            self.project_layer = nn.Conv2d(128, num_classes, kernel_size=1)

        self.weights_init()

    def weights_init(self):
        for idx, m in enumerate(self.modules()):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def connection(self, start_node, end_node, start_tensor, end_tensor):
        if [start_node, end_node] not in self.connection_information['start_end_node']:
            return end_tensor
        else:
            index = [i for i, x in enumerate(self.connection_information['start_end_node'])
                     if x == [start_node, end_node]]
            for i in index:
                operation_ = self.connection_information['start_operation_list'][i]
                if len(operation_) == 0:
                    start_to_end_tensor = start_tensor
                else:
                    temp_tensor = start_tensor
                    for operator in operation_:
                        temp_tensor = operator(temp_tensor)
                    start_to_end_tensor = temp_tensor
                if self.connection_information['connection_mode'][i] == 'concat':
                    return torch.cat([start_to_end_tensor, end_tensor], dim=1)
                else:
                    return start_to_end_tensor + end_tensor

        pass

    def forward(self, x):
        connection_idx = 0
        input = x

        # /2
        downsample_1 = self.downsample_1(input)  # /2
        # connection check
        name = 'downsample_1'
        index = self.output_name.index(name)
        for i in range(index):
            start_name = self.output_name[i]
            downsample_1 = self.connection(start_name, name, locals()[start_name], downsample_1)

        temp_tensor = downsample_1
        # print temp_tensor.size()


        # regular 1
        if self.regular_1 is not None:
            for regular_layer in self.regular_1:
                temp_tensor = regular_layer(temp_tensor)
            regular_1 = temp_tensor
        else:
            regular_1 = downsample_1

        # connection check
        name = 'regular_1'
        index = self.output_name.index(name)
        for i in range(index):
            start_name = self.output_name[i]
            regular_1 = self.connection(start_name, name, locals()[start_name], regular_1)

        # /4
        downsample_2 = self.downsample_2(regular_1)
        # connection check
        name = 'downsample_2'
        index = self.output_name.index(name)
        for i in range(index):
            start_name = self.output_name[i]
            downsample_2 = self.connection(start_name, name, locals()[start_name], downsample_2)
        temp_tensor = downsample_2
        # print temp_tensor.size()

        # regular 2
        if self.regular_2 is not None:
            for regular_layer in self.regular_2:
                temp_tensor = regular_layer(temp_tensor)
            regular_2 = temp_tensor
        else:
            regular_2 = downsample_2

        # connection check
        name = 'regular_2'
        index = self.output_name.index(name)
        for i in range(index):
            start_name = self.output_name[i]
            regular_2 = self.connection(start_name, name, locals()[start_name], regular_2)

        # /8
        downsample_3 = self.downsample_3(regular_2)  # /8

        name = 'downsample_3'
        index = self.output_name.index(name)
        for i in range(index):
            start_name = self.output_name[i]
            downsample_3 = self.connection(start_name, name, locals()[start_name], downsample_3)

        temp_tensor = downsample_3
        # print temp_tensor.size()

        # regular 3
        if self.regular_3 is not None:
            for regular_layer in self.regular_3:
                temp_tensor = regular_layer(temp_tensor)
            regular_3 = temp_tensor
        else:
            regular_3 = downsample_3

        name = 'regular_3'
        index = self.output_name.index(name)
        for i in range(index):
            start_name = self.output_name[i]
            regular_3 = self.connection(start_name, name, locals()[start_name], regular_3)

        if not self.decoder:
            output = self.project_layer(regular_3)
            return F.interpolate \
                (output, scale_factor=8, mode='bilinear', align_corners=True)
        # /4
        upsample_4 = self.upsample_4(regular_3)  # /4
        name = 'upsample_4'
        index = self.output_name.index(name)
        for i in range(index):
            start_name = self.output_name[i]
            upsample_4 = self.connection(start_name, name, locals()[start_name], upsample_4)
        temp_tensor = upsample_4
        # print temp_tensor.size()

        # regular 4
        if self.regular_4 is not None:
            for regular_layer in self.regular_4:
                temp_tensor = regular_layer(temp_tensor)
            regular_4 = temp_tensor
        else:
            regular_4 = upsample_4

        name = 'regular_4'
        index = self.output_name.index(name)
        for i in range(index):
            start_name = self.output_name[i]
            regular_4 = self.connection(start_name, name, locals()[start_name], regular_4)

        # /2
        upsample_5 = self.upsample_5(regular_4)

        name = 'upsample_5'
        index = self.output_name.index(name)
        for i in range(index):
            start_name = self.output_name[i]
            upsample_5 = self.connection(start_name, name, locals()[start_name], upsample_5)

        temp_tensor = upsample_5
        # print temp_tensor.size()

        # regular 5
        if self.regular_5 is not None:
            for regular_layer in self.regular_5:
                temp_tensor = regular_layer(temp_tensor)
            regular_5 = temp_tensor
        else:
            regular_5 = upsample_5

        name = 'regular_5'
        index = self.output_name.index(name)
        for i in range(index):
            start_name = self.output_name[i]
            regular_5 = self.connection(start_name, name, locals()[start_name], regular_5)
        # 1
        upsample_6 = self.upsample_6(regular_5)
        return upsample_6


if __name__ == '__main__':
    input = Variable(torch.randn(1, 3, 512, 1024))
    # for the inference only mode
    net = Model().eval()
    setattr(net, 'test', 1)
    # for the training mode
    # net = EDANet().train()
    output = net(input)
    print(output.size())





# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import *
from operation_factory import *
import copy


def weights_init(model):
    for idx, m in enumerate(model.modules()):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    return model
def add_regular_block(block_name,basenet):
    assert hasattr(basenet,block_name), "Must contain the block"
    transormation_net = copy.deepcopy(basenet)
    config = getattr(basenet,block_name+'_config')
    config.block_num = 1
    insert_block = get_Regular_block(config)
    basenet_block = getattr(basenet,block_name)
    basenet_block.append(insert_block[0])
    #basenet_block.insert(position,insert_block[0])
    transormation_net_config = getattr(transormation_net,block_name+'_config')
    block_num = transormation_net_config.block_num
    channel_in = transormation_net_config.channel_in[1] if block_num >1 else transormation_net_config.channel_in[0]
    channel_out = transormation_net_config.channel_out[1] if block_num > 1 else transormation_net_config.channel_out[0]
    transormation_net_config.block_num += 1
    transormation_net_config.channel_in.append(channel_in)
    transormation_net_config.channel_out.append(channel_out)
    setattr(transormation_net,block_name,basenet_block)
    return transormation_net
def delete_regular_block(block_name,basenet):
    assert hasattr(basenet, block_name), "Must contain the block"
    transormation_net = copy.deepcopy(basenet)
    config = getattr(basenet, block_name + '_config')
    #assert position < config.block_num
    basenet_block = getattr(basenet, block_name)
    transormation_block_list = nn.ModuleList()
    position = len(basenet_block) -1
    for index,block in enumerate(basenet_block):
        if index == position:
            pass
        else:
            transormation_block_list.append(block)
    if len(transormation_block_list) == 0:
        setattr(transormation_net, block_name, None)
    else:
        setattr(transormation_net, block_name, transormation_block_list)
        transormation_net_config = getattr(transormation_net, block_name + '_config')
        transormation_net_config.block_num -= 1
        del transormation_net_config.channel_in[position]
        del transormation_net_config.channel_out[position]
    return transormation_net
'''
添加connection函数，添加不同stage 之间的connection
basenet： 输入网络结构，为一个Basenet类
start_node: 开始的node的名字
end_node: 结束的node的名字
spatial: max/avg/3x3_conv/upconv前面三个是降低维度，后面一个对spatial进行提升
channel_operation：1x1_conv，1x1的卷积操作
channel_dim: 输出的channel dimensions

输入 basenet对象，以及开始，结束的节点，然后对网络进行操作。
//为了方便计算，在进行add前进行是否包含concat的查询(并没有实现)
'''
def add_connection(basenet, start_node, end_node,
                   connection_mode='concat',
                   spatial_operation = None, 
                   channel_operation = None,
                   channel_dim = None):
    
    transormation_net = copy.deepcopy(basenet)
    start_node_index = transormation_net.output_name.index(start_node)
    start_node_size = transormation_net.output_channel[start_node_index]
    end_node_index = transormation_net.output_name.index(end_node)
    end_node_size = transormation_net.output_channel[end_node_index]

    operation_list = nn.ModuleList()
    transformed_size = start_node_size[1]
    if spatial_operation is not None:
        if start_node_size[0]<end_node_size[0]:
            iteration_num = end_node_size[0] / start_node_size[0] / 2
            for i in range(iteration_num):
                if spatial_operation == 'max':
                    operation_list.append(nn.MaxPool2d(3,stride = 2,padding = 1))
                elif spatial_operation == 'avg':
                    operation_list.append(nn.AvgPool2d(3,stride = 2,padding = 1))
                elif spatial_operation == '3x3_conv':
                    #在进行3x3卷积的时候，默认进行输入输出一样的变化
                    operation_list.append(nn.Conv2d(start_node_size[1], start_node_size[1],
                                                    (3, 3), stride=2, padding=(1, 1),
                                                    bias=False, dilation=1))
                else:
                    raise NotImplementedError(spatial_operation + " is not supported")
        elif start_node_size[0] > end_node_size[0]:
            iteration_num = end_node_size[0] / start_node_size[0] / 2
            for i in range(iteration_num):
                operation_list.append(nn.ConvTranspose2d(start_node_size[1], start_node_size[1],
                                                         3, stride=2, padding=1, output_padding=1, bias=False))
    if channel_operation is not None:
        end_size = end_node_size[1] if channel_dim is None else channel_dim
        operation_list.append(nn.Conv2d(start_node_size[1], end_size,
                                                    (1, 1), stride=1, padding=(0, 0),
                                                    bias=False))
        transformed_size = end_size

    if connection_mode == 'add':
        assert transformed_size == end_node_size[1],"in add mode, size should be same!"
    else:
        transformed_size = end_node_size[1] + transformed_size
        transormation_net.output_channel[end_node_index][1] = transformed_size
        #change the operation and weight in latter layer
        successor_layer_index = end_node_index + 1
        successor_layer_name = transormation_net.output_name[successor_layer_index]
        successor_layer = getattr(transormation_net,successor_layer_name)
        while successor_layer is None:
            successor_layer_index += 1
            successor_layer_name = transormation_net.output_name[successor_layer_index]
            successor_layer = getattr(transormation_net, successor_layer_name)
        successor_layer_config = getattr(transormation_net,successor_layer_name+'_config')
        if 'regular' in successor_layer_name:
            old_channel_in = successor_layer_config.channel_in[0]
            successor_layer_config.channel_in[0] = transformed_size
            channel_in = successor_layer_config.channel_in[0]
            channel_out = successor_layer_config.channel_out[0]
            kernel_size = successor_layer_config.kernel_size
            stride = successor_layer_config.stride
            first_successor_layer_ = CBR(channel_in,channel_out, kernel_size,stride)
            first_successor_layer_.conv = weights_init(first_successor_layer_.conv)
            first_successor_layer_.conv.weight.data[:,0:old_channel_in,:,:] = successor_layer[0].conv.weight.data
            successor_layer[0] = first_successor_layer_
            setattr(transormation_net, successor_layer_name,successor_layer)
        else:
            successor_layer = getattr(transormation_net, successor_layer_name)
            old_channel_in = successor_layer_config.channel_in
            successor_layer_config.channel_in = transformed_size
            channel_in = successor_layer_config.channel_in
            channel_out = successor_layer_config.channel_out
            kernel_size = successor_layer_config.kernel_size
            stride = successor_layer_config.stride
            first_successor_layer_ = CBR(channel_in, channel_out, kernel_size, stride)
            first_successor_layer_.conv = weights_init(first_successor_layer_.conv)
            first_successor_layer_.conv.weight.data[:, 0:old_channel_in, :, :] = successor_layer.conv.weight.data
            successor_layer = first_successor_layer_
            setattr(transormation_net, successor_layer_name, successor_layer)

    operation_list = weights_init(operation_list)
    transormation_net.connection_information['start_end_node'].append([start_node,end_node])
    transormation_net.connection_information['connection_mode'].append(connection_mode)
    transormation_net.connection_information['start_operation_list'].append(operation_list)
    return transormation_net
    pass
def delete_decoder(basenet):
    assert basenet.decoder == True, "Decoder already be deleted"
    transormation_net = model.Model(basenet.num_classes,False)

    for i in [1,2,3]:
        setattr(transormation_net,'downsample_%d_config'%i,copy.deepcopy(getattr(basenet,'downsample_%d_config'%i)))
        setattr(transormation_net, 'downsample_%d' % i, copy.deepcopy(getattr(basenet, 'downsample_%d'%i)))
        setattr(transormation_net, 'regular_%d_config' % i, copy.deepcopy(getattr(basenet, 'regular_%d_config'%i)))
        setattr(transormation_net, 'regular_%d' % i, copy.deepcopy(getattr(basenet, 'regular_%d'%i)))

    return transormation_net

if __name__ == '__main__':
    import model
    from utils import *
    from torchsummary import summary
    input = Variable(torch.randn(1, 3, 512, 1024))
    basenet = model.Model(num_classes=20)
    # output = basenet.forward(input) test
    #torch.save(basenet,'test.pth')
    # print netParams(basenet)
    # print output.size()
    #connection
    #summary(basenet,(3,512,1024),device="cpu")
    # net = add_connection(basenet, 'input', 'downsample_1',
    #                connection_mode='concat',
    #                spatial_operation='max',
    #                channel_operation=None,
    #                channel_dim=None)
    # output = net.forward(input)
    # print netParams(basenet)
    # print output.size()
    # summary(net, (3, 512, 1024), device="cpu")
    net = delete_regular_block('regular_2',basenet)
    output = net.forward(input)
    summary(net, (3, 512, 1024), device="cpu")
    net = add_regular_block('regular_2',net)
    output = net.forward(input)
    summary(net, (3, 512, 1024), device="cpu")
    net = delete_decoder(net)
    output = net.forward(input)
    summary(net, (3, 512, 1024), device="cpu")
    #torch.save(net, 'test2.pth')
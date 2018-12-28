# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BR(nn.Module):
    def __init__(self, nOut):
        super(BR,self).__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)
    def forward(self, input):
        output = self.bn(input)
        output = self.act(output)
        return output
class CBR(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1,dilated = 1):
        super(CBR,self).__init__()
        padding = int((kSize - 1)/2)*dilated
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=dilated)
        self.BR = BR(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.BR(output)
        return output

class C(nn.Module):
    def __init__(self, nIn, nOut, kSize = 3, stride=1,dilated = 1):
        super(C,self).__init__()
        padding = int((kSize - 1)/2)*dilated
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=dilated)

    def forward(self, input):
        output = self.conv(input)
        return output

class CB(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1,dilated = 1):
        super(CB,self).__init__()
        padding = int((kSize - 1)/2)*dilated
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=dilated)
        self.B = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        output = self.conv(input)
        output = self.B(output)
        return output
class upCBR(nn.Module):
    def __init__(self, nIn, nOut,kSize,stride = 2):
        super(upCBR,self).__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.ConvTranspose2d(nIn, nOut, kSize, stride=stride, padding=padding, output_padding=padding, bias=False)
        self.BR = BR(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.BR(output)
        return output

class CrossEntropyLoss2d(nn.Module):
    '''
    This file defines a cross entropy loss for 2D images
    '''
    def __init__(self, weight=None):
        '''
        :param weight: 1D weight vector to deal with the class-imbalance
        '''
        super(CrossEntropyLoss2d,self).__init__()

        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, 1), targets)

class ConvBlock(nn.Module):
    def __init__(self,channel_in,channel_out,kernel,stride,param):
        super(ConvBlock, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel = kernel
        self.stride = stride
        self.in_bottle = param['in_bottle'] if param['in_bottle'] is None else channel_in * param['in_bottle']
        self.out_bottle = param['out_bottle'] if param['out_bottle'] is None else channel_in * param['out_bottle']
        self.in_out_connection = param['in_out_connection']
        assert self.in_out_connection in ['res','dense',None]
        self.main_stream_in = channel_in if self.in_bottle is None else self.in_bottle
        self.main_stream_out = channel_out
        self.main_stream = param['main_stream_fn'](self.main_stream_in,self.main_stream_out,self.kernel,self.stride,
                                                   channel_split = param['channel_split'],spatial_split = param['spatial_split'],
                                                   dilated = param['dilated'])
        #self.se_block = param['se_block']
        if self.in_bottle is not None:
            self.in_bottle_conv = CB(self.channel_in, self.in_bottle,1)
        if self.out_bottle is not None:
            self.out_bottle_conv = CB(self.main_stream_out, self.out_bottle,1)
            self.channel_out = self.out_bottle
        if self.in_out_connection == 'res':
            if not self.out_bottle ==  self.channel_in:
                self.in_out_conv = CB(self.channel_in, self.channel_out, 1)
            else:
                self.in_out_conv = None
        elif self.in_out_connection == 'dense':
            self.channel_out = self.channel_in + self.channel_out
        # if self.se_block:
        #     self.se_block_layer = SELayer(self.channel_out)

        self.out_tensor = {'input':None,
                                'in_bottle': None,
                                'main_stream': None,
                                'out_bottle': None,
                                'out_connection':None,
                                'out_put': None,
                                }
    def forward(self, x):
        self.out_tensor['input'] = x
        if self.in_bottle is not None:
            self.out_tensor['in_bottle'] = self.in_bottle_conv(self.out_tensor['in_bottle'])
        else:
            self.out_tensor['in_bottle'] = self.out_tensor['input']
        #main stream 部分
        self.out_tensor['main_stream'] = self.main_stream(self.out_tensor['in_bottle'])
        #是否需要进行out bottle
        if self.out_bottle is not None:
            self.out_tensor['out_bottle'] = self.out_bottle_conv(self.out_tensor['main_stream'])
        else:
            self.out_tensor['out_bottle'] = self.out_tensor['main_stream']
        pass
        #connection
        if self.in_out_connection == 'res':
            if self.in_out_conv is not None:
                self.out_tensor['out_connection'] = self.in_out_conv(self.out_tensor['out_bottle'])
                self.out_tensor['out_connection'] = self.out_tensor['input']
        elif self.in_out_connection == 'dense':
            self.out_tensor['out_connection'] = torch.cat([self.out_tensor['input'],self.out_tensor['out_bottle']],dim=1)
        else:
            self.out_tensor['out_connection'] = self.out_tensor['out_bottle']
        #是否进行se 12/28：删除se判断，se应该是加入到每一个stage后面，而不是网络结构
        # if self.se_block:
        #     self.out_tensor['out_put'] = self.se_block_layer(self.out_tensor['out_bottle'])
        # else:
        #     self.out_tensor['out_put'] = self.out_tensor['out_bottle']
        return self.out_tensor['out_put']

class main_stream(nn.Module):
    def __init__(self,channel_in,channel_out,kernel,stride,
                 channel_split = 1, spatial_split = False, dilated = 1):
        super(main_stream, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel = kernel
        self.stride = stride
        self.channel_split = channel_split
        self.spatial_split = spatial_split
        self.dilated = dilated
        if type(self.dilated) is int:
            self.dilated = [self.dilated]*self.channel_split
        assert len(self.dilated) == self.channel_split
        if self.channel_split == 1:
            self.channel_split_list = [self.channel_out]
        else:
            n = int(channel_out / self.channel_split)
            n1 = channel_out - self.channel_split * n
            self.channel_split_list = [n1]+[n]*(self.channel_split - 1)

        for i in range(len(self.channel_split_list)):
            name_index = i+1
            split_channel_in = self.channel_in
            split_channel_out = self.channel_split_list[i]
            if self.spatial_split:
                setattr(self,'conv'+str(name_index),
                        spatial_split_conv(split_channel_in,split_channel_out,kernel=kernel,dilated=self.dilated[i]))
            setattr(self,'conv'+str(name_index),
                    C(split_channel_in, split_channel_out, kSize = kernel, stride=self.stride,dilated = self.dilated[i]))
        self.tensors = [None]*len(self.channel_split_list)
        self.bn =nn.BatchNorm2d(channel_out, eps=1e-03)
    def forward(self, x):
        add_tensor = []
        for i in range(len(self.channel_split_list)):
            self.tensors[i] = getattr(self,'conv'+str(i+1))(x)
        for i in range(len(self.channel_split_list)):
            if i == 0 or i == 1:
                add_tensor.append(self.tensors[i])
            else:
                prev_add_tensor = add_tensor[i-1]
                add_tensor.append(prev_add_tensor+self.tensors[i])
        output_tensor = torch.cat(add_tensor,1)
        output_tensor = self.bn(output_tensor)
        return output_tensor

class spatial_split_conv(nn.Module):
    def __init__(self, chann, chann_out, kernel = 3,stride = 1,dilated = 1):
        super(spatial_split_conv, self).__init__()
        padding = int((kernel - 1) / 2) * dilated
        self.conv3x1 = nn.Conv2d(chann, chann, (kernel, 1), stride=stride, padding=(1 * padding, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3 = nn.Conv2d(chann, chann_out, (1, kernel), stride=stride, padding=(0, 1 * padding), bias=True,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)


    def forward(self, input):

        output = self.conv3x1_2(input)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)


        return F.relu(output + input)

class Basic_Layer(nn.Module):
    def __init__(self):
        super(Basic_Layer, self).__init__()
        self.input_channel = None
        self.output_channel = None
        self.operation = None
        self.in_bottle = False
        self.out_bottle = False
        self.in_out_connection = None
        self.main_stream = None

class Downsample_block(Basic_Layer):
    def __init__(self,in_channel,out_channel,in_bottle=0,out_bottle=0,in_out_connection = None):
        super(Downsample_block, self).__init__()
        self.input_channel = in_channel
        self.output_channel = out_channel
        self.in_bottle = in_bottle
        self.out_bottle = out_bottle
        self.in_out_connection = in_out_connection

#get function
def get_down_sample_conv_block(structure_param,block_param):
    return ConvBlock(structure_param[0],structure_param[1],structure_param[2],structure_param[3],
                     block_param)
def get_regular_conv_block(structure_param,block_param):
    module_list = nn.ModuleList()
    if type(structure_param[0]) is int:
        return None
    else:
        assert len(structure_param) == len(block_param)
        for index,structure in enumerate(structure_param):
            module_list.append(ConvBlock(structure[0],structure[1],structure[2],structure[3],
                     block_param[index]))
        return ConvBlock

def get_upbr(structure_param,block_param):
    return upCBR(structure_param[0], structure_param[1], structure_param[2], structure_param[3])

def get_basic_cbr(structure):
    return CBR(structure['channel_in'],structure['channel_out'],structure['kernel_size'],structure['stride'])

def get_basic_upbr(structure):
    return upCBR(structure['channel_in'],structure['channel_out'],structure['kernel_size'],structure['stride'])

def get_basic_cbr_list(structure):
    module_list = nn.ModuleList()
    if structure['block_num'] == 0:
        return None
    else:
        for i in range(structure['block_num']):
            module_list.append(CBR(structure['channel_in'],structure['channel_out'],structure['kernel_size'],structure['stride']))
    return module_list


def get_Downsample_block(downsample_block_config):
    return CBR(downsample_block_config.channel_in,downsample_block_config.channel_out,
               downsample_block_config.kernel_size,downsample_block_config.stride)

def get_Upsample_block(upsample_block_config):
    return upCBR(upsample_block_config.channel_in,upsample_block_config.channel_out,
                 upsample_block_config.kernel_size,upsample_block_config.stride)


def get_Regular_block(regular_block_config):
    regular_block = nn.ModuleList()
    if regular_block_config.block_num == 0:
        return None
    else:
        for i in range(regular_block_config.block_num):
            regular_block.append(CBR(regular_block_config.channel_in[i],regular_block_config.channel_out[i],
                                     regular_block_config.kernel_size,regular_block_config.stride))
        return regular_block


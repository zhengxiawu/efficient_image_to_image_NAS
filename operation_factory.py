import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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


'''
cell config
the config is used for to generate the Cell and record the operation
'''
# class basic_cell_config():
#     def __init__(self,channel_in,channel_out,kernel_size,stride):
#         self.channel_in = channel_in
#         self.channel_out = channel_out
#         self.stride = stride
#         self.kernel_size = kernel_size test222
from operation_factory import *
OUT_TENSOR_NAME = ['input',
                    'down_sample_1', 'regular_1',
                    'down_sample_2', 'regular_2',
                    'down_sample_3', 'regular_3',
                    'up_sample_4', 'regular_4',
                    'up_sample_5', 'regular_5',
                    'up_sample_6']
LAYER_NAME = OUT_TENSOR_NAME[1:]
DEFAULT_STRUCTURE_PARAM = {'down_sample_1': {'channel_in': 3, 'channel_out': 16, 'kernel_size': 3, 'stride': 2},
                           'regular_1': {'channel_in': 16, 'channel_out': 16, 'kernel_size': 3, 'stride': 1,'block_num':0},
                           'down_sample_2': {'channel_in': 16, 'channel_out': 64, 'kernel_size': 3, 'stride': 2},
                           'regular_2': {'channel_in': 64, 'channel_out': 64, 'kernel_size': 3, 'stride': 1,'block_num':0},
                           'down_sample_3': {'channel_in': 64, 'channel_out': 128, 'kernel_size': 3, 'stride': 2},
                           'regular_3': {'channel_in': 128, 'channel_out': 128, 'kernel_size': 3, 'stride': 1,'block_num':0},
                           'up_sample_4': {'channel_in': 128, 'channel_out': 64, 'kernel_size': 3, 'stride': 2},
                           'regular_4': {'channel_in': 64, 'channel_out': 64, 'kernel_size': 3, 'stride': 1,'block_num':0},
                           'up_sample_5': {'channel_in': 64, 'channel_out': 32, 'kernel_size': 3, 'stride': 2},
                           'regular_5': {'channel_in': 32, 'channel_out': 32, 'kernel_size': 3, 'stride': 1,'block_num':0},
                           'up_sample_6':{'channel_in': 32, 'channel_out': 16, 'kernel_size': 3, 'stride': 2}
                           }
DEFAULT_MODEL_FN = {'down_sample':get_basic_cbr,'regular':get_basic_cbr_list,'up_sample':get_basic_upbr}

class Downsample_config():
    def __init__(self,channel_in,channel_out,kernel_size,stride):
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.kernel_size = kernel_size
        #self.spatial_size = spatial_size

class Upsample_config():
    def __init__(self,channel_in,channel_out,kernel_size,stride):
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.kernel_size = kernel_size
        #self.spatial_size = spatial_size

class Regular_config():
    def __init__(self,channel_in,channel_out,kernel_size,stride,block_num,):
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.kernel_size = kernel_size
        self.block_num = block_num
        #self.spatial_size = spatial_size

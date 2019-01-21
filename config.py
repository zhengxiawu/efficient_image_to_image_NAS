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
DEFAULT_DOWN_SAMPLE_PARAM = {'in_bottle':None,'out_bottle':None,'in_out_connection':None,
                            'channel_split':1,'spatial_split':False,'dilated':1,
                             'main_stream_fn':main_stream,}
DEFAULT_REGULAR_PARAM = {'in_bottle':None,'out_bottle':None,'in_out_connection':None,
                            'channel_split':1,'spatial_split':False,'dilated':1,
                             'main_stream_fn':main_stream,}
DEFAULT_UP_SAMPLE_PARAM = {None}
DEFAULT_BLOCK_PARAM = {'down_sample':DEFAULT_DOWN_SAMPLE_PARAM,
                       'regular':DEFAULT_REGULAR_PARAM,
                       'up_sample':DEFAULT_UP_SAMPLE_PARAM
                       }
DEFAULT_STRUCTURE_PARAM = {'down_sample_1': [3,16,3,2],
                           'regular_1': [16,16,3,1,0],
                           'down_sample_2': [16,64,3,2],
                           'regular_2': [64,64,3,1,0],
                           'down_sample_3': [64,128,3,2],
                           'regular_3': [128,128,3,1,0],
                           'up_sample_4': [128,64,3,2],
                           'regular_4': [64,64,3,1,0],
                           'up_sample_5': [64,32,3,2],
                           'regular_5': [32,32,3,1,0],
                           'up_sample_6':[32,16,3,2],
                           }

DEFAULT_MODEL_FN = {'down_sample':get_down_sample_conv_block,
                    'regular':get_regular_conv_block,
                    'up_sample':get_upbr}


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

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

from operation_factory import *
from config import OUT_TENSOR_NAME,LAYER_NAME
import time
up_layers = ['up_sample_4','regular_4','up_sample_5','regular_5','up_sample_6']

class Model(nn.Module):
    def __init__(self,num_classes, decoder, model_fn, structure_param,block_param):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.decoder = decoder
        self.model_fn = model_fn
        self.structure_param = structure_param
        self.structure_param['up_sample_5'][1] = 2* num_classes
        self.structure_param['up_sample_6'][1] = num_classes
        self.structure_param['up_sample_6'][0] = 2* num_classes
        if type(self.structure_param['regular_5'][0]) is not int:
            self.structure_param['regular_5'][0][0] = 2 *num_classes
            self.structure_param['regular_5'][0][1] = 2 * num_classes

        self.connection_information = {'start_end_node': [],
                                       'connection_mode': [],
                                       'start_operation_list': [], }
        self.action_list = {'stage_1': [],
                            'stage_2': [],
                            'stage_3': [], }

        self.layer_name = LAYER_NAME
        self.out_tensor_name = OUT_TENSOR_NAME
        self.out_tensor_size = [[1, 3],
                                [2, 16], [2, 16],
                                [4, 64], [4, 64],
                                [8, 128], [8, 128],
                                [4, 64], [4, 64],
                                [2, 2 * num_classes], [2, 2 * num_classes],
                                [1, num_classes]]
        self.out_tensor_dict = dict(zip(self.out_tensor_name,[None]*len(self.out_tensor_name)))
        self.out_tensor_size = dict(zip(self.out_tensor_name, self.out_tensor_size))

        for layer in self.layer_name:
            if self.decoder:
                setattr(self,layer,model_fn[layer[0:-2]](structure_param[layer],block_param[layer[0:-2]]))
            else:
                if layer in up_layers:
                    setattr(self, layer, None)
                else:
                    setattr(self, layer, model_fn[layer[0:-2]](structure_param[layer], block_param[layer[0:-2]]))

        if not self.decoder:
            self.project_layer = nn.Conv2d(128, num_classes, kernel_size=1)
        self.weights_init()

    def connection(self, start_node, end_node, start_tensor, end_tensor):
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

    def weights_init(self):
        for idx, m in enumerate(self.modules()):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 and not classname == 'ConvBlock':
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        self.out_tensor_dict['input'] = x
        for tensor_name in self.out_tensor_name[1:]:
            operation = getattr(self,tensor_name)
            tensor_index = self.out_tensor_name.index(tensor_name)
            if operation is not None:
                if isinstance(operation,nn.ModuleList):
                    for index,sub_operation in enumerate(operation):
                        if index == 0:
                            self.out_tensor_dict[tensor_name] = \
                                sub_operation(self.out_tensor_dict[self.out_tensor_name[tensor_index - 1]])
                        else:
                            self.out_tensor_dict[tensor_name] = \
                                sub_operation(self.out_tensor_dict[tensor_name])
                else:
                    self.out_tensor_dict[tensor_name] = \
                        operation(self.out_tensor_dict[self.out_tensor_name[tensor_index-1]])

                for i in range(tensor_index):
                    start_tensor_name = self.out_tensor_name[i]
                    if [start_tensor_name, tensor_name] in self.connection_information['start_end_node']:
                        self.out_tensor_dict[tensor_name] = \
                            self.connection(start_tensor_name, tensor_name,
                                            self.out_tensor_dict[start_tensor_name], self.out_tensor_dict[tensor_name])
            else:
                self.out_tensor_dict[tensor_name] = self.out_tensor_dict[self.out_tensor_name[tensor_index-1]]
            #print tensor_name + ":" + str(time.time()-start_time)
        if not self.decoder:
            self.out_tensor_dict['up_sample_6'] = self.project_layer(self.out_tensor_dict['up_sample_6'])
            self.out_tensor_dict['up_sample_6'] = F.interpolate \
                (self.out_tensor_dict['up_sample_6'], scale_factor=8, mode='bilinear', align_corners=True)
        return self.out_tensor_dict['up_sample_6']
def create_default_model(num_classes,decoder):
    from config import DEFAULT_STRUCTURE_PARAM,DEFAULT_MODEL_FN,DEFAULT_BLOCK_PARAM
    return Model(num_classes, decoder, DEFAULT_MODEL_FN, DEFAULT_STRUCTURE_PARAM, DEFAULT_BLOCK_PARAM)
if __name__ == '__main__':
    from config import DEFAULT_STRUCTURE_PARAM,DEFAULT_MODEL_FN,DEFAULT_BLOCK_PARAM
    from utils import *
    structure_test = {'down_sample_1': [3, 16, 3, 2],
                               'regular_1': [[16, 16, 3, 1]],
                               'down_sample_2': [16, 64, 3, 2],
                               'regular_2': [[64, 64, 3, 1]],
                               'down_sample_3': [64, 128, 3, 2],
                               'regular_3': [[128, 128, 3, 1]],
                               'up_sample_4': [128, 64, 3, 2],
                               'regular_4': [[64, 64, 3, 1]],
                               'up_sample_5': [64, 32, 3, 2],
                               'regular_5': [[32, 32, 3, 1]],
                               'up_sample_6': [32, 16, 3, 2],
                               }
    num_classes = 20
    decoder = True
    model_test = Model(num_classes,decoder,DEFAULT_MODEL_FN,structure_test,DEFAULT_BLOCK_PARAM)
    input = Variable(torch.randn(1, 3, 512, 1024))
    DEFAULT_REGULAR_PARAM = {'in_bottle': None, 'out_bottle': None, 'in_out_connection': None,
                             'channel_split': 1, 'spatial_split': False, 'dilated': 1, 'se_block': True,
                             'main_stream_fn': main_stream, }
    #model_test.cuda()
    #input = input.cuda()
    output = model_test(input)
    print(netParams(model_test))
    print(output.size())
    pass
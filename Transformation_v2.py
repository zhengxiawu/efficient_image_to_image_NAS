from torchsummary import summary
from operation_factory import *
from config import DEFAULT_REGULAR_PARAM,DEFAULT_MODEL_FN,DEFAULT_DOWN_SAMPLE_PARAM,DEFAULT_UP_SAMPLE_PARAM
import copy
import tensorly
tensorly.set_backend('pytorch')
up_layers = ['up_sample_4','regular_4','up_sample_5','regular_5','up_sample_6']

# stage1 add regular block
def add_regular_block(block_name,input_model):
    assert hasattr(input_model,block_name), "Must contain the block"
    if not input_model.decoder:
        assert not block_name in up_layers, "The decoder has been deleted! cannot add the block in decoder!"
    transform_model = input_model
    if type(transform_model.structure_param[block_name][0]) is int:
        transform_model.structure_param[block_name] = [transform_model.structure_param[block_name][0:-1]]
        setattr(transform_model,block_name,DEFAULT_MODEL_FN['regular'](transform_model.structure_param[block_name],
                                                                       DEFAULT_REGULAR_PARAM))
        return transform_model
    else:
        transform_model.structure_param[block_name].append(transform_model.structure_param[block_name][0])
        transform_block = getattr(transform_model,block_name)
        transform_block.append(transform_block[len(transform_block)-1])
        setattr(transform_model,block_name,transform_block)
        return transform_model
        pass
    pass


#stage add connection
def add_connection(input_model, start_node, end_node,
                   connection_mode='concat',
                   spatial_operation = None,
                   channel_operation = None,
                   channel_dim = None):
    assert not getattr(input_model,end_node) == None,"the layer should not be none!"
    start_node_index = input_model.out_tensor_name.index(start_node)
    start_node_size = input_model.out_tensor_size[start_node]
    end_node_index = input_model.out_tensor_name.index(end_node)
    end_node_size = input_model.out_tensor_size[end_node]
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
        #transformed_size == end_node_size[1]
        assert transformed_size == end_node_size[1],"in add mode, size should be same!"
    else:
        transformed_size = end_node_size[1] + transformed_size
        input_model.out_tensor_size[end_node][1] = transformed_size
        #change the operation and weight in latter layer
        successor_layer_index = end_node_index
        successor_layer_name = input_model.layer_name[successor_layer_index]
        successor_layer = getattr(input_model,successor_layer_name)
        while successor_layer is None:
            successor_layer_index += 1
            successor_layer_name = input_model.layer_name[successor_layer_index]
            successor_layer = getattr(input_model, successor_layer_name)
        successor_layer_config = input_model.structure_param[successor_layer_name]

        if 'regular' in successor_layer_name:

            successor_layer_config = successor_layer_config[0]
            old_channel_in = successor_layer_config[0]
            channel_in = transformed_size
            channel_out = successor_layer_config[1]
            kernel_size = successor_layer_config[2]
            stride = successor_layer_config[3]
            first_successor_layer_ = get_regular_conv_block([[channel_in, channel_out, kernel_size, stride]],
                                                                DEFAULT_REGULAR_PARAM)
            first_successor_layer_ = first_successor_layer_[0]
            first_successor_layer_.main_stream.conv1 = weights_init(first_successor_layer_.main_stream.conv1)
            first_successor_layer_.main_stream.conv1.conv.weight.data[:, 0:old_channel_in, :, :] = \
                successor_layer[0].main_stream.conv1.conv.weight.data
            successor_layer[0] = first_successor_layer_
            setattr(input_model, successor_layer_name,successor_layer)
            input_model.structure_param[successor_layer_name][0][0] = transformed_size
        else:
            successor_layer = getattr(input_model, successor_layer_name)
            old_channel_in = successor_layer_config[0]
            channel_in = transformed_size
            channel_out = successor_layer_config[1]
            kernel_size = successor_layer_config[2]
            stride = successor_layer_config[3]
            if 'down_sample' in successor_layer_name:
                first_successor_layer_ = get_down_sample_conv_block([channel_in,channel_out,kernel_size,stride],
                                                                    DEFAULT_DOWN_SAMPLE_PARAM)
                first_successor_layer_.main_stream.conv1 = weights_init(first_successor_layer_.main_stream.conv1)
                first_successor_layer_.main_stream.conv1.conv.weight.data[:, 0:old_channel_in,:,:] = \
                    successor_layer.main_stream.conv1.conv.weight.data
            else:
                first_successor_layer_ = get_upbr([channel_in,channel_out,kernel_size,stride],None)
                first_successor_layer_ = weights_init(first_successor_layer_)
                first_successor_layer_.conv.weight.data[0:old_channel_in,: , :, :] = \
                    successor_layer.conv.weight.data
            setattr(input_model, successor_layer_name, first_successor_layer_)
            input_model.structure_param[successor_layer_name][0] = transformed_size

    operation_list = weights_init(operation_list)
    input_model.connection_information['start_end_node'].append([start_node,end_node])
    input_model.connection_information['connection_mode'].append(connection_mode)
    input_model.connection_information['start_operation_list'].append(operation_list)
    input_model.action_list['stage_2'].append('addConnection_%s_%s_%s_%s_%s'%
                                                     (start_node, end_node, connection_mode,
                                                      str(spatial_operation), str(channel_operation)))
    return input_model

# stage 3 in_bottle out_bottle
def in_bottle_layer_transformation(new_layer,old_layer, in_bottle_ratio):
    new_layer.in_bottle = int(new_layer.channel_in * in_bottle_ratio)
    new_layer.in_bottle_conv = CB(new_layer.channel_in, int(new_layer.in_bottle), 1)
    new_layer.main_stream_in = new_layer.in_bottle
    assert new_layer.main_stream.channel_split == 1, "the channel split should be done after in bottle"
    assert not new_layer.main_stream.spatial_split, "the spatial split should be done after in bottle"
    new_main_stream = main_stream(new_layer.main_stream_in, new_layer.main_stream_out, new_layer.kernel,
                                  new_layer.stride,
                                  channel_split=new_layer.main_stream.channel_split,
                                  spatial_split=new_layer.main_stream.spatial_split,
                                  dilated=new_layer.main_stream.dilated)
    old_weight = old_layer.main_stream.conv1.conv.weight.data
    n, c, h, w = old_weight.size()
    temp_weight_tensor = torch.transpose(old_weight, 0, 1)  # CxNxHxW
    temp_weight_tensor = temp_weight_tensor.reshape(c, n * h * w)
    U, S, V = tensorly.partial_svd(temp_weight_tensor, new_layer.in_bottle)
    S = torch.diag(S)
    S = torch.sqrt(S)
    U = torch.mm(U, S)  # C x R
    V = torch.mm(S, V)  # R x (N x H x W)
    U = U.t().reshape(new_layer.in_bottle, c, 1, 1)  # transpose and reshape R x C x 1 x 1
    V = V.reshape(new_layer.in_bottle, n, h, w)  # R x N x H x W
    V = torch.transpose(V, 0, 1)
    new_layer.in_bottle_conv.conv.weight.data = U
    new_main_stream.conv1.conv.weight.data = V
    new_layer.main_stream = new_main_stream
    return new_layer
def in_bottle(input_model,in_bottle_ratio, block_name):
    for name in block_name:
        this_layer = getattr(input_model, name)
        if this_layer is not None:
            if 'regular' in name:
                temp_ = nn.ModuleList()
                for sub_layer_index,sub_layer in enumerate(this_layer):
                    deep_copy_sub_layer = copy.deepcopy(sub_layer)
                    temp_.append(in_bottle_layer_transformation(deep_copy_sub_layer,sub_layer,in_bottle_ratio))
            else:
                deep_copy_layer = copy.deepcopy(this_layer)
                temp_ = in_bottle_layer_transformation(deep_copy_layer,this_layer,in_bottle_ratio)
            setattr(input_model, name, temp_)
    return input_model


def out_bottle_layer_transformation(new_layer,old_layer, out_bottle_ratio):
    new_layer.main_stream_out = int(new_layer.main_stream_out * out_bottle_ratio)
    r = new_layer.main_stream_out
    new_layer.out_bottle_conv = CB(new_layer.main_stream_out, new_layer.channel_out, 1)
    new_layer.out_bottle = new_layer.channel_out
    assert new_layer.main_stream.channel_split == 1, "the channel split should be done after in bottle"
    assert not new_layer.main_stream.spatial_split, "the spatial split should be done after in bottle"
    new_main_stream = main_stream(new_layer.main_stream_in, new_layer.main_stream_out, new_layer.kernel,
                                  new_layer.stride,
                                  channel_split=new_layer.main_stream.channel_split,
                                  spatial_split=new_layer.main_stream.spatial_split,
                                  dilated=new_layer.main_stream.dilated)
    old_weight = old_layer.main_stream.conv1.conv.weight.data
    n, c, h, w = old_weight.size()
    temp_weight_tensor = old_weight.reshape(n,c*h*w) # n * (c*h*w)
    U, S, V = tensorly.partial_svd(temp_weight_tensor, new_layer.main_stream_out)
    S = torch.diag(S)
    S = torch.sqrt(S)
    U = torch.mm(U, S)  # n x r
    V = torch.mm(S, V)  # R x (c x H x W)
    U = U.t().reshape(n, r, 1, 1)  # transpose and reshape n x r x 1 x 1
    V = V.reshape(r, c, h, w)  # R x N x H x W
    V = torch.transpose(V, 0, 1)
    new_layer.out_bottle_conv.conv.weight.data = U
    new_main_stream.conv1.conv.weight.data = V
    new_layer.main_stream = new_main_stream
    return new_layer
def out_bottle(input_model,out_bottle_ratio, block_name):
    for name in block_name:
        this_layer = getattr(input_model, name)
        if this_layer is not None:
            if 'regular' in name:
                temp_ = nn.ModuleList()
                for sub_layer_index,sub_layer in enumerate(this_layer):
                    deep_copy_sub_layer = copy.deepcopy(sub_layer)
                    temp_.append(out_bottle_layer_transformation(deep_copy_sub_layer,sub_layer,out_bottle_ratio))
            else:
                deep_copy_layer = copy.deepcopy(this_layer)
                temp_ = out_bottle_layer_transformation(deep_copy_layer,this_layer,out_bottle_ratio)
            setattr(input_model, name, temp_)
    return input_model
# torch summary with deep copy
def get_torch_summary(input_model,size,device):
    summary_model = copy.deepcopy(input_model)
    summary(summary_model,size,device = device)
if __name__ == '__main__':
    from Model_v2 import create_default_model
    from utils import *

    num_class = 20
    decoder = True
    test_model = create_default_model(num_class,decoder)
    #input = Variable(torch.randn(1, 3, 512, 1024))
    test_model = add_regular_block('regular_1', test_model)
    test_model = add_regular_block('regular_1', test_model)
    test_model = add_regular_block('regular_1', test_model)
    test_model = add_regular_block('regular_1', test_model)
    test_model = add_regular_block('regular_1', test_model)
    test_model = add_regular_block('regular_1', test_model)
    test_model = add_regular_block('regular_1', test_model)
    test_model = add_regular_block('regular_1', test_model)

    # copy the model in summary
    get_torch_summary(test_model, (3, 512, 1024), "cpu")
    test_model = in_bottle(test_model, 0.5, ['regular_1'])
    test_model = out_bottle(test_model, 0.5, ['regular_1'])
    get_torch_summary(test_model, (3, 512, 1024), "cpu")
    pass
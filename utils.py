
def netParams(model):
    '''
    helper function to see total network parameters
    :param model: model
    :return: total network parameters
    '''
    total_paramters = 0
    test = model.parameters()
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters

def weights_init(model):
    for idx, m in enumerate(model.modules()):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and not classname == 'ConvBlock':
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    return model
#def channel_operation(operation_name,)



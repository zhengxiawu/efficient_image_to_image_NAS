from operation_factory import *
from config import OUT_TENSOR_NAME,LAYER_NAME
class Model(nn.Module):
    def __init__(self,num_classes, decoder, model_fn, structure_param):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.decoder = decoder
        self.model_fn = model_fn

        self.layer_name = LAYER_NAME
        self.out_tensor_name = OUT_TENSOR_NAME
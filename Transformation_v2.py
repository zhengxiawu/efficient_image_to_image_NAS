from utils import weights_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import *
from operation_factory import *
import copy

def add_regular_block(block_name,input_model):
    assert hasattr(input_model,block_name), "Must contain the block"
    transform_model = copy.deepcopy(input_model)
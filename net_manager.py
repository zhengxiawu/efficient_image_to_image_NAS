# -*- coding: utf-8 -*-
'''
主要用来进行网络训练，输入数据集，网络，以及一些超级参数，进行网络的训练更新
'''
import os
import json
import torch
import time
import pickle
import data_loader.DataSet as  myDataLoader
import data_loader.Transforms as myTransforms
from data_loader import loadData
from myIOUEval import iouEval
from model import *
from utils import *
from torch.autograd import Variable
import torch.backends.cudnn as cudnn


def val(classes, val_loader, model, criterion,ignore_label):
    '''
    :param args: general arguments
    :param val_loader: loaded for validation dataset
    :param model: model
    :param criterion: loss function
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    #switch to evaluation mode
    model.eval()
    epoch_loss = []
    total_batches = len(val_loader)
    iouEvalVal = iouEval(classes,total_batches,ignore_label)
    for i, (input, target) in enumerate(val_loader):
        start_time = time.time()

        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # run the mdoel
        output = model(input_var)

        # compute the loss
        loss = criterion(output, target_var)

        epoch_loss.append(loss.item())

        time_taken = time.time() - start_time
        # compute the confusion matrix
        iouEvalVal.addBatch(output.max(1)[1].data, target_var.data)

        print('[%d/%d] loss: %.3f time: %.2f' % (i, total_batches, loss.item(), time_taken))

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)

    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalVal.getMetric()

    return average_epoch_loss_val, overall_acc, per_class_acc, per_class_iu, mIOU

def train(classes, train_loader, model, criterion, optimizer, epoch):
    '''
    :param train_loader: loaded for training dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    # switch to train mode
    model.train()



    epoch_loss = []

    total_batches = len(train_loader)
    iouEvalTrain = iouEval(classes,total_batches)
    for i, (input, target) in enumerate(train_loader):
        start_time = time.time()

        input = input.cuda()
        target = target.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        #run the mdoel
        output = model(input_var)

        #set the grad to zero
        optimizer.zero_grad()
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        time_taken = time.time() - start_time

        #compute the confusion matrix
        iouEvalTrain.addBatch(output.max(1)[1].data, target_var.data)

        print('[%d/%d] loss: %.3f time:%.2f' % (i, total_batches, loss.item(), time_taken))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalTrain.getMetric()

    return average_epoch_loss_train, overall_acc, per_class_acc, per_class_iu, mIOU

def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    '''
    helper function to save the checkpoint
    :param state: model state
    :param filenameCheckpoint: where to save the checkpoint
    :return: nothing
    '''
    torch.save(state, filenameCheckpoint)

def multi_scale_loader(scales,random_crop_size,scale_in, batch_size, data):
    #input check
    assert len(scales) == len(random_crop_size), "the length of scales and random_crop_size should be same!!!"

    #transform
    data_loader = []
    for i,scale in enumerate(scales):
        if random_crop_size[i]>0:
            this_transform = myTransforms.Compose([
                myTransforms.Normalize(mean=data['mean'], std=data['std']),
                myTransforms.Scale(scale[0], scale[1]),
                myTransforms.RandomCropResize(random_crop_size[i]),
                myTransforms.RandomFlip(),
                #myTransforms.RandomCrop(64).
                myTransforms.ToTensor(scale_in),
                #
                ])
        else:
            this_transform = myTransforms.Compose([
                myTransforms.Normalize(mean=data['mean'], std=data['std']),
                myTransforms.Scale(scale[0], scale[1]),
                myTransforms.RandomFlip(),
                # myTransforms.RandomCrop(64).
                myTransforms.ToTensor(scale_in),
                #
            ])
        data_loader.append(
            torch.utils.data.DataLoader(myDataLoader.MyDataset(
                   data['trainIm'],
                   data['trainAnnot'],
                   transform=this_transform,data_name=data['name']),
                   batch_size=batch_size,
                   shuffle=True,
                   num_workers=8,
                   pin_memory=True))
    return data_loader

class NetworkManager:

    def __init__(self,model_path,config_file_name):
        #self.dataset = dataset
        self.mode_path = model_path
        self.config_file_name = config_file_name
        #model_path = '/home/zhengxiawu/work/efficient_image_to_image_NAS'
        # load config
        #config_file = os.path.join(self.model_path, 'config/Basenet_camVid.json')
        self.config_file = os.path.join(self.model_path, self.config_file_name)
        self.config = json.load(open(self.config_file))

        # set file name
        self.data_dir = os.path.join(model_path, self.config['DATA']['data_dir'])
        self.data_cache_file = os.path.join(self.data_dir, self.config['DATA']['cached_data_file'])
        self.save_dir = os.path.join(self.model_path, 'para', self.config['name']) + '/'
        assert not os.path.isfile(os.path.join(self.save_dir, 'best.pth'))

        # data hyper parameters
        self.classes = self.config['DATA']['classes']
        self.width = self.config['DATA']['width']
        self.height = self.config['DATA']['height']
        self.scales = self.config['DATA']['train_args']['scale']
        self.random_crop_size = self.config['DATA']['train_args']['random_crop_size']
        self.scale_in = self.config['DATA']['scale_in']
        self.val_scale = self.config['DATA']['val_args']['scale']
        self.batch_size = self.config['DATA']['train_args']['batch_size']
        self.data_name = self.config['DATA']['name']
        self.ignore_label = self.config['DATA']['ignore_label']

        # network hyper parameters
        self.lr = self.config['lr']
        self.lr_step = self.config['lr_step']
        self.save_step = self.config['save_step']

        # set GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.config['gpu_num'])
        # load the dataset
        if not os.path.isfile(self.data_cache_file):
            dataLoad = loadData.LoadData(self.data_dir, self.config['DATA']['classes'], self.data_cache_file, dataset=self.data_name)
            data = dataLoad.processData()
            if data is None:
                print('Error while pickling data. Please check.')
                exit(-1)
        else:
            data = pickle.load(open(self.data_cache_file, "rb"))

        self.data['name'] = self.data_name

        self.valDataset = myTransforms.Compose([
            myTransforms.Normalize(mean=data['mean'], std=data['std']),
            myTransforms.Scale(self.width, self.height),
            myTransforms.ToTensor(self.scale_in),
            #
        ])
        self.val_data_loader = torch.utils.data.DataLoader(
            myDataLoader.MyDataset(data['valIm'], data['valAnnot'], transform=self.valDataset, data_name=self.data_name),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

        weight = torch.from_numpy(self.data['classWeights'])  # convert the numpy array to torch
        weight = weight.cuda()
        self.criteria = CrossEntropyLoss2d(weight)  # weight
        self.train_data_loaders = multi_scale_loader(self.scales, self.random_crop_size, self.scale_in, self.batch_size, self.data)
        cudnn.benchmark = True
        self.start_epoch = 0
    def get_reward(self, model):
        pass

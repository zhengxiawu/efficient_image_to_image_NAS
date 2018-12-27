# -*- coding: utf-8 -*-
'''
训练整个NAS的程序
'''
import net_manager
import controller
import model

if __name__ == '__main__':
    ACTION_LIST = {
        'stage_1': ['delete_regular_block', 'insert_regular_block', 'delete_decoder'],
        'stage_2': ['add_connection'],
        'stage_3': []
    }

    STAGE_1_OPERATION_OBJECT = ['regular_1','regular_2','regular_3','regular_4','regular_5',]


    #get network manager
    network_manager = net_manager.NetworkManager('/home/zhengxiawu/work/efficient_image_to_image_NAS',
                                                 'config/Basenet_camVid.json')

    # controller
    controller = controller.RandomController(ACTION_LIST)
    #get model
    #初始化默认模型，3->16(/2)->64(/4)->64(regular * 5)->128(/8)->128(regular * 8) -> 64 (/4) -> 64 (regular *2)
    #-> 2*num_classes (/2)
    model = model.Model(num_classes=network_manager.classes)


    #stage 1 action



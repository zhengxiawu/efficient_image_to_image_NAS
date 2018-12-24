# -*- coding: utf-8 -*-
'''
控制器：主要作用为对整个流程的控制，如何采样，如何根据得到的结构以及结果进行下一次采样
'''

class RandomController(object):
	"""随机控制器，随机选择采样"""
	def __init__(self, action_list):
		super(RandomController, self).__init__()
		self.action_list = action_list


		

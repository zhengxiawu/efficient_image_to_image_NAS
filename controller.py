# -*- coding: utf-8 -*-
'''
控制器：主要作用为对整个流程的控制，如何采样，如何根据得到的结构以及结果进行下一次采样
'''
import random
from Transformation import *
class RandomController(object):
	"""随机控制器，随机选择采样"""
	def __init__(self, action_list):
		super(RandomController, self).__init__()
		self.action_list = action_list

	def stage1_select_action(self, input_model):
		regular_name_list = ['regular_1', 'regular_2', 'regular_3', 'regular_4', 'regular_5']
		stage_action_list = self.action_list['stage_1']
		action = random.choice(stage_action_list)
		if action == 'delete_decoder' and input_model.decoder:
			return delete_decoder(input_model)
		block = random.choice(regular_name_list)
		if not action == 'delete_decoder':
			while True:
				try:
					new_model = locals()[action](block,input_model)
					break
				except AssertionError:
					block = random.choice(regular_name_list)
					continue
		return new_model

	@staticmethod
	def stage2_select_action(input_model):
		output_name = input_model.output_name
		first_node = random.choice(output_name[0:-2])
		first_node_index = output_name.index(first_node)
		second_node = random.choice(output_name[first_node_index:-1])
		output_channel = input_model.output_channel
		first_node_spatial_size = output_channel[first_node_index][0]
		first_node_channel_size = output_channel[first_node_index][1]
		second_node_spatial_size = output_channel[second_node][0]
		second_node_channel_size = output_channel[second_node][1]
		if  first_node_spatial_size == second_node_spatial_size:
			spatial_action = None
		elif first_node_spatial_size > second_node_spatial_size:
			spatial_action = random.choice(['max','avg','3x3_conv'])
		else:
			spatial_action = 'up_conv'
		connection = random.choice(['add','concat'])
		channel_dim = None
		if connection == 'concat':
			channel_operation = None
		else:
			if first_node_channel_size == second_node_channel_size:
				channel_operation = None
			else:
				channel_operation = '1x1_conv'
				channel_dim = second_node_channel_size
		return add_connection(input_model,first_node,second_node,connection,spatial_action,channel_operation,channel_dim)





	# def select_action(self,stage,model):
	# 	stage_action_list = self.action_list[stage]
	# 	if stage == 'stage_1':
	# 		action = random.choice(stage_action_list)
	#
	# 	pass


		

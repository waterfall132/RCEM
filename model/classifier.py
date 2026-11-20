import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
import utils
import pdb
from abc import abstractmethod
from torch.nn.utils.weight_norm import WeightNorm


class Prototype_Metric(nn.Module):
	def __init__(self, way_num=5, shot_num=5, neighbor_k=3):
		super(Prototype_Metric, self).__init__()
		self.way_num = way_num
		self.avgpool = nn.AdaptiveAvgPool2d(1)


	def cal_EuclideanDis(self, input1, input2):
	
		query = self.avgpool(input1).squeeze(3).squeeze(2)
		query = query.unsqueeze(1)
   

		input2 = self.avgpool(input2).squeeze(3).squeeze(2)
		support_set = input2.contiguous().view(self.way_num, -1, input2.size(1))
		support_set = torch.mean(support_set, 1)


		proto_dis = -torch.pow(query-support_set, 2).sum(2)
		

		return proto_dis


	def forward(self, x1, x2):

		proto_dis = self.cal_EuclideanDis(x1, x2)

		return proto_dis


class ImgtoClass_Metric(nn.Module):
	def __init__(self, way_num=5, shot_num=5, neighbor_k=3):
		super(ImgtoClass_Metric, self).__init__()
		self.neighbor_k = neighbor_k
		self.shot_num = shot_num


	def cal_cosinesimilarity(self, input1, input2):

		input1 = input1.contiguous().view(input1.size(0), input1.size(1), -1)
		input1 = input1.permute(0, 2, 1)

		
		input2 = input2.contiguous().view(input2.size(0), input2.size(1), -1)
		input2 = input2.permute(0, 2, 1)


		input1_norm = torch.norm(input1, 2, 2, True)
		query = input1/input1_norm
		query = query.unsqueeze(1)


		input2_norm = torch.norm(input2, 2, 2, True)
		support_set = input2/input2_norm
		support_set = support_set.contiguous().view(-1,
				self.shot_num*support_set.size(1), support_set.size(2))
		support_set = support_set.permute(0, 2, 1)


		innerproduct_matrix = torch.matmul(query, support_set)


		topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 3)
		img2class_sim = torch.sum(torch.sum(topk_value, 3), 2)


		return img2class_sim


	def forward(self, x1, x2):

		img2class_sim = self.cal_cosinesimilarity(x1, x2)

		return img2class_sim

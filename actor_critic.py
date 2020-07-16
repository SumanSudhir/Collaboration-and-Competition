import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):

	def __init__(self, state_size, action_size, hid1_nodes=256, hid2_nodes=128, p1=0.1, p2=0.2):
		"""
		state_size (int): Dimension of each state
		action_size (int): Dimension of each action
		hid1_nodes (int): Number of nodes in first hidden layer
		hid2_nodes (int): Number of nodes in second hidden layer
		p1 (int) : dropout probability for first hidden layer
		p2 (int) : dropout probability for second hidden layer
		"""

		#Dense Layer
		super(Actor, self).__init__()
		self.fc1 = nn.Linear(state_size*2, hid1_nodes)  # ()*2 as each agent takes state of both agents as input in MADDGP
		self.fc2 = nn.Linear(hid1_nodes, hid2_nodes)
		self.fc3 = nn.Linear(hid2_nodes, action_size)
		# self.fc3.weight.data.uniform_(-3e-3, 3e-3)

		#Normalization
		self.bn1 = nn.BatchNorm1d(hid1_nodes)
		self.bn2 = nn.BatchNorm1d(hid2_nodes)

		#Dropout
		self.drop_layer1 = nn.Dropout(p=p1)
		self.drop_layer2 = nn.Dropout(p=p2)

		self.reset_parameter()

	def reset_parameter(self):
	    self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
	    self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
	    self.fc3.weight.data.uniform_(-3e-3, 3e-3)

	def forward(self, state):
		"""Mapping of states to actions"""
		temp = F.relu(self.bn1(self.fc1(state)))
# 		temp = F.relu(self.fc1(state))
# 		temp = self.drop_layer1(temp)
		temp = F.relu(self.bn2(self.fc2(temp)))
# 		temp = F.relu(self.fc2(temp))
# 		temp = self.drop_layer2(temp)
		return F.tanh(self.fc3(temp))

class Critic(nn.Module):

	def __init__(self, state_size, action_size, hid1_nodes=256, hid2_nodes=128, p1=0.1, p2=0.2):
		"""
		state_size (int): Dimension of each state
		action_size (int): Dimension of each action
		hid1_nodes (int): Number of nodes in first hidden layer
		hid2_nodes (int): Number of nodes in second hidden layer
		p1 (int) : dropout probability for first hidden layer
		p2 (int) : dropout probability for second hidden layer
		"""

		#Dense Layer
		super(Critic, self).__init__()
		self.fc1 = nn.Linear(state_size*2, hid1_nodes) # ()*2 as each agent takes state of both agents as input in MADDGP
		self.fc2 = nn.Linear(hid1_nodes + (action_size*2), hid2_nodes)
		self.fc3 = nn.Linear(hid2_nodes, 1)

# 		#Normalization
		self.bn1 = nn.BatchNorm1d(hid1_nodes)
		self.bn2 = nn.BatchNorm1d(hid2_nodes)

		#Dropout
		self.drop_layer1 = nn.Dropout(p=p1)
		self.drop_layer2 = nn.Dropout(p=p2)

		self.reset_parameter()

	def reset_parameter(self):
	    self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
	    self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
	    self.fc3.weight.data.uniform_(-3e-3, 3e-3)


	def forward(self, state, action):
		"""Mapping of states to actions"""
# 		temp = F.relu(self.bn1(self.fc1(state)))
		temp = F.relu(self.fc1(state))
# 		temp = self.drop_layer1(temp)
# 		temp = F.relu(self.bn2(self.fc2(torch.cat((temp, action), dim=1))))
		temp = F.relu(self.fc2(torch.cat((temp, action), dim=1)))
# 		temp = self.drop_layer2(temp)
		return self.fc3(temp)
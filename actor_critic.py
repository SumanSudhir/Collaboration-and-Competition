import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim  1.0/ np.sqrt(fan_in)
    return (-lim,lim)

class Actor(nn.Module):

    def __init__(self, state_size, action_size, seed, hid1_nodes=256, hid2_nodes=128):
        """
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        seed (int): Random seed
        hid1_nodes (int): Number of nodes in first hidden layer
        hid2_nodes (int): Number of nodes in second hidden layer
        """

        #Dense Layer
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hid1_nodes)
        self.fc2 = nn.Linear(hid1_nodes, hid2_nodes)
        self.fc3 = nn.Linear(hid2_nodes, action_size)

        #Normalization
        #self.bn1 = nn.BatchNorm1d(hid1_nodes)
        #self.bn2 = nn.BatchNorm1d(hid2_nodes)

        self.reset_parameter()

    def reset_parameter(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Mapping of states to actions"""
        temp = F.relu(self.fc1(state))
        #temp = self.bn1(temp)
        temp = F.relu(self.fc2(temp))

        return F.tanh(self.fc3())

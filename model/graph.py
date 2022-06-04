import torch
from torch import Tensor
import torch.nn as nn

class NodeFeaExtractor(nn.Module):
    def __init__(self, hidden_size_st, fc_dim):
        super(NodeFeaExtractor, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 8, 10, stride=1)  
        self.conv2 = torch.nn.Conv1d(8, 16, 10, stride=1) 
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(hidden_size_st)
        self.fc = torch.nn.Linear(fc_dim, hidden_size_st)
       

    def forward(self, node_fea):
        t, n = node_fea.shape
        x = node_fea.transpose(1, 0).reshape(n, 1, -1)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.bn2(x)
        x = x.view(n, -1)
        x = self.fc(x)
        x = torch.relu(x)
        x = self.bn3(x)
        return x

class normal_conv(nn.Module):
    def __init__(self, dg_hidden_size):
        super(normal_conv, self).__init__()
        self.fc1 = nn.Linear(dg_hidden_size, 1)
        self.fc2 = nn.Linear(dg_hidden_size * 2, dg_hidden_size)

    def forward(self, y, shape):
        support = self.fc1(torch.relu(self.fc2(y))).reshape(shape)
        return support


class EvolvingGraphLearner(nn.Module):
    def __init__(self, input_size: int, dg_hidden_size: int):
        super(EvolvingGraphLearner, self).__init__()
        self.rz_gate = nn.Linear(input_size + dg_hidden_size, dg_hidden_size * 2)
        self.dg_hidden_size = dg_hidden_size
        self.h_candidate = nn.Linear(input_size + dg_hidden_size, dg_hidden_size)
        self.conv = normal_conv(dg_hidden_size)
        self.conv2 = normal_conv(dg_hidden_size)
   
    def forward(self, inputs: Tensor, states):
        """
        :param inputs: inputs to cal dynamic relations   [B,N,C]
        :param states: recurrent state [B, N,C]
        :return:  graph[B,N,N]       states[B,N,C]
        """
        b,n,c = states.shape
        r_z = torch.sigmoid(self.rz_gate(torch.cat([inputs, states], -1)))
        r, z = r_z.split(self.dg_hidden_size, -1)
        h_ = torch.tanh(self.h_candidate(torch.cat([inputs, r * states], -1)))
        new_state = z * states + (1 - z) * h_

        dy_sent = torch.unsqueeze(torch.relu(new_state), dim=-2).repeat(1, 1, n, 1)
        dy_revi = dy_sent.transpose(1,2)
        y = torch.cat([dy_sent, dy_revi], dim=-1)
        support = self.conv(y, (b, n, n))      
        mask = self.conv2(y,(b,n,n))
        support = support * torch.sigmoid(mask)
            
        return support, new_state
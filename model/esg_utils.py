import math
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F


class Dilated_Inception(nn.Module):
    def __init__(self, cin, cout, kernel_set, dilation_factor=2):
        super(Dilated_Inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = kernel_set
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):
        #input [B, D, N, T]
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        """
         :param X: tensor, [B, D, N, T]
         :param A: tensor [N, N] , [B, N, N] or [T*, B, N, N]
         :return: tensor [B, D, N, T]        
        """
        #x = torch.einsum('ncwl,vw->ncvl',(x,A))
        if len(A.shape) == 2:
            a_ ='vw'
        elif len(A.shape) == 3:
            a_ ='bvw'
        else:
            a_ = 'tbvw'
        x = torch.einsum(f'bcwt,{a_}->bcvt',(x,A))
        return x.contiguous()
       

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class MixProp(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(MixProp, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
    
    def forward(self,x,adj):
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,adj)
            out.append(h)
        ho = torch.cat(out,dim=1)# [B, D*(1+gdep), N, T]
        ho = self.mlp(ho) #[B, c_out, N, T]
        return ho

   
class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)





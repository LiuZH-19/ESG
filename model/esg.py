import torch
from torch import nn, Tensor
from torch.nn import functional as F


from .esg_utils import Dilated_Inception, MixProp, LayerNorm
from .graph import  NodeFeaExtractor, EvolvingGraphLearner


class TConv(nn.Module):
    def __init__(self, residual_channels: int, conv_channels: int, kernel_set, dilation_factor: int, dropout:float):
        super(TConv, self).__init__()
        self.filter_conv = Dilated_Inception(residual_channels, conv_channels,kernel_set, dilation_factor)
        self.gate_conv = Dilated_Inception(residual_channels, conv_channels, kernel_set, dilation_factor)
        self.dropout = dropout

    def forward(self, x: Tensor):
        _filter = self.filter_conv(x)
        filter = torch.tanh(_filter)
        _gate = self.gate_conv(x)
        gate = torch.sigmoid(_gate)
        x = filter * gate  
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class Evolving_GConv(nn.Module):
    def __init__(self, conv_channels: int, residual_channels: int, gcn_depth: int,  st_embedding_dim: int, 
                dy_embedding_dim: int, dy_interval: int, dropout=0.3, propalpha=0.05):
        super(Evolving_GConv, self).__init__()
        self.linear_s2d = nn.Linear(st_embedding_dim, dy_embedding_dim)
        self.scale_spc_EGL = EvolvingGraphLearner(conv_channels, dy_embedding_dim)
        self.dy_interval = dy_interval         

        self.gconv = MixProp(conv_channels, residual_channels, gcn_depth, dropout, propalpha)

    def forward(self, x, st_node_fea):

        b, _, n, t = x.shape 
        dy_node_fea = self.linear_s2d(st_node_fea).unsqueeze(0)  
        states_dy = dy_node_fea.repeat( b, 1, 1) #[B, N, C]

        x_out = []
    
       
        for i_t in range(0,t,self.dy_interval):     
            x_i =x[...,i_t:min(i_t+self.dy_interval,t)]

            input_state_i = torch.mean(x_i.transpose(1,2),dim=-1)
          
            dy_graph, states_dy= self.scale_spc_EGL(input_state_i, states_dy)
            x_out.append(self.gconv(x_i, dy_graph))
        
        x_out = torch.cat(x_out, dim= -1) #[B, c_out, N, T]
        return x_out

class Extractor(nn.Module):
    def __init__(self, residual_channels: int, conv_channels: int, kernel_set, dilation_factor: int, gcn_depth: int, 
                st_embedding_dim, dy_embedding_dim, 
           skip_channels:int, t_len: int, num_nodes: int, layer_norm_affline, propalpha: float, dropout:float, dy_interval: int):
        super(Extractor, self).__init__()

        self.t_conv = TConv(residual_channels, conv_channels, kernel_set, dilation_factor, dropout)
        self.skip_conv = nn.Conv2d(conv_channels, skip_channels, kernel_size=(1, t_len))
      
        self.s_conv = Evolving_GConv(conv_channels, residual_channels, gcn_depth, st_embedding_dim, dy_embedding_dim, 
                                    dy_interval, dropout, propalpha)

        self.residual_conv = nn.Conv2d(conv_channels, residual_channels, kernel_size=(1, 1))
        
        self.norm = LayerNorm((residual_channels, num_nodes, t_len),elementwise_affine=layer_norm_affline)
       

    def forward(self, x: Tensor,  st_node_fea: Tensor):
        residual = x # [B, F, N, T]
        # dilated convolution
        x = self.t_conv(x)       
        # parametrized skip connection
        skip = self.skip_conv(x)
        #graph convolution
        x = self.s_conv(x,  st_node_fea)         
        #residual connection
        x = x + residual[:, :, :, -x.size(3):]
        x = self.norm(x)
        return x, skip


class Block(nn.ModuleList):
    def __init__(self, block_id: int, total_t_len : int, kernel_set, dilation_exp: int, n_layers: int, residual_channels: int, conv_channels: int,
    gcn_depth: int, st_embedding_dim, dy_embedding_dim,  skip_channels:int, num_nodes: int, layer_norm_affline, propalpha: float, dropout:float, dy_interval: int):
        super(Block, self).__init__()
        kernel_size = kernel_set[-1]
        if dilation_exp > 1:
            rf_block = int(1+ block_id*(kernel_size-1)*(dilation_exp**n_layers-1)/(dilation_exp-1))
        else:
            rf_block = block_id*n_layers*(kernel_size-1) + 1
        
        dilation_factor = 1
        for i in range(1, n_layers+1):            
            if dilation_exp>1:
                rf_size_i = int(rf_block + (kernel_size-1)*(dilation_exp**i-1)/(dilation_exp-1))
            else:
                rf_size_i = rf_block + i*(kernel_size-1)
            t_len_i = total_t_len - rf_size_i +1

            self.append(
                Extractor(residual_channels, conv_channels, kernel_set, dilation_factor, gcn_depth, st_embedding_dim, dy_embedding_dim, 
                 skip_channels, t_len_i, num_nodes, layer_norm_affline, propalpha, dropout, dy_interval[i-1])
            )
            dilation_factor *= dilation_exp



    def forward(self, x: Tensor, st_node_fea: Tensor, skip_list):
        flag = 0
        for layer in self:
            flag +=1
            x, skip = layer(x, st_node_fea)
            skip_list.append(skip)           
        return x, skip_list


class ESG(nn.Module):
    def __init__(self,                 
                 dy_embedding_dim: int,
                 dy_interval: list,
                 num_nodes: int,
                 seq_length: int,
                 pred_len : int,
                 in_dim: int,
                 out_dim: int,
                 n_blocks: int,
                 n_layers: int,                
                 conv_channels: int,
                 residual_channels: int,
                 skip_channels: int,
                 end_channels: int,
                 kernel_set: list,
                 dilation_exp: int,
                 gcn_depth: int,                                
                 device,
                 fc_dim: int,
                 st_embedding_dim=40,
                 static_feat=None,
                 dropout=0.3,
                 propalpha=0.05,
                 layer_norm_affline=True            
                 ):
        super(ESG, self).__init__()
       
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.num_nodes = num_nodes        
        self.device = device
        self.pred_len = pred_len
        self.st_embedding_dim = st_embedding_dim
        self.seq_length = seq_length
        kernel_size = kernel_set[-1]
        if dilation_exp>1:
            self.receptive_field = int(1+n_blocks*(kernel_size-1)*(dilation_exp**n_layers-1)/(dilation_exp-1))
        else:
            self.receptive_field = n_blocks*n_layers*(kernel_size-1) + 1
        self.total_t_len = max(self.receptive_field, self.seq_length)      
      
        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))
        self.blocks = nn.ModuleList()
        for block_id in range(n_blocks):
            self.blocks.append(
                Block(block_id, self.total_t_len, kernel_set, dilation_exp, n_layers, residual_channels, conv_channels, gcn_depth,
                 st_embedding_dim, dy_embedding_dim, skip_channels, num_nodes, layer_norm_affline, propalpha, dropout, dy_interval))

        self.skip0 = nn.Conv2d(in_dim, skip_channels, kernel_size=(1, self.total_t_len), bias=True)
        self.skipE = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, self.total_t_len-self.receptive_field+1), bias=True)
        

        in_channels = skip_channels
        final_channels = pred_len * out_dim

        
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, end_channels, kernel_size=(1,1), bias=True),
            nn.ReLU(),
            nn.Conv2d(end_channels, final_channels, kernel_size=(1,1), bias=True)     
        )
        self.stfea_encode = NodeFeaExtractor(st_embedding_dim, fc_dim)
        self.static_feat = static_feat
       

    def forward(self, input):
        """
        :param input: [B, n_hist, N, in_dim]
        :return: [B, n_pred, N, out_dim]
        """

        b, _, n, t = input.shape
        assert t==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = F.pad(input,(self.receptive_field-self.seq_length,0,0,0), mode='replicate')
        
        x = self.start_conv(input)
           
        st_node_fea = self.stfea_encode(self.static_feat)


        skip_list = [self.skip0(F.dropout(input, self.dropout, training=self.training))]
        for j in range(self.n_blocks):    
            x, skip_list= self.blocks[j](x, st_node_fea , skip_list)
                    
        skip_list.append(self.skipE(x)) 
        skip_list = torch.cat(skip_list, -1)#[B, skip_channels, N, n_layers+2]
       
        skip_sum = torch.sum(skip_list, dim=3, keepdim=True)  #[B, skip_channels, N, 1]
        x = self.out(skip_sum) #[B, pred_len* out_dim, N, 1] 
        x = x.reshape(b, self.pred_len, -1, n).transpose(-1, -2) #[B, pred_len, N, out_dim]
        return x 





        







    
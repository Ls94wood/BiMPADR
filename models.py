import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import math

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class Create(nn.Module):
    def __init__ (self,args):
        super(Create,self).__init__()
        self.ds = Drugin(args)
        self.ss = Sein(args)
        self.cs = Interact(args)

    def forward(self, edge_index, drug_struc, drug_expr, se_struc, pair):
        drug_index, se_index = pair
        drug_features = self.ds(drug_struc)
        se_features = self.ss(edge_index, drug_expr, se_struc)
        adj_matrices = self.cs(drug_features[drug_index,:], se_features[se_index,:])
        return [adj_matrices, se_features]

class Drugin(nn.Module):
    def __init__(self,args):
        super(Drugin,self).__init__()
        self.drug_1 = nn.Sequential(nn.Linear(args.fdrugstr, args.fdrugl1),
                                    nn.BatchNorm1d(args.fdrugl1),
                                    nn.LeakyReLU(),
                                    nn.Dropout(p=args.drop))
        self.drug_2 = nn.Sequential(nn.Linear(args.fdrugl1, args.fdrugl2),
                                    nn.BatchNorm1d(args.fdrugl2),
                                    nn.LeakyReLU(),
                                    nn.Dropout(p=args.drop))
        # self.drug_3 = nn.Sequential(nn.Linear(args.fdrugl1*2, args.fdrugl2),
        #                             nn.BatchNorm1d(args.fdrugl2),
        #                             nn.LeakyReLU(),
        #                             nn.Dropout(p=args.drop))        
    def forward(self, drug_struc):
        f_drug = self.drug_1(drug_struc)
        f_drug = self.drug_2(f_drug)
        # f_drug = self.drug_3(f_drug)
        return f_drug
    
class Sein(nn.Module):
    def __init__(self,args):
        super(Sein,self).__init__()
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.GAT = GATConv2(in_channels = self.in_channels, out_channels = self.out_channels, 
                           add_self_loops=args.loop, concat=False, flow="source_to_target")
        self.sigmoid = nn.Sigmoid()

    def forward(self, edge_index, drug_expr, se_struc):
        f_se = self.GAT(edge_index = edge_index, x = (drug_expr.to(torch.float32), se_struc.to(torch.float32)), size = (drug_expr.shape[0], se_struc.shape[0]))
        for i in range(f_se.shape[0]):
            f_se[i] = self.sigmoid(f_se[i])
        return f_se

class Interact(nn.Module):
    def __init__(self, args):
        super(Interact, self).__init__()
        self.mlp_1 = nn.Sequential(nn.Linear(int(args.fdrugl2+args.fdrugexpr), int(args.fcon)),
                                   nn.BatchNorm1d(int(args.fcon)),
                                   nn.LeakyReLU(),
                                   nn.Dropout(p=args.drop))
        self.mlp_2 = nn.Sequential(nn.Linear(int(args.fcon), int(args.fcon//2)),
                                   nn.BatchNorm1d(int(args.fcon//2)),
                                   nn.LeakyReLU(),
                                   nn.Dropout(p=args.drop))
        self.mlp_3 = nn.Sequential(nn.Linear(int(args.fcon//2), int(args.fcon//4)),
                                   nn.BatchNorm1d(int(args.fcon//4)),
                                   nn.LeakyReLU(),
                                   nn.Dropout(p=args.drop))
        # self.mlp_4 = nn.Sequential(nn.Linear(int(args.fcon//4), int(args.fcon//8)),
        #                            nn.BatchNorm1d(int(args.fcon//8)),
        #                            nn.LeakyReLU(),
        #                            nn.Dropout(p=args.drop))
        self.mlp_5 = nn.Sequential(nn.Linear(int(args.fcon//4), 1),
                                   nn.Sigmoid())

    def forward(self, drug_feature, se_feature):
        pair_feature = torch.cat([drug_feature, se_feature], dim=1)
        embedding_1 = self.mlp_1(pair_feature)
        embedding_2 = self.mlp_2(embedding_1)
        embedding_3 = self.mlp_3(embedding_2)
        # embedding_4 = self.mlp_4(embedding_3)
        outputs = self.mlp_5(embedding_3)
        return outputs
    
class GATConv2(MessagePassing):

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        super(GATConv2, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = alpha_r = (x_l * self.att_l).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None
        
        # if self.add_self_loops:
        #     if isinstance(edge_index, Tensor):
        #         num_nodes = x_l.size(0)
        #         num_nodes = size[1] if size is not None else num_nodes
        #         num_nodes = x_r.size(0) if x_r is not None else num_nodes
        #         edge_index, _ = remove_self_loops(edge_index)
        #         edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        #     elif isinstance(edge_index, SparseTensor):
        #         edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)
        if self.add_self_loops:
            out = out + x_r
        
        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
    
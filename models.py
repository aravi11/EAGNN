from audioop import bias
from re import A
from tkinter import S
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import layers
from layers import MeanAggregator, LSTMAggregator, MaxPoolAggregator, MeanPoolAggregator
from layers import ConvolutionLayer, DGNNConvolutionLayer, AAAgregationLayer, EGNNCLayer, DAAAgregationLayer

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.414)

class GraphSAGE(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim,
                 dropout=0.5, agg_class=MaxPoolAggregator, num_samples=5,
                 device='cpu'):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features.
        hidden_dims : list of ints
            Dimension of hidden layers. Must be non empty.
        output_dim : int
            Dimension of output node features.
        dropout : float
            Probability of setting an element to 0 in dropout layer. Default: 0.5.
        agg_class : An aggregator class.
            Aggregator. One of the aggregator classes imported at the top of
            this module. Default: MaxPoolAggregator.
        num_samples : int
            Number of neighbors to sample while aggregating. Default: 25.
        device : string
            'cpu' or 'cuda:0'. Default: 'cpu'.
        """
        super(GraphSAGE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.agg_class = agg_class
        self.num_samples = num_samples
        self.device = device
        self.num_layers = len(hidden_dims) + 1

        c = 3 if agg_class == LSTMAggregator else 2
        d = 0   # Add distance as feature
        if agg_class == MeanAggregator:
            c = 1
            d = 0

        self.aggregators = nn.ModuleList([agg_class(input_dim+d, input_dim+d, device)])
        self.aggregators.extend([agg_class(dim+d, dim+d, device) for dim in hidden_dims])



        self.fcs = nn.ModuleList([nn.Linear(c*(input_dim)+d, (hidden_dims[0]))])
        self.fcs.extend([nn.Linear(c*(hidden_dims[i-1])+d, (hidden_dims[i])) for i in range(1, len(hidden_dims))])
        self.fcs.extend([nn.Linear(c*(hidden_dims[-1])+d, output_dim)])

        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for hidden_dim in hidden_dims])

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        

    def forward(self, features, node_layers, mappings, rows, dist):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        node_layers : list of numpy array
            node_layers[i] is an array of the nodes in the ith layer of the
            computation graph.
        mappings : list of dictionary
            mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
            in node_layers[i] to its position in node_layers[i]. For example,
            if node_layers[i] = [2,5], then mappings[i][2] = 0 and
            mappings[i][5] = 1.
        rows : numpy array
            rows[i] is an array of neighbors of node i.
        Returns
        -------
        out : torch.Tensor
            An (len(node_layers[-1]) x output_dim) tensor of output node features.
        """
        out = features
        for k in range(self.num_layers):
            nodes = node_layers[k+1]
            mapping = mappings[k]
            init_mapped_nodes = np.array([mappings[0][v] for v in nodes], dtype=np.int64)
            cur_mapped_nodes = np.array([mapping[v] for v in nodes], dtype=np.int64)
            cur_rows = rows[init_mapped_nodes]
            cur_dist = dist[init_mapped_nodes, :]
            cur_dist = dist[:, init_mapped_nodes]
            aggregate = self.aggregators[k](out, nodes, mapping, cur_rows, dist, mappings[0],
                                            self.num_samples[k])
            if self.agg_class != MeanAggregator:
                out = torch.cat((out[cur_mapped_nodes, :], aggregate), dim=1)
            else:
                out = aggregate
            out = self.fcs[k](out)

            if k+1 < self.num_layers:
                out = self.relu(out)
                out = self.bns[k](out)
                out = self.dropout(out)
                out = out.div(out.norm(dim=1, keepdim=True)+1e-6)

        return out


class DGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features.
        hidden_dim : int
            Dimension of hidden layer. Must be non empty.
        output_dim : int
            Dimension of output node features.
        dropout : float
            Probability of setting an element to 0 in dropout layer. Default: 0.5.
        device : string
            'cpu' or 'cuda:0'. Default: 'cpu'.
        """
        super(DGNN, self).__init__()

        self.egc1 = DGNNConvolutionLayer(input_dim, hidden_dim, device=device)
        self.egc2 = DGNNConvolutionLayer(hidden_dim, output_dim, device=device)

        self.dropout = dropout
        self.device = device

    
    def forward(self, features, dist):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        dist : torch.Tensor
            An (n x n) tensor of distance between pairs of neighboring nodes.
        Returns
        -------
        out : torch.Tensor
            An (len(node_layers[-1]) x output_dim) tensor of output node features.
        """

        x = F.relu(self.egc1(features, dist))
        x = F.dropout(x, self.dropout, training=self.training)
        out = self.egc2(x, dist)
        return out


class AAGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features.
        hidden_dim : int
            Dimension of hidden layer. Must be non empty.
        output_dim : int
            Dimension of output node features.
        dropout : float
            Probability of setting an element to 0 in dropout layer. Default: 0.5.
        device : string
            'cpu' or 'cuda:0'. Default: 'cpu'.
        """
        super(AAGNN, self).__init__()

        self.agg1 = AAAgregationLayer(input_dim, input_dim, device=device)
        self.conv1 = DGNNConvolutionLayer(input_dim, hidden_dim, device=device)
        self.conv2 = DGNNConvolutionLayer(hidden_dim + input_dim, output_dim, device=device)

        self.dropout = dropout

        self.device = device

    
    def forward(self, features, dist, adj_relative_cos):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        dist : torch.Tensor
            An (n x n) tensor of distance between pairs of neighboring nodes.
        adj_relative_cos : Dict[int, Dict[tuple, torch.Tensor]]
            adj_relative_cos[i][(j, k)] is the cosine value between a pair of relative vectors node(i -> j) and node(i -> k).
        Returns
        -------
        out : torch.Tensor
            An (len(node_layers[-1]) x output_dim) tensor of output node features.
        """

        # (conv1, agg1) => (conv2)
        x_conv = self.conv1(features, dist)
        x_angle = self.agg1(features, adj_relative_cos)
        x = F.relu(torch.cat((x_conv, x_angle), 1))
        x = F.dropout(x, self.dropout, training=self.training)
        out = self.conv2(x, dist)

        return out


class EGNNC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, channel_dim, dropout=0.5, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features.
        hidden_dim : int
            Dimension of hidden layer. Must be non empty.
        output_dim : int
            Dimension of output node features.
        dropout : float
            Probability of setting an element to 0 in dropout layer. Default: 0.5.
        device : string
            'cpu' or 'cuda:0'. Default: 'cpu'.
        """
        super(EGNNC, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.channel_dim = channel_dim
        self.dropout = dropout
        self.device = device

        self.elu = nn.ELU()
        self.egnn1 = EGNNCLayer(input_dim, hidden_dim, channel_dim, device=device)
        self.egnn2 = EGNNCLayer(hidden_dim*channel_dim, output_dim, channel_dim, device=device)

    def forward(self, features, edge_features):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n x input_dim) tensor of input node features.
        edge_features : torch.Tensor
            An (p x n x n) tensor of edge features.
        Returns
        -------
        out : torch.Tensor
            An (len(node_layers[-1]) x output_dim) tensor of output node features.
        """
        x = self.egnn1(features, edge_features)
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.egnn2(x, edge_features)
        return x


class MLPTwoLayers(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout=0.5, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features.
        hidden_dim : int
            Dimension of hidden layer. Must be non empty.
        output_dim : int
            Dimension of output node features.
        dropout : float
            Probability of setting an element to 0 in dropout layer. Default: 0.5.
        device : string
            'cpu' or 'cuda:0'. Default: 'cpu'.
        """
        super(MLPTwoLayers, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.device = device

        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=True).to(device)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=True).to(device)
        '''
        TO add if you want additional layers in the MLP
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=True).to(device)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=True).to(device)
        self.linear3 = nn.Linear(hidden_dim, output_dim, bias=True).to(device)
        '''

    def forward(self, features):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        Returns
        -------
        out: torch.Tensor
            Output of two layer MLPs
        """

        x = F.relu(self.linear1(features))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear2(x)
        out = x.reshape(-1)
        return out
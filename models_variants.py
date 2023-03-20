from re import A
from tkinter import S
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import layers
from layers import MeanAggregator, LSTMAggregator, MaxPoolAggregator, MeanPoolAggregator
from layers import ConvolutionLayer, DGNNConvolutionLayer, AAAgregationLayer, DAAAgregationLayer



class EAAGNN(nn.Module):
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
        super(EAAGNN, self).__init__()

        self.conv1 = ConvolutionLayer(input_dim, hidden_dim)
        self.agg1 = DAAAgregationLayer(input_dim, input_dim)
        self.conv2 = ConvolutionLayer(hidden_dim + input_dim, output_dim)

        self.dropout = dropout
        self.device = device

    
    def forward(self, features, dist, adj_relative_cos):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        adj: torch.Tensor
            An adjancy matrix of the graph.
        adj_relative_cos : Dict[int, Dict[tuple, torch.Tensor]]
            adj_relative_cos[i][(j, k)] is the cosine value between a pair of relative vectors node(i -> j) and node(i -> k).
        Returns
        -------
        out : torch.Tensor
            An (len(node_layers[-1]) x output_dim) tensor of output node features.
        """

        # (conv1, agg1) => (conv2)
        x_conv = self.conv1(features, dist)
        x_angle = self.agg1(features, dist, adj_relative_cos)
        x = F.relu(torch.cat((x_conv, x_angle), 1))
        x = F.dropout(x, self.dropout, training=self.training)
        out = self.conv2(x, dist)

        return out


class EAACGNN(nn.Module):
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
        super(EAACGNN, self).__init__()

        self.conv1 = ConvolutionLayer(input_dim, hidden_dim)
        self.agg1 = DAAAgregationLayer(input_dim, input_dim)
        self.conv2 = ConvolutionLayer(hidden_dim + input_dim, output_dim)

        self.dropout = dropout
        self.device = device

    
    def forward(self, features, adj, dist, adj_relative_cos):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        adj: torch.Tensor
            An adjancy matrix of the graph.
        adj_relative_cos : Dict[int, Dict[tuple, torch.Tensor]]
            adj_relative_cos[i][(j, k)] is the cosine value between a pair of relative vectors node(i -> j) and node(i -> k).
        Returns
        -------
        out : torch.Tensor
            An (len(node_layers[-1]) x output_dim) tensor of output node features.
        """

        # (conv1, agg1) => (conv2)
        x_conv = self.conv1(features, adj)
        x_angle = self.agg1(features, dist, adj_relative_cos)
        x = F.relu(torch.cat((x_conv, x_angle), 1))
        x = F.dropout(x, self.dropout, training=self.training)
        out = self.conv2(x, adj)

        return out
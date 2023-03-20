from ast import Param
from inspect import Parameter
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

np.random.seed(0)


class Aggregator(nn.Module):

    def __init__(self, input_dim=None, output_dim=None, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int or None.
            Dimension of input node features. Used for defining fully
            connected layer in pooling aggregators. Default: None.
        output_dim : int or None
            Dimension of output node features. Used for defining fully
            connected layer in pooling aggregators. Currently only works when
            input_dim = output_dim. Default: None.
        """
        # super(Aggregator, self).__init__()
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

    def forward(self, features, nodes, mapping, rows, dist, init_mapping, num_samples=5):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        nodes : numpy array
            nodes is a numpy array of nodes in the current layer of the computation graph.
        mapping : dict
            mapping is a dictionary mapping node v (labelled 0 to |V|-1) to
            its position in the layer of nodes in the computationn graph
            before nodes. For example, if the layer before nodes is [2,5],
            then mapping[2] = 0 and mapping[5] = 1.
        rows : numpy array
            rows[i] is an array of neighbors of node i which is present in nodes.
        num_samples : int
            Number of neighbors to sample while aggregating. Default: 25.
        Returns
        -------
        out : torch.Tensor
            An (len(nodes) x output_dim) tensor of output node features.
            Currently only works when output_dim = input_dim.
        """
        _choice, _len, _min = np.random.choice, len, min
        mapped_rows = [np.array([mapping[v] for v in row], dtype=np.int64) for row in rows]
        #init_mapped_rows = [np.array([init_mapping[v] for v in row], dtype=np.int64) for row in rows]
        if num_samples == -1:
            sampled_rows = mapped_rows
        else:
            sampled_rows = []
            init_sampled_rows = []
            inds = [_choice(len(row), _min(_len(row), num_samples), _len(row) < num_samples) for row in mapped_rows] # len(rows) x num_samples
            for i in range(len(inds)):
                sampled_rows.append(mapped_rows[i][inds[i]])
                init_sampled_rows.append(np.array(rows[i])[inds[i]])
        n = _len(nodes)
        if self.__class__.__name__ == 'LSTMAggregator':
            out = torch.zeros(n, 2*self.output_dim).to(self.device)
        else:
            out = torch.zeros(n, self.output_dim).to(self.device)

        for i in range(n):
            if _len(sampled_rows[i]) != 0:
                if self.__class__.__name__ == 'MeanAggregator':
                    out[i, :] = self._aggregate(torch.cat((features[mapping[nodes[i]], :].view(1,-1), features[sampled_rows[i], :])), dist[nodes[i], init_sampled_rows[i]])#
                else:
                    out[i, :] = self._aggregate(features[sampled_rows[i], :])
                    #out[i, :] = self._aggregate(torch.cat((features[sampled_rows[i], :], dist[nodes[i], init_sampled_rows[i]].view(-1,1).float()),1))
        return out

    def _aggregate(self, features):
        """
        Parameters
        ----------
        Returns
        -------
        """
        raise NotImplementedError

class MeanAggregator(Aggregator):

    def _aggregate(self, features, dist):
        """
        Parameters
        ----------
        features : torch.Tensor
            Input features.
        Returns
        -------
        Aggregated feature.
        """
        min_dist = torch.min(dist)
        dist = torch.div(min_dist, dist)
        dist = torch.cat((dist, torch.ones(1, dtype=torch.float64)))
        sum_dist = torch.sum(dist)

        return torch.div(torch.sum(torch.mul(features, dist.view(-1, 1)), dim=0), sum_dist)    # Return weighted average
        #return torch.mean(features, dim=0) # Return mean of features

class PoolAggregator(Aggregator):

    def __init__(self, input_dim, output_dim, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features. Used for defining fully connected layer.
        output_dim : int
            Dimension of output node features. Used for defining fully connected layer. Currently only works when output_dim = input_dim.
        """
        # super(PoolAggregator, self).__init__(input_dim, output_dim, device)
        super().__init__(input_dim, output_dim, device)

        self.fc1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def _aggregate(self, features):
        """
        Parameters
        ----------
        features : torch.Tensor
            Input features.
        Returns
        -------
        Aggregated feature.
        """
        # print('features.shape', features.shape)
        out = self.relu(self.fc1(features))
        return self._pool_fn(out)

    def _pool_fn(self, features):
        """
        Parameters
        ----------
        Returns
        -------
        """
        raise NotImplementedError

class MaxPoolAggregator(PoolAggregator):

    def _pool_fn(self, features):
        """
        Parameters
        ----------
        features : torch.Tensor
            Input features.
        Returns
        -------
        Aggregated feature.
        """
        return torch.max(features, dim=0)[0]

class MeanPoolAggregator(PoolAggregator):

    def _pool_fn(self, features):
        """
        Parameters
        ----------
        features : torch.Tensor
            Input features.
        Returns
        -------
        Aggregated feature.
        """
        return torch.mean(features, dim=0)[0]

class LSTMAggregator(Aggregator):

    def __init__(self, input_dim, output_dim, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features. Used for defining LSTM layer.
        output_dim : int
            Dimension of output node features. Used for defining LSTM layer. Currently only works when output_dim = input_dim.
        """
        # super(LSTMAggregator, self).__init__(input_dim, output_dim, device)
        super().__init__(input_dim, output_dim, device)

        self.lstm = nn.LSTM(input_dim, output_dim, bidirectional=True, batch_first=True)

    def _aggregate(self, features):
        """
        Parameters
        ----------
        features : torch.Tensor
            Input features.
        Returns
        -------
        Aggregated feature.
        """
        perm = np.random.permutation(np.arange(features.shape[0]))
        features = features[perm, :]
        features = features.unsqueeze(0)

        out, _ = self.lstm(features)
        out = out.squeeze(0)
        out = torch.sum(out, dim=0)

        return out


class ConvolutionLayer(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, bias=True, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int or None.
            Dimension of input node features. Used for defining fully
            connected layer in pooling aggregators. Default: None.
        output_dim : int or None
            Dimension of output node features. Used for defining fully
            connected layer in pooling aggregators. Currently only works when
            input_dim = output_dim. Default: None.
        """
        super(ConvolutionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight0 = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.weight1 = Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_dim))
        self._reset_parameters()
    

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight1)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


    def forward(self, features, adj):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        adj : torch.Tensor
            An adjacency matrix
        Returns
        -------
        out : torch.Tensor
            An (len(nodes) x output_dim) tensor of output node features.
            Currently only works when output_dim = input_dim.
        """
        support0 = torch.mm(features, self.weight0)
        support1 = torch.mm(features, self.weight1)
        output = support0 + torch.spmm(adj, support1)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'    


class EGNNCLayer(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, channel_dim=None, bias=True, device='cpu'):
        super(EGNNCLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_dim = channel_dim

        self.weight0 = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.weight1 = Parameter(torch.FloatTensor(input_dim, output_dim))

        if bias:
            self.bias = Parameter(torch.FloatTensor(output_dim * channel_dim))
        self._reset_parameters()
    

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight1)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
    

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
            An (len(nodes) x output_dim) tensor of output node features.
            Currently only works when output_dim = input_dim.
        """
        support0 = torch.matmul(features, self.weight0)
        support1 = torch.matmul(features, self.weight1)

        x = torch.matmul(edge_features, support1) + support0
        

        output = torch.cat([xi for xi in x], dim=1)
        
        # x = torch.matmul(features, self.weight)
        # x = torch.matmul(edge_features, x)
        # output = torch.cat([xi for xi in x], dim=1)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
                + str(self.input_dim) + ' -> ' \
                + str(self.output_dim * self.channel_dim) + ')'

    
class DGNNConvolutionLayer(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, bias=True, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int or None.
            Dimension of input node features. Used for defining fully
            connected layer in pooling aggregators. Default: None.
        output_dim : int or None
            Dimension of output node features. Used for defining fully
            connected layer in pooling aggregators. Currently only works when
            input_dim = output_dim. Default: None.
        """
        super(DGNNConvolutionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight0 = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.weight1 = Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_dim))
        self._reset_parameters()
    

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight1)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


    def forward(self, features, dist):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        dist : torch.Tensor
            An (n x n) tensor of the graph.
            dist[i][j] contain the distance between node i and j if they are adjacent, otherwise 0.
        Returns
        -------
        out : torch.Tensor
            An (len(nodes) x output_dim) tensor of output node features.
        """

        support0 = torch.mm(features, self.weight0)
        support1 = torch.mm(features, self.weight1)
        output = support0 + torch.spmm(dist, support1)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


class AAAgregationLayer(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, bias=True, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int or None.
            Dimension of input node features. Used for defining fully
            connected layer in pooling aggregators. Default: None.
        output_dim : int or None
            Dimension of output node features. Used for defining fully
            connected layer in pooling aggregators. Currently only works when
            input_dim = output_dim. Default: None.
        """
        super(AAAgregationLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_dim))
        self._reset_parameters()
    

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


    def forward(self, features, adj_relative_cos):
        """
        Parameters
        ----------
        nodes : List[int]
            A list of nodes
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        adj_relative_cos : Dict[int, Dict[tuple, torch.Tensor]]
            adj_relative_cos[i][(j, k)] is the cosine value between a pair of relative vectors node(i -> j) and node(i -> k).
        Returns
        -------
        out : torch.Tensor
            An (len(nodes) x output_dim) tensor of output node features.
        """
        nodes = list(adj_relative_cos.keys())
        for ind, node in enumerate(nodes):
            agg = torch.zeros(self.input_dim, dtype=float)
            for i, j in adj_relative_cos[node]:
                agg += (features[i] + features[j]) * adj_relative_cos[node][i, j]
            agg = agg.float()
            if ind == 0:
                output = torch.reshape(agg, (1, -1))
            else:
                output = torch.cat((output, torch.reshape(agg, (1, -1))), 0)
        output = torch.mm(output, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'



class DAAAgregationLayer(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, bias=True, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int or None.
            Dimension of input node features. Used for defining fully
            connected layer in pooling aggregators. Default: None.
        output_dim : int or None
            Dimension of output node features. Used for defining fully
            connected layer in pooling aggregators. Currently only works when
            input_dim = output_dim. Default: None.
        """
        super(DAAAgregationLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_dim))
        self._reset_parameters()
    

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


    def forward(self, features, dist, adj_relative_cos):
        """
        Parameters
        ----------
        nodes : List[int]
            A list of nodes
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        dist : torch.Tensor
            An (n x n) tensor of the graph.
            dist[i][j] contain the distance between node i and j if they are adjacent, otherwise 0.
        adj_relative_cos : Dict[int, Dict[tuple, torch.Tensor]]
            adj_relative_cos[i][(j, k)] is the cosine value between a pair of relative vectors node(i -> j) and node(i -> k).
        Returns
        -------
        out : torch.Tensor
            An (len(nodes) x output_dim) tensor of output node features.
        """
        nodes = list(adj_relative_cos.keys())
        for ind, node in enumerate(nodes):
            agg = torch.zeros(self.input_dim, dtype=float)
            for i, j in adj_relative_cos[node]:
                agg += (features[i] + features[j]) * adj_relative_cos[node][i, j] * dist[node, i] * dist[node, j]
            agg = agg.float()
            if ind == 0:
                output = torch.reshape(agg, (1, -1))
            else:
                output = torch.cat((output, torch.reshape(agg, (1, -1))), 0)
        output = torch.mm(output, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


### Code for adding Attention Mechanism
class AttentionModule(torch.nn.Module):
    """
   Global Attention Module to make a pass on graph.
    """
    def __init__(self):
        """ 
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
    
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        filters_3 = 64
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(filters_3,
                                                             filters_3))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """
        act_fn = nn.Softplus()
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = act_fn(global_context)
        sigmoid_scores = torch.sigmoid(torch.mm(embedding, transformed_global.view(-1, 1)))
        representation = torch.sigmoid(torch.mm(torch.t(embedding), sigmoid_scores))
        return representation

class BiTensorNetworkModule(torch.nn.Module):
    """
     Bi-directional Tensor Network module to calculate similarity vector.
    """
    def __init__(self):
        """
        :param args: Arguments object.
        """
        super(BiTensorNetworkModule, self).__init__()
        #self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.filters_3 = 64
        self.tensor_neurons =64
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(64,
                                                             64,
                                                             64))

        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(64,2*64))
        #self.weight_matrix_block_1 = torch.nn.Parameter(torch.Tensor(64,2*64))
        #self.weight_matrix_block_2 = torch.nn.Parameter(torch.Tensor(64,2*64))        
        self.bias = torch.nn.Parameter(torch.Tensor(64, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        #torch.nn.init.xavier_uniform_(self.weight_matrix_block_1)
        #torch.nn.init.xavier_uniform_(self.weight_matrix_block_2)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        #print('>>>>>> Self weight matrix shape : ' + str(self.weight_matrix.view(64, -1).shape))
        embedding_1 = embedding_1[:, None]
        embedding_2 = embedding_2[:,None]
                
        #print('>>>>>> embedding shape : ' + str(torch.t(embedding_1).shape))

        scoring = torch.mm(torch.t(embedding_1), self.weight_matrix.view(64, -1))
        scoring = scoring.view(64, 64)
        scoring = torch.mm(torch.t(scoring), embedding_2)
        combined_representation = torch.cat((embedding_1, embedding_2))
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)
        #combined_representation_1 = torch.cat((embedding_1, embedding_2))
        #combined_representation_2 = torch.cat((embedding_2, embedding_1))
        #block_scoring_1 = torch.mm(self.weight_matrix_block_1, combined_representation_1)
        #block_scoring_2 = torch.mm(self.weight_matrix_block_2, combined_representation_2)
        scores = scoring + block_scoring + self.bias
        #scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        return scores




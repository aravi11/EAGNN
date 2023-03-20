import argparse
import importlib
import json
import sys
from xmlrpc.client import boolean

# import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import glob

from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

import torch
import torch.nn as nn
from itertools import combinations

import networkx as nx

# from tsp_solver.greedy import solve_tsp

from sklearn.neighbors import NearestNeighbors
from zmq import device

from datasets import link_prediction
from layers import MeanAggregator, LSTMAggregator, MaxPoolAggregator, MeanPoolAggregator, BiTensorNetworkModule
import models
import layers


# Functions to visualize bounding boxes and class labels on an image.
# Based on https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py

BOX_COLOR = [(1, 0, 0), (1, 1, 1), (0, 1, 0), (0, 0, 1)]
TEXT_COLOR = [(255, 255, 255),(0, 0, 0),(255, 255, 255),(255, 255, 255)]
KI_CLASSES = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial',
              'epithelial']


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


# MINE -----------------------------------------------------
def export_prediction_as_json(image_name, mode, edges, neg_edges, results_dir, model_id):
    all_edges = []

    edges_copy = edges.copy()  # .tolist()
    for i in range(len(edges_copy)):
        edges_copy[i].append(1)  # crossing edge

    neg_edges_copy = neg_edges.copy()  # .tolist()
    for i in range(len(neg_edges_copy)):
        neg_edges_copy[i].append(0)  # close to edge

    all_edges.extend(edges_copy)
    all_edges.extend(neg_edges_copy)
    jsonString = json.dumps(all_edges, cls=NpEncoder)

    with open(f"{results_dir}/{model_id}_{image_name}_{mode}_edges_result.json", 'w') as f:
        f.write(str(jsonString))



def get_agg_class(agg_class):
    """
    Parameters
    ----------
    agg_class : str
        Name of the aggregator class.
    Returns
    -------
    layers.Aggregator
        Aggregator class.
    """
    return getattr(sys.modules[__name__], agg_class)

def get_criterion(task):
    """
    Parameters
    ----------
    task : str
        Name of the task.
    Returns
    -------
    criterion : torch.nn.modules._Loss
        Loss function for the task.
    """
    if task == 'link_prediction':
        # Pos weight to balance dataset without oversampling
        criterion = nn.BCELoss()#pos_weight=torch.FloatTensor([7.]))

    return criterion


def get_focal_loss_criterion(scores, labels):
    """
    Parameters
    ----------
    task : str
        Name of the task.
    Returns
    -------
    criterion : torch.nn.modules._Loss
        Loss function for the task.
    """
    
    # Pos weight to balance dataset without oversampling
    criterion = nn.BCELoss()#pos_weight=torch.FloatTensor([7.]))

    alpha=0.75
    gamma=5
    bce_loss = criterion(scores, labels.float())

    p_t = torch.exp(-bce_loss)
    alpha_tensor = (1 - alpha) + labels * (2 * alpha - 1)  # alpha if label = 1 and 1 - alpha if label = 0
    f_loss = alpha_tensor * (1 - p_t) ** gamma * bce_loss
    return f_loss.mean()
        

def get_dataset(args, dataset_folder, setPath=None, add_self_edges=False, is_debug=False):
    """
    Parameters
    ----------
    args : tuple
        Tuple of task, dataset name and other arguments required by the dataset constructor.
    setPath: list
        List of path data, example ['P7_HE_Default_Extended_3_1', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_3_1.txt']
    Returns
    -------
    dataset : torch.utils.data.Dataset
        The dataset.
    """
    datasets = []
    mode, num_layers = args

    # folder = dataset_folder
    #folder = '102445_ep49'

    train_paths = []
    test_paths = []
    val_paths = []

    # folder = "dataset"
    folder = "entropy_files/sorenson"

    
    if not is_debug:
        train_glob = glob.glob(
            f'datasets/{folder}/Train/*')
        test_glob = glob.glob(
            f'datasets/{folder}/Test/*')
        val_glob = glob.glob(
            f'datasets/{folder}/Val/*')
    else:
        train_glob = glob.glob(
            f'datasets/{folder}_debug/Train/*')
        test_glob = glob.glob(
            f'datasets/{folder}_debug/Test/*')
        val_glob = glob.glob(
            f'datasets/{folder}_debug/Val/*')

    train_glob = sorted(train_glob)
    test_glob = sorted(test_glob)
    val_glob = sorted(val_glob)

    for i in range(0, len(train_glob), 2):
        train_paths.append([train_glob[i].split('/')[-1]
        .replace('_delaunay_orig_forGraphSAGE_edges.csv', ''), train_glob[i], train_glob[i+1]])

    for i in range(0, len(test_glob), 2):
        test_paths.append([test_glob[i].split('/')[-1]
        .replace('_delaunay_orig_forGraphSAGE_edges.csv', ''), test_glob[i], test_glob[i+1]])
    
    for i in range(0, len(val_glob), 2):
        val_paths.append([val_glob[i].split('/')[-1]
        .replace('_delaunay_orig_forGraphSAGE_edges.csv', ''), val_glob[i], val_glob[i+1]])


    if setPath == None:
        if mode == 'train':
            for path in train_paths:
                # class_attr = getattr(importlib.import_module('datasets.link_prediction'), 'KIGraphDataset')
                class_attr = getattr(importlib.import_module('datasets.link_prediction'), 'KIGraphDataset2')
                dataset = class_attr(path, mode, num_layers, add_self_edges)
                datasets.append(dataset)
        elif mode == 'val':
            for path in val_paths:
                # class_attr = getattr(importlib.import_module('datasets.link_prediction'), 'KIGraphDataset')
                class_attr = getattr(importlib.import_module('datasets.link_prediction'), 'KIGraphDataset2')
                dataset = class_attr(path, mode, num_layers, add_self_edges)
                datasets.append(dataset)
        elif mode == 'test':
            for path in test_paths:
                # class_attr = getattr(importlib.import_module('datasets.link_prediction'), 'KIGraphDataset')
                class_attr = getattr(importlib.import_module('datasets.link_prediction'), 'KIGraphDataset2')
                dataset = class_attr(path, mode, num_layers, add_self_edges)
                datasets.append(dataset)
    else:
        # class_attr = getattr(importlib.import_module('datasets.link_prediction'), 'KIGraphDataset')
        class_attr = getattr(importlib.import_module('datasets.link_prediction'), 'KIGraphDataset2')
        dataset = class_attr(setPath, mode, num_layers)
        datasets.append(dataset)

    return datasets


def get_dataset_gcn(args, dataset_folder, setPath=None, add_self_edges=False, is_debug=False):
    """
    Parameters
    ----------
    args : tuple
        Tuple of task, dataset name and other arguments required by the dataset constructor.
    setPath: list
        List of path data, example ['P7_HE_Default_Extended_3_1', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_3_1.txt']
    Returns
    -------
    dataset : torch.utils.data.Dataset
        The dataset.
    """
    datasets = []
    mode, num_layers = args

    train_paths = []
    test_paths = []
    val_paths = []
   
    folder = "entropy_files/sorenson"
    #folder = "cell_density_dataset"  ### Use only if debug in config.json is true. Used for testing and debuging
    if not is_debug:
        train_glob = glob.glob(
            f'datasets/{folder}/Train/*')
        test_glob = glob.glob(
            f'datasets/{folder}/Test/*')
        val_glob = glob.glob(
            f'datasets/{folder}/Val/*')
    else:
        train_glob = glob.glob(
            f'datasets/{folder}_debug/Train/*')
        test_glob = glob.glob(
            f'datasets/{folder}_debug/Test/*')
        val_glob = glob.glob(
            f'datasets/{folder}_debug/Val/*')


    train_glob = sorted(train_glob)
    test_glob = sorted(test_glob)
    val_glob = sorted(val_glob)
    print(len(train_glob))

    for i in range(0, len(train_glob), 2):
        train_paths.append([train_glob[i].split('/')[-1]
        .replace('_delaunay_orig_forGraphSAGE_edges.csv', ''), train_glob[i], train_glob[i+1]])

    for i in range(0, len(test_glob), 2):
        test_paths.append([test_glob[i].split('/')[-1]
        .replace('_delaunay_orig_forGraphSAGE_edges.csv', ''), test_glob[i], test_glob[i+1]])
    
    for i in range(0, len(val_glob), 2):
        val_paths.append([val_glob[i].split('/')[-1]
        .replace('_delaunay_orig_forGraphSAGE_edges.csv', ''), val_glob[i], val_glob[i+1]])


    if setPath == None:
        if mode == 'train':
            for path in train_paths:
                class_attr = getattr(importlib.import_module('datasets.link_prediction'), 'KIGraphDatasetGCN')
                dataset = class_attr(path, mode, num_layers, add_self_edges)
                datasets.append(dataset)
        elif mode == 'val':
            for path in val_paths:
                class_attr = getattr(importlib.import_module('datasets.link_prediction'), 'KIGraphDatasetGCN')
                dataset = class_attr(path, mode, num_layers, add_self_edges)
                datasets.append(dataset)
        elif mode == 'test':
            for path in test_paths:
                class_attr = getattr(importlib.import_module('datasets.link_prediction'), 'KIGraphDatasetGCN')
                dataset = class_attr(path, mode, num_layers, add_self_edges)
                datasets.append(dataset)
    else:
        class_attr = getattr(importlib.import_module('datasets.link_prediction'), 'KIGraphDatasetGCN')
        dataset = class_attr(setPath, mode, num_layers)
        datasets.append(dataset)

    return datasets


def get_fname(config):
    """
    Parameters
    ----------
    config : dict
        A dictionary with all the arguments and flags.
    Returns
    -------
    fname : str
        The filename for the saved model.
    """
    model = config['model']
    agg_class = config['agg_class']
    hidden_dims_str = '_'.join([str(x) for x in config['hidden_dims']])
    num_samples = config['num_samples']
    batch_size = config['batch_size']
    epochs = config['epochs']
    lr = config['lr']
    weight_decay = config['weight_decay']
    fname = f"{model}.pth"

    return fname


def normalize_matrix_rows(mat):
    """
    Parameters
    ----------
    dist : 2d torch.Tensor
        A 2d tensor
    Returns
    ----------
    dist_norm : 2d torch.Tensor
        A normilized tensor along rows
    """
    mat_sum = torch.sum(mat, dim=1)
    mat_N = torch.reshape(mat_sum, (-1, 1))
    mat_norm = torch.div(mat, mat_N)
    mat_norm = torch.nan_to_num(mat_norm) # fill nan with 0 for given by zero division
    mat_norm = mat_norm.float()
    return mat_norm


def normalize_edge_features_rows(edge_features):
    """
    Parameters
    ----------
    edge_features : numpy array
        3d numpy array (P x N x N).
        edge_features[p, i, j] is the jth feature of node i in pth channel
    Returns
    -------
    edge_features_normed : numpy array
        normalized edge_features.
    """
    deno = np.sum(np.abs(edge_features), axis=2, keepdims=True)
    return np.divide(edge_features, deno, where = deno != 0)


def normalize_edge_feature_doubly_stochastic(edge_features):
    """
    Parameters
    ----------
    edge_features : numpy array
        3d numpy array (P x N x N).
        edge_features[p, i, j] is the jth feature of node i in pth channel
    Returns
    -------
    edge_features_normed : numpy array
        normalized edge_features.
    """
    edge_features_deno = np.sum(edge_features, axis=2, keepdims=True)
    edge_features_tilda = np.divide(edge_features, edge_features_deno, where=edge_features_deno!=0)

    channel = edge_features.shape[0]
    size = edge_features.shape[1]
    edge_features_normed = np.zeros((channel, size, size))
    for p in range(channel):
        d = np.sum(edge_features_tilda[p,:,:], axis=0)
        mul = np.matmul(np.divide(edge_features_tilda[p,:,:], d, where = d != 0),
                        edge_features_tilda[p,:,:].T)
        edge_features_normed[p] = mul
    return edge_features_normed


def get_relative_cos_list(adj_list, coordinates):
    """
    Parameters
    ----------
    adj_list : List
        dictionary of adjacent list
    coordinates : torch.FloatTensor (n x 2)
        x/y coordinates of nodes
    device : string
        'cpu' or 'cuda:0'. Default: 'cpu'.
    Returns
    ----------
    adj_relative_cos : Dict {int: Dict{tuple: torch.Tensor}}
        adj_relative_cos[i][(j, k)] is the cosine value between a pair of relative vectors node(i -> j) and node(i -> k).
    """
    nodes = [i for i in range(len(adj_list))]
    relative_coords = {n: {} for n in nodes}

    for node, adj_nodes in enumerate(adj_list):
        for adj_node in adj_nodes:
            relative_coords[node][adj_node] = coordinates[adj_node] - coordinates[node]    

    adj_relative_cos =  {n: {} for n in nodes}

    for node, adj_nodes in enumerate(adj_list):
        combs = combinations(adj_nodes, 2)
        for pair_nodes in list(combs):
            coor1 = relative_coords[node][pair_nodes[0]].float()
            coor2 = relative_coords[node][pair_nodes[1]].float()
            adj_relative_cos[node][pair_nodes] = (torch.sum(coor1 * coor2) / (torch.linalg.norm(coor1) * torch.linalg.norm(coor2)))

    return adj_relative_cos

def get_scores_multiplication(features):
    """
    Parameters
    ----------
    features : torch.Tensor
        model's features. features[i] is the representation of node i.
    Returns
    ----------
    scores: torch.Tensor
        score matrix. scores[i][j] is the score between node i and node j.
    """
    scores = torch.mm(features, features.t())
    return scores


def concat_node_representations(features, edges, device="cpu"):
    """
    Parameters
    ----------
    features : torch.Tensor
        features[i] is the representation of node i.
    device : string
        'cpu' or 'cuda:0'. Default: 'cpu'.
    Returns
    ----------
    out: torch.Tensor
        Concatinated features.
    """
    out = torch.FloatTensor().to(device)
    for node1, node2 in edges:
        node12 = torch.cat((features[node1], features[node2])).reshape(1, -1)
        out = torch.cat((out, node12), dim=0)

    return out


def concat_node_representations_double(features, edges, device="cpu"):
    """
    Parameters
    ----------
    features : torch.Tensor
        features[i] is the representation of node i.
    device : string
        'cpu' or 'cuda:0'. Default: 'cpu'.
    Returns
    ----------
    out1: torch.Tensor
        Concatinated features.
    out2: torch.Tensor
        Concatinated features.
    """
    out1 = torch.FloatTensor().to(device)
    out2 = torch.FloatTensor().to(device)

    for node1, node2 in edges:

        ### Get edge features here 
        #edge12 = edge_features[:, node1, node2]
        #edge21 = edge_features[:, node2, node1]
        #temp = np.zeros(124) ### adding padding to make edge feature + padding = 128
        #temp = temp.astype(np.float)
        #temp = torch.from_numpy(temp).type(torch.FloatTensor)

        node12 = torch.cat((features[node1], features[node2])).reshape(1, -1)
        node21 = torch.cat((features[node2], features[node1])).reshape(1, -1)

        ### Add edge features to the node dimensions

        #node12 = torch.cat((features[node1], features[node2], edge12, temp)).reshape(1,-1)
        #node21 = torch.cat((features[node2], features[node1], edge21,temp)).reshape(1,-1)
        
        out1 = torch.cat((out1, node12), dim=0)
        #print('<<<<< Out put concat shape ' + str(out1.shape))
        out2 = torch.cat((out2, node21), dim=0)

    return out1, out2

def concat_node_respresentations_double_with_biNTN(features, edges,  device="cpu"):
    """
    Parameters
    ----------
    features : torch.Tensor
        features[i] is the representation of node i.
    device : string
        'cpu' or 'cuda:0'. Default: 'cpu'.
    Returns
    ----------
    out1: torch.Tensor
        Concatinated features.
    out2: torch.Tensor
        Concatinated features.
    """
    out1 = torch.FloatTensor().to(device)
    out2 = torch.FloatTensor().to(device)

    ntn_layer = layers.BiTensorNetworkModule()

    '''
    attention = layers.AttentionModule()
    ntn_layer = layers.TenorNetworkModule()
    node_list = {}
    emd_list = torch.FloatTensor().to(device)
    for node1,node2 in edges:

        if node1 not in node_list.keys(): 
            node_list[node1] = 0

        if node2 not in node_list.keys(): 
            node_list[node2] = 0
        
    emd_list = torch.empty(size=(len(node_list), 64))
    counter = 0
    for eachKey in node_list.keys():
        
        emd_list[counter] = features[eachKey]
        counter = counter +1


    attention_emd = attention(emd_list)
    attention_emd = torch.reshape(attention_emd, (-1,))
    #print('before sigmoid')
    #print(attention_emd)
    attention_emd = torch.sigmoid(attention_emd)
    #print('after sigmoid')
    #print(attention_emd)
    '''


    for node1, node2 in edges:
        ntn_emd_1 = ntn_layer(features[node1], features[node2])
        ntn_emd_2 = ntn_layer(features[node2], features[node1])

        ntn_emd_1 = torch.reshape(ntn_emd_1, (-1,))
        ntn_emd_2 = torch.reshape(ntn_emd_2, (-1,))
        #print('Node shape: '+ str(features[node1].shape))
        #print('NTN shape: '+ str(ntn_emd.shape))
        node12 = torch.cat((features[node1], features[node2], ntn_emd_1)).reshape(1, -1)
        node21 = torch.cat((features[node2], features[node1], ntn_emd_2)).reshape(1, -1)
        out1 = torch.cat((out1, node12), dim=0)
        out2 = torch.cat((out2, node21), dim=0)
        

    return out1,out2


def parse_args():
    """
    Returns
    -------
    config : dict
        A dictionary with the required arguments and flags.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--json', type=str, default='config.json',
                        help='path to json file with arguments, default: config.json')

    parser.add_argument('--stats_per_batch', type=int, default=16,
                        help='print loss and accuracy after how many batches, default: 16')

    # parser.add_argument('--dataset_path', type=str,
    #                     # required=True,
    #                     help='path to dataset')

    parser.add_argument('--results_dir', type=str,
                        # required=True,
                        help='path to save json edge results')
    
    parser.add_argument('--saved_models_dir', type=str,
                        # required=True,
                        help='path to save models')

    parser.add_argument('--task', type=str,
                        choices=['unsupervised', 'link_prediction'],
                        default='link_prediction',
                        help='type of task, default=link_prediction')

    parser.add_argument('--agg_class', type=str,
                        choices=[MeanAggregator, LSTMAggregator, MaxPoolAggregator, MeanPoolAggregator],
                        default=MaxPoolAggregator,
                        help='aggregator class, default: MaxPoolAggregator')
    parser.add_argument('--cuda', action='store_true',
                        help='whether to use GPU, default: False')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout out, currently only for GCN, default: 0.5')
    parser.add_argument('--hidden_dims', type=int, nargs="*",
                        help='dimensions of hidden layers, length should be equal to num_layers, specify through config.json')
    parser.add_argument('--out_dim', type=int, default=1,
                        help='dimension of the model\'s output layer, default=1')                    
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='number of neighbors to sample, default=-1')
    parser.add_argument('--classifier', type=str,
                        choices=['pos_sig', 'neg_sig', 'mlp'],
                        default='mlp',
                        help='classifier type, default: mlp')
    parser.add_argument('--model_id', type=str,
                        default='default_model',
                        help='id of model, default: default_model')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='training batch size, default=32')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of training epochs, default=2')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate, default=1e-4')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay, default=5e-4')

    parser.add_argument('--debug', type=bool, default=False,
                        help="whether debug mode, default: False")

    parser.add_argument('--save', action='store_true',
                        help='whether to save model in trained_models/ directory, default: False')
    parser.add_argument('--test', action='store_true',
                        help='load model from trained_models and run on test dataset')
    parser.add_argument('--val', action='store_true',
                        help='load model from trained_models and run on validation dataset')

    args = parser.parse_args()
    config = vars(args)
    if config['json']:
        with open(config['json']) as f:
            json_dict = json.load(f)
            config.update(json_dict)

            for (k, v) in config.items():
                if config[k] == 'True':
                    config[k] = True
                elif config[k] == 'False':
                    config[k] = False

    config['num_layers'] = len(config['hidden_dims']) + 1

    print('--------------------------------')
    print('Config:')
    for (k, v) in config.items():
        print("    '{}': '{}'".format(k, v))
    print('--------------------------------')

    return config

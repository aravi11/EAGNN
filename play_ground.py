from math import ceil
import json
import os
import sys
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import link_prediction
from layers import MeanAggregator, LSTMAggregator, MaxPoolAggregator, MeanPoolAggregator
from models import DGNN, AAGNN
from models_variants import EAAGNN, EAACGNN
import utils

# Set up arguments for datasets, models and training.
config = utils.parse_args()
print(config)
print(config['classifier'])
# config['num_layers'] = len(config['hidden_dims']) + 1

# if config['cuda'] and torch.cuda.is_available():
#     device = 'cuda:0'
# else:
#     device = 'cpu'
# config['device'] = device


# print(f"val: {config['val']}")
# print(f"test: {config['test']}")


# # Get the dataset, dataloader and model.
# if not config['val'] and not config['test']:
#     dataset_args = ('train', config['num_layers'])

# if config['val']:
#     dataset_args = ('val', config['num_layers'])

# if config['test']:
#     dataset_args = ('test', config['num_layers'])

# datasets = utils.get_dataset_gcn(dataset_args, config['dataset_folder'], is_debug=True)


# loader = DataLoader(dataset=datasets[0], batch_size=config['batch_size'],
#                     shuffle=False, collate_fn=datasets[0].collate_wrapper)

# loaders = []
# for i in range(len(datasets)):
#     loaders.append(DataLoader(dataset=datasets[i], batch_size=config['batch_size'],
#                     shuffle=True, collate_fn=datasets[i].collate_wrapper))

# input_dim, output_dim = datasets[0].get_dims()
# model = AAGNN(input_dim, config['hidden_dims'][0], output_dim,
#                     config['dropout'], config['device'])
# model.to(config['device'])


# sigmoid = nn.Sigmoid()
# criterion = utils.get_criterion(config['task'])

# optimizer = optim.Adam(model.parameters(), lr=config['lr'],
#                     weight_decay=config['weight_decay'])
# epochs = config['epochs']
# epochs = 10

# #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.8)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25], gamma=0.5) # Epoch decay
# model.train()
# print('--------------------------------')
# print('Training.')
# for epoch in range(epochs):
#     print('Epoch {} / {}'.format(epoch+1, epochs))

#     epoch_loss = 0.0
#     epoch_roc = 0.0
#     epoch_batches = 0
#     shuffle = list(range(len(loaders)))
#     random.shuffle(shuffle) # Shuffle order of graphs
#     for i in shuffle:
#         num_batches = int(ceil(len(datasets[i]) / config['batch_size']))
#         epoch_batches += num_batches
#         graph_roc = 0.0
#         running_loss = 0.0
#         for (idx, batch) in enumerate(loaders[i]):
#             adj, adj_list, features, coords, edges, labels, dist, node_layers = batch
#             labels = labels.to(device)
#             optimizer.zero_grad()

#             adj_relative_cos = utils.get_relative_cos_list(adj_list, coords, device)
#             adj, features = adj.to(device), features.to(device)
#             out = model(features, adj, adj_relative_cos)

#             all_pairs = torch.mm(out, out.t())
#             all_pairs = sigmoid(all_pairs)
#             scores = all_pairs[edges.T]
#             loss = criterion(scores, labels.float())
#             loss.backward()
#             optimizer.step()
#             with torch.no_grad():
#                 running_loss += loss.item()
#                 epoch_loss += loss.item()
#                 if (torch.sum(labels.long() == 0).item() > 0) and (torch.sum(labels.long() == 1).item() > 0):
#                     area = roc_auc_score(labels.detach().cpu().numpy(), scores.detach().cpu().numpy())
#                     epoch_roc += area
#                     graph_roc += area
#         running_loss /= num_batches
#         print('    Graph {} / {}: loss {:.4f}'.format(
#             i+1, len(datasets), running_loss))
#         print('    ROC-AUC score: {:.4f}'.format(graph_roc/num_batches))

#     scheduler.step()
#     print("Epoch avg loss: {}".format(epoch_loss / epoch_batches))
#     print("Epoch avg ROC_AUC score: {}".format(epoch_roc / epoch_batches))

# print('Finished training.')
# print('--------------------------------')

# y_true, y_scores = [], []
# for batch in loader:
#     adj, adj_list, features, coords, edges, labels, dist, node_layers = batch
#     labels = labels.to(device)
#     adj_relative_cos = utils.get_relative_cos_list(adj_list, coords, device)
#     adj, features = adj.to(device), features.to(device)
#     out = model(features, adj, adj_relative_cos)
#     all_pairs = torch.mm(out, out.t())
#     all_pairs = sigmoid(all_pairs)
#     scores = all_pairs[edges.T]
#     y_true.extend(labels.detach().cpu().numpy())
#     y_scores.extend(scores.detach().cpu().numpy())

# y_true = np.array(y_true).flatten()
# y_scores = np.array(y_scores).flatten()
# area = roc_auc_score(y_true, y_scores)

# print(y_scores)
# print(y_scores[np.isnan(y_scores)])
# print(area)
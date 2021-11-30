import time
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.profile import get_model_size, get_data_size, count_parameters
import matplotlib.pyplot as plt
from models import *
from dataloader import GraphDataset
from utils import *
import os.path as osp
from sklearn.metrics import classification_report

dataset_name = 'Type_Prewitt_v1'
gnn_layer_by_name = {
    "GCN"      : GCNConv,
    "GAT"      : GATConv,
    "GraphConv": GraphConv
}
layer_name = "GraphConv"
batch_size=512
dataset = GraphDataset(root='/home/linh/Downloads/data/', name=dataset_name, use_node_attr=True)
data_size = len(dataset)
#checking some of the data attributes comment out these lines if not needed to check
print()
print(f'Dataset name: {dataset}:')
print('==================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.
print()
print(data)
print('==================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

# Information of Model setting
print("*"*12)
#print(f'number of hidden dim: {args.hidden_dim}')
#print(f'Dropout parameter setting: {args.dropout}')
print("*"*12)

torch.manual_seed(12345)
dataset = dataset.shuffle()
#this is equivalent of doing
#perm = torch.randperm(len(dataset))
#dataset = dataset[perm]

train_dataset = dataset[:6700]
val_dataset = dataset[6700:8150]
test_dataset = dataset[8150:]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of val graphs: {len(val_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')
print("**************************")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GraphGNNModel(c_in=dataset.num_node_features, 
                      c_out=dataset.num_classes, 
                      c_hidden=64, 
                      layer_name=layer_name, 
                      num_layers=3, 
                      dp_rate_linear=0.5, 
                      dp_rate=0.5).to(device)
print('*****Model size is: ', get_model_size(model))
print("=====Model parameters are: ", count_parameters(model))
print(model)
print("*****Data sizes are: ", get_data_size(data))
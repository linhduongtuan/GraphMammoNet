#import general packages
import time
import random
import numpy as np
import argparse
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

#import torch and PYG
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.profile import get_model_size, get_data_size, count_parameters

#import my source code
from models import *
from utils import *
from dataloader import GraphDataset

#Define arguments
parser = argparse.ArgumentParser(description='PYG version of Mammography Classification')

# Setting Data path and dataset name
parser.add_argument('--root', type=str, default='/home/linh/Downloads/data/', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset_name', type=str, default='BIRAD_Prewitt_v2',
                    help='Choose dataset to train')

# Setting hardwares and random seeds
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA to train a model')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='choose a random seed (default: 42)')
parser.add_argument('--num_workers', type=int, default=4,
                    help='set number of workers (default: 4)')

# Setting training parameters
parser.add_argument('--num_epochs', type=int, default=100, metavar='E',
                    help='Set numbers of epochs for training (default: 100')
parser.add_argument('-b','--batch_size', type=int, default=1024, metavar='B',
                    help='input batch size for training (default: 1024')
parser.add_argument('--lr', type=float, default=0.001, metavar='lr',
                    help='Set learning rate (default: 0.001')
parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='WD',
                    help='Set weight decay (default: 5-e4')
parser.add_argument('--step_size', type=int, default=20, metavar='SS',
                    help='Set step size for scheduler of learning rate (default: 20')

# Setting model configuration
parser.add_argument('--layer_name', type=str, default='GraphConv',
                    help='choose model type either GAT, GCN, or GraphConv (Default: GraphConv')
parser.add_argument('--c_hidden', type=int, default=64,
                    help='Choose numbers of output channels (default: 64')
parser.add_argument('--num_layers', type=int, default=3,
                    help='Choose numbers of Graph layers for the model (default: 3')
parser.add_argument('--dp_rate_linear', type=float, default=0.5,
                    help='Set dropout rate at the linear layer (default: 0.5)')
parser.add_argument('--dp_rate', type=float, default=0.5,
                    help='Set dropout rate at every graph layer (default: 0.5)')

args = parser.parse_args()


if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probaly run with --cuda")
    
    else:
        device_id = torch.cuda.current_device()
        print("***** USE DEVICE *****", device_id, torch.cuda.get_device_name(device_id))
device = torch.device("cuda" if args.cuda else "cpu")
print("==== DEVICE ====", device)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

#############################
dataset = GraphDataset(root=args.root, name=args.dataset_name, use_node_attr=True)
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


dataset = dataset.shuffle()
#this is equivalent of doing
#perm = torch.randperm(len(dataset))
#dataset = dataset[perm]
if dataset.num_classes == 8:
    train_dataset = dataset[6743:]
    val_dataset   = dataset[6743:8187]
    test_dataset  = dataset[8187:]
else:
    train_dataset = dataset[6642:]
    val_dataset   = dataset[6642:8065]
    test_dataset  = dataset[8065:]



train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

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

model = GraphGNNModel(c_in=dataset.num_node_features, 
                      c_out=dataset.num_classes,
                      layer_name=args.layer_name, 
                      c_hidden=args.c_hidden, 
                      num_layers=args.num_layers, 
                      dp_rate_linear=args.dp_rate_linear, 
                      dp_rate=args.dp_rate).to(device)
print('*****Model size is: ', get_model_size(model))
print("=====Model parameters are: ", count_parameters(model))
print(model)
print("*****Data sizes are: ", get_data_size(data))
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

#@profileit()
def train():
    model.train()

    for data in tqdm(train_loader, desc="Iteration"):  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients. 

#@timeit()
def test(loader):
    model.eval()
    correct = 0
    y_pred = []
    y_true = []
    for data in tqdm(loader, desc="Iteration"):  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels

        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(np.squeeze(pred.cpu().numpy().T))
    report = classification_report(y_true, y_pred, digits=4)
    print(report)   
    return correct / len(loader.dataset) # Derive ratio of correct predictions.


start = time.time()
best_val_acc = 0.9
train_accs, val_accs = [], []
for epoch in range(1, args.num_epochs):
    train()
    train_acc = test(train_loader)
    val_acc = test(val_loader)
    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_weight_path = osp.join(args.root + "weights/Graph_" + args.layer_name + "_" + args.dataset_name + "_best" + ".pth")
        print('New best model saved to:', save_weight_path)
        torch.save(model.state_dict(), save_weight_path)

    if epoch % 10 == 0:
        print(f'Epoch numbers: {epoch:03d}, Train Acc: {train_acc:.4f}, Validation Acc: {val_acc:.4f}')
    
    train_accs.append(train_acc)
    val_accs.append(val_acc)
# Visualization at the end of training
fig, ax = plt.subplots()
ax.plot(train_accs, c="steelblue", label="Training")
ax.plot(val_accs, c="orangered", label="Validation")
ax.grid()
ax.legend()
ax.set_xlabel('Epoch Numbers')
ax.set_ylabel('Accuracy')
ax.legend(loc='best')
ax.set_title("Accuracy evolution")
#plt.show()
plt.savefig(args.root + "results/Evolution_training_" + args.layer_name + "_" + args.dataset_name + ".png")
        
end = time.time()
time_to_train = (end - start)/60
print("Total training time to train on GPU (min):", time_to_train)
print("****End training process here******")


def inference(loader):
    model.eval()
    correct = 0
    y_pred = []
    y_true = []
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        #correct += int((pred == data.y).sum())  # Check against ground-truth labels

        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(np.squeeze(pred.cpu().numpy().T))
    report = classification_report(y_true, y_pred, digits=4)
    print(report)
    cm = confusion_matrix(y_true, y_pred)
    # plot the confusion matrix
    if dataset.num_classes == 4:
        display_labels = ['Type A', 'Type B', 'Type C', 'Type D']
    else:
        display_labels = ['BIRAD_0', 'BIRAD_1', 'BIRAD_2','BIRAD_3','BIRAD_4A','BIRAD_4B','BIRAD_4C','BIRAD_5']
    plot_cm(cm=cm, display_labels=display_labels)
    #return torch.sum(y_pred == y_true).item() / len(y_true)      
    return correct / len(loader.dataset) # Derive ratio of correct predictions.

# Inference test set
print("******Start inference on test set*****")
start_2 = time.time()
inference(test_loader)
end_2 = time.time()
time_to_train_2 = (end_2 - start_2)/60
print("Total Inference time to train on GPU (min):", time_to_train_2)


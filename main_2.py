import time
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from models import *
from dataloader import GraphDataset
from utils import *
import os.path as osp
from sklearn.metrics import classification_report

name_dataset = 'Mammograms_Prewitt_v2'

dataset = GraphDataset(root='/home/linh/Downloads/data/', name=name_dataset, use_node_attr=True)
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
#to be sure that data was shuffled before the split, so have to use:
dataset = dataset.shuffle()
#this is equivalent of doing
perm = torch.randperm(len(dataset))
dataset = dataset[perm]


train_dataset = dataset[:6700]
val_dataset = dataset[6700:8150]
test_dataset = dataset[8150:]

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

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
#model = GCN(num_classes=dataset.num_classes, hidden_dim=64, node_features_dim=dataset.num_node_features).to(device)
model = GNN(num_classes=dataset.num_classes, hidden_dim=64, node_features_dim=dataset.num_node_features).to(device)
num_of_parameters = sum(map(torch.numel, model.parameters()))
print('*****Total parameter numbers of the model are: ', num_of_parameters)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        label = data.y
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients. 

def test(loader):
    model.eval()
    correct = 0
    y_pred = []
    y_true = []
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to('cuda')
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels

        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(np.squeeze(pred.cpu().numpy().T))
    report = classification_report(y_true, y_pred, digits=4)
    print(report)   
    return correct / len(loader.dataset) # Derive ratio of correct predictions.


num_epochs = 100
start = time.time()
best_val_acc = 0.9
train_accs, val_accs = [], []
for epoch in range(1, num_epochs):
    train()
    train_acc = test(train_loader)
    val_acc = test(val_loader)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_weight_path = osp.join(f"GNN_" + name_dataset + "_best" + ".pth")
        print('New best model saved to:', save_weight_path)
        torch.save(model.state_dict(), save_weight_path)

    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Validation Acc: {val_acc:.4f}')
    
    train_accs.append(train_acc)
    val_accs.append(val_acc)
# Visualization at the end of training
fig, ax = plt.subplots()
ax.plot(train_accs, c="steelblue", label="Training")
ax.plot(val_accs, c="orangered", label="Validation")
ax.grid()
ax.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend(loc='best')
ax.set_title("Accuracy evolution")
#plt.show()
plt.savefig(f"Evolution_training_GNN_" + name_dataset + ".png")
        
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
        data = data.to('cuda')
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        #correct += int((pred == data.y).sum())  # Check against ground-truth labels

        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(np.squeeze(pred.cpu().numpy().T))
    report = classification_report(y_true, y_pred, digits=4)
    print(report)
    cm = confusion_matrix(y_true, y_pred)
    # plot the confusion matrix
    display_labels = ['BIRAD_0', 'BIRAD_1', 'BIRAD_2', 'BIRAD_3','BIRAD_4A', 'BIRAD_4B','BIRAD_4C', 'BIRAD_5']
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
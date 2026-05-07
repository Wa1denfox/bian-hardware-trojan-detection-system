import os
from getdata import getdata
import torch
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GraphConv
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
import torch
import random
dataset=[]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def pdfFilesPath(path):
    filePaths = []  # 存储目录下的所有文件名，含路径
    for root, dirs, files in os.walk(path):
        for file in files:
            filePaths.append(os.path.join(root, file))
    return list(map(lambda x:x.replace("\\",'/'),filePaths))
for file in pdfFilesPath('data/TjIn'):
            try:
                dataset.append(getdata(file,1).to(device))
            except:
                pass
for file in pdfFilesPath('data/TjFree'):
            try:
                dataset.append(getdata(file,0).to(device))
            except:
                pass
n=len(dataset)
random.shuffle(dataset)
print(dataset)
train_dataset=dataset[:int(0.9*n)]
test_dataset=dataset[int(0.9*n)//9:]
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)


class GCN(torch.nn.Module):

    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(64, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)


    def forward(self, x, edge_index,edge_attr, batch):
        x = self.conv1(x, edge_index,edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index,edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)
for step,data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

model = GCN(hidden_channels=64).to(device)
print(model)
print(len(dataset))
model=GCN(hidden_channels=64)
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
criterion=torch.nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
criterion=torch.nn.CrossEntropyLoss()

def train():
    model.train()
    for data in train_loader:
         out=model(data.x,data.edge_index,data.edge_attr,data.batch)
         loss=criterion(out,data.y)
         loss.backward()
         optimizer.step()
         optimizer.zero_grad()

def test(loader):
     model.eval()
     correct=0
     tp=fp=tn=fn=0
     for data in loader:
         out=model(data.x,data.edge_index,data.edge_attr,data.batch)
         out=F.softmax(out,dim=1)
         pred=out.argmax(dim=1)
     for i in range(len(data)):
         if int(data.y[i])==1 and int(pred[i])==1:
             tp+=1
         elif int(data.y[i]) == 0 and int(pred[i])== 0:
             tn += 1
         elif int(data.y[i]) == 0 and int(pred[i]) == 1:
             fp += 1
         elif int(data.y[i]) == 1 and int(pred[i]) == 0:
             fn += 1
     Accuracy=(tp+tn)/(tp+tn+fp+fn)
     try:
        Precision=tp/(tp+fp)
        Recall=tp/(tp+fn)
        Specificity=tn/(fp+tn)
        F1_Score=(2*Precision*Recall)/(Precision+Recall)
     except:
         Precision = 0.0
         Recall = 0.0
         Specificity = 0.0
         F1_Score = 0.0
     return [Accuracy,Precision,Recall,Specificity,F1_Score]

for epoch in range(1,50):
    train()
    train_acc=test(train_loader)
    test_acc=test(test_loader)
    print(f'Epoch: {epoch:03d}, Train metrics: {train_acc}, Test metrics: {test_acc}')
torch.save(model.state_dict(), 'model.pth')



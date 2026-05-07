import torch
import os
import regex
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GraphConv
import torch.nn.functional as F
from getdata import getdata
import argparse
from pyentrp import entropy as ent
def pdfFilesPath(path):
    filePaths = []  # 存储目录下的所有文件名，含路径
    for root, dirs, files in os.walk(path):
        for file in files:
            filePaths.append(os.path.join(root, file))
    return list(map(lambda x:x.replace("\\",'/'),filePaths))
def find(file):
    print("以下是可疑的LUT门：")
    with open(file, 'r', encoding='utf-8') as file:
        content = file.readlines()
    modulename=''
    count=5
    for i in range(len(content)-1):
        if count==0:
            break
        if 'module' in content[i] and 'enmodule' not in content[i]:
            p=content[i].split('(')[0]
            q=p.split(' ')[-1]
            modulename=q
        if 'LUT' in content[i] and "'" in content[i+1]:
            lut=content[i+1]
            try:
                while lut[0] != "'":
                    lut = lut[1:]
                if lut[1] == 'b':
                    lut = lut[2:-2]
                    init = lut
                elif lut[1] == 'h':
                    lut = lut[2:-2]
                    init=bin(int(lut, 16))[2:]
                entroy=ent.shannon_entropy(list(init))
                if entroy<0.2:
                        count-=1
                        print(f"LUT位置：第{i}行，位于{modulename}模块内")
                        print("模块信息：")
                        print(content[i],end='')
                        print(content[i+1], end='')
                        print(f"模块信息熵={entroy},该LUT翻转可能性小，疑似为木马模块！")
            except:
                pass
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
def testfunc(file,model_path):

    model=GCN(hidden_channels=64)
    model.load_state_dict(torch.load(model_path))
    data=getdata(file,1)
    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
    out = F.softmax(out, dim=1)
    pred = out.argmax(dim=1)
    filename=file.split("/")[-1]
    print(f"文件名:{filename}")
    if(int(pred[0])==1):
        print(f"该设计疑似被植入木马！")
    else:
        print(f"该设计未被检测出木马.")
    print("可疑指数：",end='')
    for i in range(0,int(out[0][1]*100)):
        print("▎", end='')
    print(round(float(out[0][1]),4))
    if (int(pred[0]) == 1):
        find(file)
    print("该文件检测完毕")
    print("--------------------------------")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='test/MC8051-T600_syn.v')
    parser.add_argument('--model', type=str, default='model_show.pth')
    args = parser.parse_args()
    print("======================开始检测======================")
    testfunc(args.file,args.model)
    print("======================检测结束======================")
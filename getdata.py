import re as regex
from pyentrp import entropy as ent
import torch
import os
from torch_geometric.data import Data
def getdata(filename,type):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    luts=regex.findall(r"LUT[0-9]+.*\n.*.INIT\([0-9]+'[\w]*\)", content)
    #print(luts)
    inits=[]
    features=[]
    #print(features)
    edge=[[],[]]
    edge_feature=[]
    for lut in luts:
        while lut[0]!="'":
            lut=lut[1:]
        if lut[1]=='b':
            lut=lut[2:-1]
            inits.append(lut)
        elif lut[1]=='h':
            lut=lut[2:-1]
            inits.append(bin(int(lut, 16))[2:])
    for init in inits:
        features.append(list(map(int,list(init.rjust(64,'0')))))
    if not features:
        return None
    for i in range(len(inits)):
        for j in range(i+1,len(inits)):
            if  len(inits[i]+inits[j])>=128:
                edge[0].append(i)
                edge[1].append(j)
                edge_feature.append(round(ent.shannon_entropy(list(inits[i]+inits[j])),5))
    x=torch.tensor(features,dtype=torch.float32)
    y=torch.tensor([type])
    edge_index = torch.tensor(edge,dtype=torch.long)
    edge_attr = torch.tensor(edge_feature,dtype=torch.float32)
    data = Data(x=x, y=y,edge_index=edge_index, edge_attr=edge_attr)
    return data




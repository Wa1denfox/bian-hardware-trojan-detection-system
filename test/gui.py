import re as regex
from pyentrp import entropy as ent
import torch
from torch_geometric.data import Data
def just(str0,length):
    if len(str0)>length:
        return str0.rjust(length,'0')
    i=1
    while 2**i<len(str0):
        i+=1
    str1='0'*(2**i-len(str0))+str0
    while len(str1)<length:
        str1+=str1

    return str1
def getdata1(filename,type):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    luts=regex.findall(r"LUT[0-9]+.*\n.*.INIT\([0-9]+'[\w]*\)", content)
    inits=[]
    features=[]
    edge=[[],[]]
    edge_feature=[]
    maxn=0
    for lut in luts:
        while lut[0]!="'":
            lut=lut[1:]
        if lut[1]=='b':
            lut=lut[2:-1]
            inits.append(lut)
        elif lut[1]=='h':
            lut = lut[2:-1]
            lut=bin(int(lut, 16))[2:]
            inits.append(lut)

    for init in inits:
        features.append(list(map(int,list(just(init,64)))))
    for i in range(len(inits)):
        for j in range(i+1,len(inits)):
                edge[0].append(i)
                edge[1].append(j)
                edge_feature.append(round(ent.shannon_entropy(list(inits[i]+inits[j])),5))
    x=torch.tensor(features,dtype=torch.float32)
    y=torch.tensor([type])
    edge_index = torch.tensor(edge,dtype=torch.long)
    edge_attr = torch.tensor(edge_feature,dtype=torch.float32)
    data = Data(x=x, y=y,edge_index=edge_index, edge_attr=edge_attr)
    print(data,filename)
    return data


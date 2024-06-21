import iCallds2


import torch as th
import dgl, os
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from sklearn.decomposition import PCA

import iCallds3, random

import dgl.nn as dglnn

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import seaborn as sns
import umap
import umap.plot
from sklearn.metrics import precision_recall_curve




class LinkPredictor(th.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()
        # Append different modulList
        # th.nn.Linear(in_channels, hidden_channels) ? ? first layer
        self.lins = th.nn.ModuleList()
        self.lins.append(th.nn.Linear(in_channels, hidden_channels))
        # middle layers
        # Eg: [2, 100] [100, 2]
        for _ in range(num_layers - 2):
            self.lins.append(th.nn.Linear(hidden_channels, hidden_channels))
        # last layers
        self.lins.append(th.nn.Linear(hidden_channels, out_channels))
        # all Linear layers
        self.dropout = dropout
        self.reset_parameters()

    # backward
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):#_i, x_j):
        #x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return th.sigmoid(x)


# node is the output of palmtree and other features(vector)
# one of GN relationational for same node and same link
class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        # 异构图 同构图
        # 不能把神经网络变深 GN如何实现 图神经网络输入的形状是不确定的 传统的神经网络输入输出都是确定的
        # 每一个节点 有多少个领结节点是不确定的 图神经网络的卷积是求的领结节点的平均值message passing
        # GN层数无法做深？
        # rel: relation
        self.conv1 = dglnn.HeteroGraphConv({
            rel[1]: dglnn.GraphConv(in_feats[rel[0]], hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel[1]: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel[1]: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}
        h = self.conv3(graph, h)
        return h

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names, dropout):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names, dropout)
        #self.pred = HeteroDotProductPredictor(out_features*2)

    def forward(self, g, x):
        h = self.sage(g, x)
        return h #self.pred(h, qlist)

def init_dataset(Revedges = True, Adddata = True, Addfunc = True, DataRefedgs = True, Calledges = True, CodeRefedgs = True, Laplacian_pe = False):
    dataset = iCallds2.iCallds2(Revedges=Revedges, Calledges=Calledges, Laplacian_pe=Laplacian_pe,
                 Adddata = Adddata, Addfunc = Addfunc, DataRefedgs = DataRefedgs, CodeRefedgs = CodeRefedgs)

    # rev -> reverse
    rel_names = [('code', 'code2func_edges', 'func'),
                 ('code', 'code2code_edges', 'code'),
                 ('code', 'codecall_edges', 'code'),
                 ('code', 'codexrefcode_edges', 'code'),
                 ('code', 'codexrefdata_edges', 'data'),
                 ('data', 'dataxrefcode_edges', 'code'),
                 ('data', 'dataxrefdata_edges', 'data'),
                 ('code', 'rev_code2code_edges', 'code'),
                 ('code', 'rev_codecall_edges', 'code'),
                 ('code', 'rev_codexrefcode_edges', 'code'),
                 ('data', 'rev_codexrefdata_edges', 'code'),
                 ('code', 'rev_dataxrefcode_edges', 'data'),
                 ('data', 'rev_dataxrefdata_edges', 'data'),
                 ('func', 'rev_code2func_edges', 'code')
                 ]
    if not Revedges:
        rel_names = [o for o in rel_names if not o[1].startswith('rev_')]
    if not Adddata:
        rel_names = [o for o in rel_names if o[0]!='data'and o[2]!='data']
    if not Addfunc:
        rel_names = [o for o in rel_names if o[0]!='func'and o[2]!='func']
    if not DataRefedgs:
        rel_names = [o for o in rel_names if not o[1].endswith('xrefdata_edges')]
    if not CodeRefedgs:
        rel_names = [o for o in rel_names if not o[1].endswith('xrefcode_edges')]
    if not Calledges:
        rel_names = [o for o in rel_names if not o[1].endswith('codecall_edges')]


    return dataset, rel_names

device = None

# icallds2
def get_one_graph(dataset, i, Adddata = True, Addfunc = True, Laplacian_pe=False):
    print(f"graph{i} is loading")
    g, glabels = dataset[i]
    g = g.to(device)
    if Adddata:
        print(f'{i} Code: {g.num_nodes("code")}, Data: {g.num_nodes("data")}, Edges: {g.num_edges()}, GT: {glabels["GT_edges"].shape[1]}')
        node_features = {
            'code': g.nodes['code'].data['feat'].view(g.nodes['code'].data['feat'].shape[0], -1).float(),
            'data': g.nodes['data'].data['feat'].view(g.nodes['data'].data['feat'].shape[0], -1).float()}
    else:
        print(f'{i} Code: {g.num_nodes("code")}, Edges: {g.num_edges()}, GT: {glabels["GT_edges"].shape[1]}')
        node_features = {
            'code': g.nodes['code'].data['feat'].view(g.nodes['code'].data['feat'].shape[0], -1).float()}
    if Addfunc:
        if Laplacian_pe:
            node_features['func'] = g.nodes['func'].data['feat'].view(g.nodes['func'].data['feat'].shape[0], -1).float()
        else:
            node_features['func'] = th.zeros(g.num_nodes("func")).view(-1,1).float().to(device)

    return g, glabels, node_features


def draw():
    hidden_features = 512
    savePATH = "/home/isec/test/model/result_all_70/allpe"
    Revedges = True
    Adddata = True
    Addfunc = True
    DataRefedgs = True
    Calledges = True
    CodeRefedgs = True
    Laplacian_pe= True
    callsite_list = []
    callee_list =[]
    dataset, rel_names = init_dataset(Revedges=Revedges, Adddata=Adddata, Addfunc=Addfunc, DataRefedgs = DataRefedgs, Calledges = Calledges, CodeRefedgs = CodeRefedgs, Laplacian_pe=Laplacian_pe)
    print(f"len(dataset): {len(dataset)}")
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    pe = 0
    if Laplacian_pe:
        pe = 2
    in_features = {'code':70*128+2+pe, 'data': 1+pe, 'func': 1+int(pe/2)}
    if not Adddata:
        in_features.pop('data')
    if not Addfunc:
        in_features.pop('func')
    # GN and linkpredictor model
    model = Model(in_features, hidden_features, hidden_features, rel_names, dropout = 0.2)
    predictor = LinkPredictor(hidden_features*2, hidden_features, 1, 3, 0)
    model, predictor = map(lambda x: x.to(device), (model, predictor))
    model.float()
    predictor.float()
    preds = []
    targets = []
    model.eval()
    predictor.eval()
    if Laplacian_pe:
        randomlist = []
        for i in range(dataset.__len__()):

            graphfile = os.path.join(dataset.directory, str(i) + '.graphpe')
            if os.path.exists(graphfile)  or os.path.getsize(graphfile[:-2])<70000000:
                randomlist.append(i)
        tmp = randomlist.__len__()
        #print(f'data num = {tmp}')
        validlist = randomlist[int(tmp*0.9):tmp]
        randomlist = randomlist[:int(tmp*0.9)]
    randomlist = []
    randomlist = list(range(dataset.__len__()))
    random.shuffle(randomlist)
    #test_list = []
    # model 保存模型
    if os.path.exists(os.path.join(savePATH, 'predictor.checkpoint')):
        print("used existing model and predictor")
        model.load_state_dict(th.load(os.path.join(savePATH, 'model.checkpoint')))
        predictor.load_state_dict(th.load(os.path.join(savePATH, 'predictor.checkpoint')))
    for i in range(len(randomlist)):
        g, glabels, node_features = get_one_graph(dataset=dataset, i=randomlist[i], Adddata = Adddata, Addfunc = Addfunc, Laplacian_pe=Laplacian_pe)
        pred = model(g, node_features)
        edge = glabels['GT_edges']
        pos_out = predictor(th.cat((pred['code'][edge[0]],pred['code'][edge[1]]),dim=1))
        # print(f"shape: {pred['code'][edge[0]].shape}")
        edge = glabels['GT_F_edges']
        neg_out = predictor(th.cat((pred['code'][edge[0]], pred['code'][edge[1]]),
                                               dim=1))
        preds+=pos_out.tolist()
        preds+=neg_out.tolist()
        targets+=[[1]]*pos_out.shape[0]
        targets+=[[0]]*neg_out.shape[0]
        for pre in pred['code'][edge[1]]:
            # print(pre.shape)
            # print(len(pre.tolist()))
            callsite_list.append(pre.tolist())
        print(f"row: {len(callsite_list)}")
        print(f"column{len(callsite_list[0])}")
        callee_list.extend(pred['code'][edge[1]].tolist())
    icallee_precision = []
    icallee_recall = []
    switch = 0
    cnt = 0
    cnt2 = 0
    with open("/home/isec/Desktop/mymodel/precision_recall.txt", 'r') as file:
        while True:
            line = file.readline().strip()
            if line == '' and cnt == cnt2:
                break
            if line == '':
                switch = 1
                continue
            if switch == 0:
                icallee_precision.append(float(line))
                cnt += 1
            else:
                icallee_recall.append(float(line))
                cnt2 += 1
    print(len(icallee_precision))
    print(len(icallee_recall))


    sns.set_theme(style='white', context='notebook', rc={'figure.figsize':(14,10)})
    reducer = umap.UMAP()
    # umap.plot.points(reducer)


    #Precision Recall curve
    precision, recall, _ = precision_recall_curve(targets, preds)
    # print(len(precision))
    # print(len(recall))
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='red', lw=2, label = "neucall")
    plt.plot(icallee_recall, icallee_precision, color='blue', lw=2, label="icallee")

    plt.fill_between(recall, precision, color='red', alpha=0.05)
    plt.fill_between(icallee_recall, icallee_precision, color='blue', alpha=0.05)
    plt.xlabel('Recall', fontsize = 15)
    plt.ylabel('Precision', fontsize = 15)
    plt.title('Precision-Recall Curve', fontsize = 15)
    plt.grid(True)
    # plt.text(1, 0.5, 'neucall', verticalalignment='bottom', horizontalalignment='left', fontsize=9, color='red')
    # plt.text(1, 0.2, 'icallee', verticalalignment='bottom', horizontalalignment='left', fontsize=9, color='blue')

    # Separate legends for the red and blue lines
    plt.legend(loc='lower left')
    plt.legend(['NeuCall', 'Callee'], fontsize = 15)

    plt.savefig('/home/isec/Desktop/model/graph/precision-recall_all_70.pdf', format="pdf")
    plt.show()

if __name__ == "__main__":
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    draw()
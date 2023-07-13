import argparse
import time

import torch as th
import dgl, os
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

import iCallds2, random

import dgl.nn as dglnn

class LinkPredictor(th.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = th.nn.ModuleList()
        self.lins.append(th.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(th.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(th.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.reset_parameters()

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



class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, dropout=0.2):
        super().__init__()
        self.dropout = dropout

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

def init_dataset(Revedges = True, Adddata = True, Addfunc = True, DataRefedgs = True, Calledges = True, CodeRefedgs = True, Laplacian_pe = False, Onlysave = True):
    dataset = iCallds2.iCallds2(Revedges=Revedges, Calledges=Calledges, Laplacian_pe=Laplacian_pe,
                 Adddata = Adddata, Addfunc = Addfunc, DataRefedgs = DataRefedgs, CodeRefedgs = CodeRefedgs, Onlysave = Onlysave)

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

def get_one_graph(dataset, i, Adddata = True, Addfunc = True):
    g, glabels = dataset[i]
    return
skips = []
def exp_all(epochs = 6, hidden_features = 128, savePATH = 'E:\\iCallasm\\all', Revedges = True, Adddata = True, Addfunc = True, DataRefedgs = True, Calledges = True, CodeRefedgs = True, Laplacian_pe=False, Onlysave = True):
    dataset, rel_names = init_dataset(Revedges=Revedges, Adddata=Adddata, Addfunc=Addfunc, DataRefedgs = DataRefedgs, Calledges = Calledges, CodeRefedgs = CodeRefedgs, Laplacian_pe=Laplacian_pe, Onlysave = Onlysave)

    if True:
        randomlist = []
        for i in range(dataset.__len__()):
            if i in skips:
                continue
            graphfile = os.path.join(dataset.directory, str(i) + '.graphpe')
            if not os.path.exists(graphfile):# and os.path.getsize(graphfile[:-2]) < 100000000:
                randomlist.append(i)

        x = 6
        tmp = randomlist.__len__()
        print(tmp)
        randomlist = randomlist[int(tmp/7*x):int(tmp/7*(x+1))]
        #random.shuffle(randomlist)
        for i in skips:
            if i in randomlist:
                randomlist.remove(i)

        for i in range(len(randomlist)):
            graphfile = os.path.join(dataset.directory, str(randomlist[i]) + '.graphpe')
            if os.path.exists(graphfile):
                continue
            get_one_graph(dataset=dataset, i=randomlist[i], Adddata = Adddata, Addfunc = Addfunc)
            print(f'{randomlist[i]} {i}')
    else:
        randomlist = []
        for i in range(dataset.__len__()):
            graphfile = os.path.join(dataset.directory, str(i) + '.graphpe')
            if os.path.exists(graphfile):
                get_one_graph(dataset=dataset, i=i, Adddata = Adddata, Addfunc = Addfunc)
                print(f'{i}')



if __name__ == "__main__":
    epochs = 10
    skips = [517, 3002, 2260, 2263, 2267, 2264, 2508, 2348, 3732,
             5361, 5400, 1462, 5952, 2330, 608, 5803, 2603, 2971, 6060, 6062, 2876,
             4573, 5956, 2819, 5958, 4580, 4574, 4579, 4587, 3946, 2172, 5281, 3575, 3576, 3061, 5963, 5960, 1152, 1155,
             4717, 5988,
             5151,
             ]
    hidden_features = 512
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    exp_all(epochs = epochs, hidden_features=hidden_features, savePATH = 'E:\\iCallasm50\\allpe',
            Revedges = True, Adddata = True, Addfunc = False, DataRefedgs = True, Calledges = True, CodeRefedgs = True, Laplacian_pe=True, Onlysave = True)
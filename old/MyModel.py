import argparse
import time

import torch as th
import dgl, os
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

import iCallds, random

import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

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

class HeteroDotProductPredictor(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.output = nn.Linear(in_feats, 1)
    def forward(self, h, qlist):
        #a = th.sum(h['code'][qlist[0]]*h['code'][qlist[1]],dim=-1)
        a = self.output(th.cat((h['code'][qlist[0]].view(h['code'][qlist[0]].shape[0],-1),h['code'][qlist[1]].view(h['code'][qlist[1]].shape[0],-1)),dim=1))
        return a


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, dropout=0):
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


def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = th.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})

def init_nor_palm():
    data = iCallds.iCallds()
    g = data[0]
    rel_names = ['code2code_edges', 'codecall_edges', 'codexrefcode_edges', 'codexrefdata_edges', 'dataxrefcode_edges',
                 'dataxrefdata_edges', 'GT_edges', 'GT_F_edges']
    #num_rels = 7
    return g, rel_names, len(rel_names)

def init_r_one():
    data = iCallds.iCallds()
    g = data[0]
    rel_names = ['code2code_edges', 'codecall_edges', 'codexrefcode_edges', 'codexrefdata_edges', 'dataxrefcode_edges',
     'dataxrefdata_edges',
     'code2code_edges_reverse', 'codecall_edges_reverse', 'codexrefcode_edges_reverse', 'codexrefdata_edges_reverse',
     'dataxrefcode_edges_reverse',
     'dataxrefdata_edges_reverse', 'GT_edges', 'GT_F_edges']
    num_rels = 14
    return g, rel_names, num_rels


epochs = 80
skips = []

def exp_nodata(hidden_features = 128, savePATH = 'D:\\iCallasm\\nodata'):
    dataset = iCallds.iCallds()
    #g, rel_names, num_rels = init_nor_palm() #init_r_one()
    rel_names = [('code', 'code2code_edges', 'code'),
                 ('code', 'codecall_edges', 'code'),
                 ('code', 'codexrefcode_edges', 'code'),
                 #('code', 'codexrefdata_edges', 'data'),
                 #('data', 'dataxrefcode_edges', 'code'),
                 #('data', 'dataxrefdata_edges', 'data'),
                 ('code', 'rev_code2code_edges', 'code'),
                 ('code', 'rev_codecall_edges', 'code'),
                 ('code', 'rev_codexrefcode_edges', 'code'),
                 #('data', 'rev_codexrefdata_edges', 'code'),
                 #('code', 'rev_dataxrefcode_edges', 'data'),
                 #('data', 'rev_dataxrefdata_edges', 'data'),
                 #('code', 'GT_edges', 'code'),
                 #('code', 'GT_F_edges', 'code')
                 ]

    in_features = {'code':40*128+2}
    #in_features[]
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = Model(in_features, hidden_features, hidden_features, rel_names, dropout = 0.2)
    predictor = LinkPredictor(hidden_features*2, hidden_features, 1, 3, 0)
    model, predictor = map(lambda x: x.to(device), (model, predictor))
    opt = th.optim.Adam(model.parameters())
    model.float()
    predictor.float()
    bestf1 = 0
    f1 = []
    if os.path.exists(os.path.join(savePATH, 'predictor.checkpoint')):
        model.load_state_dict(th.load(os.path.join(savePATH, 'model.checkpoint')))
        predictor.load_state_dict(th.load(os.path.join(savePATH, 'predictor.checkpoint')))
        with open(os.path.join(savePATH, 'bestf1.txt'), 'r') as f:
            bestf1 = float(f.read())
        with open(os.path.join(savePATH, 'f1s.txt'), 'r') as f:
            slist = f.read().split('\n')
            slist.remove('')
            f1 = [float(i) for i in slist]


    model.train()
    predictor.train()
    i = -1
    timep = []
    delg = []
    timep.append(time.time())
    trains = [0, int(dataset.__len__()*0.8)]
    valids = [int(dataset.__len__()*0.8), int(dataset.__len__()*0.9)]
    tests = [int(dataset.__len__()*0.9), dataset.__len__()]
    metric = torchmetrics.F1Score()
    for epoch in range(trains[1]*epochs):
        i = random.randrange(trains[0], trains[1])
        if i in skips:
            continue
        #i=1095
        g, glabels = dataset[i]
        #g = dgl.node_subgraph(g, {'code': list(range(g.num_nodes('code')))})
        g = g.node_type_subgraph(['code'])
        g = g.to(device)
        print(f'{i} Code: {g.num_nodes("code")}, Edges: {g.num_edges()}, GT: {glabels["GT_edges"].shape[1]}')
        if glabels['GT_edges'].shape[1] == 0:
            delg.append(i)
            print(f'Skipping {i}')
            continue
        #glabel = th.cat((th.ones(glabels['GT_edges'].shape[1]),th.zeros(glabels['GT_F_edges'].shape[1])),0)
        node_features = {'code': g.nodes['code'].data['feat'].view(g.nodes['code'].data['feat'].shape[0],-1).float()}
        pred = model(g, node_features)
        #loss = ((pred - glabel) ** 2).mean()


        edge = glabels['GT_edges']
        pos_out = predictor(th.cat((pred['code'][edge[0]],pred['code'][edge[1]]),dim=1))#(pred['code'][edge[0]], pred['code'][edge[1]])
        pos_loss = -th.log(pos_out + 1e-15).mean()

        edge = glabels['GT_F_edges']
        neg_out = predictor(th.cat((pred['code'][edge[0]],pred['code'][edge[1]]),dim=1))#(pred['code'][edge[0]], pred['code'][edge[1]])
        neg_loss = -th.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        timep.append(time.time())
        print(f'Total time: {timep[-1]-timep[0]:.2f}, time: {timep[-1]- timep[-2]:.2f}')
        print(f'{epoch} ({epoch/(3777*epochs):.2%}): {loss.item()}')
        pos_loss = None
        neg_loss = None
        loss = None
        g = None
        edge = None
        pos_out = None
        neg_out = None
        neg_loss = None
        loss = None
        if (epoch+1)%4000 == 0:
            preds = []
            targets = []
            model.eval()
            predictor.eval()
            timetest = time.time()
            for i in range(valids[0], valids[1]):
                if i in skips:
                    continue

                g, glabels = dataset[i]
                g = g.node_type_subgraph(['code'])
                g = g.to(device)
                if glabels['GT_edges'].shape[1] == 0:
                    delg.append(i)
                    print(f'Skipping {i}')
                    continue
                node_features = {
                    'code': g.nodes['code'].data['feat'].view(g.nodes['code'].data['feat'].shape[0], -1).float()}
                pred = model(g, node_features)

                edge = glabels['GT_edges']
                pos_out = predictor(th.cat((pred['code'][edge[0]], pred['code'][edge[1]]),
                                           dim=1))
                edge = glabels['GT_F_edges']
                neg_out = predictor(th.cat((pred['code'][edge[0]], pred['code'][edge[1]]),
                                           dim=1))
                preds+=pos_out.tolist()
                preds+=neg_out.tolist()
                targets+=[[1]]*pos_out.shape[0]
                targets+=[[0]]*neg_out.shape[0]

            f1.append(metric(th.tensor(preds), th.tensor(targets)).item())
            if bestf1 <f1[-1]:
                bestf1 = f1[-1]
                th.save(model.state_dict(), os.path.join(savePATH, 'model.checkpoint'))
                th.save(predictor.state_dict(), os.path.join(savePATH, 'predictor.checkpoint'))
                with open(os.path.join(savePATH, 'bestf1.txt'), 'w') as f:
                    f.write(str(bestf1))
            with open(os.path.join(savePATH, 'f1s.txt'), 'w') as f:
                for af in f1:
                    f.write(str(af)+'\n')


            print(f"Test time: {time.time()-timetest:.2f} F1: {f1[-1]:.4f} BestF1: {bestf1:.4f}")
            model.train()
            predictor.train()

    print(delg)

def exp_all(hidden_features = 128, savePATH = 'D:\\iCallasm\\all'):
    dataset = iCallds.iCallds()
    #g, rel_names, num_rels = init_nor_palm() #init_r_one()
    rel_names = [('code', 'code2code_edges', 'code'),
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
                 #('code', 'GT_edges', 'code'),
                 #('code', 'GT_F_edges', 'code')
                 ]

    in_features = {'code':40*128+2, 'data': 1}
    #in_features[]
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = Model(in_features, hidden_features, hidden_features, rel_names, dropout = 0.2)
    predictor = LinkPredictor(hidden_features*2, hidden_features, 1, 3, 0)
    model, predictor = map(lambda x: x.to(device), (model, predictor))
    opt = th.optim.Adam(model.parameters())
    model.float()
    predictor.float()
    bestf1 = 0
    f1 = []
    if os.path.exists(os.path.join(savePATH, 'predictor.checkpoint')):
        model.load_state_dict(th.load(os.path.join(savePATH, 'model.checkpoint')))
        predictor.load_state_dict(th.load(os.path.join(savePATH, 'predictor.checkpoint')))
        with open(os.path.join(savePATH, 'bestf1.txt'), 'r') as f:
            bestf1 = float(f.read())
        with open(os.path.join(savePATH, 'f1s.txt'), 'r') as f:
            slist = f.read().split('\n')
            slist.remove('')
            f1 = [float(i) for i in slist]


    model.train()
    predictor.train()
    i = -1
    timep = []
    delg = []
    timep.append(time.time())
    trains = [0, int(dataset.__len__()*0.8)]
    valids = [int(dataset.__len__()*0.8), int(dataset.__len__()*0.9)]
    tests = [int(dataset.__len__()*0.9), dataset.__len__()]
    metric = torchmetrics.F1Score()
    for epoch in range(trains[1]*epochs):
        i = random.randrange(trains[0], trains[1])
        if i in skips:
            continue
        #i=1095
        g, glabels = dataset[i]
        g = g.to(device)
        print(f'{i} Code: {g.num_nodes("code")}, Data: {g.num_nodes("data")}, Edges: {g.num_edges()}, GT: {glabels["GT_edges"].shape[1]}')
        if glabels['GT_edges'].shape[1] == 0:
            delg.append(i)
            print(f'Skipping {i}')
            continue
        #glabel = th.cat((th.ones(glabels['GT_edges'].shape[1]),th.zeros(glabels['GT_F_edges'].shape[1])),0)
        node_features = {'code': g.nodes['code'].data['feat'].view(g.nodes['code'].data['feat'].shape[0],-1).float(),
                         'data': g.nodes['data'].data['feat'].view(g.nodes['data'].data['feat'].shape[0],-1).float()}
        pred = model(g, node_features)
        #loss = ((pred - glabel) ** 2).mean()


        edge = glabels['GT_edges']
        pos_out = predictor(th.cat((pred['code'][edge[0]],pred['code'][edge[1]]),dim=1))#(pred['code'][edge[0]], pred['code'][edge[1]])
        pos_loss = -th.log(pos_out + 1e-15).mean()

        edge = glabels['GT_F_edges']
        neg_out = predictor(th.cat((pred['code'][edge[0]],pred['code'][edge[1]]),dim=1))#(pred['code'][edge[0]], pred['code'][edge[1]])
        neg_loss = -th.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        timep.append(time.time())
        print(f'Total time: {timep[-1]-timep[0]:.2f}, time: {timep[-1]- timep[-2]:.2f}')
        print(f'{epoch} ({epoch/(3777*epochs):.2%}): {loss.item()}')
        pos_loss = None
        neg_loss = None
        loss = None
        g = None
        edge = None
        pos_out = None
        neg_out = None
        neg_loss = None
        loss = None
        if (epoch+1)%4000 == 0:
            preds = []
            targets = []
            model.eval()
            predictor.eval()
            timetest = time.time()
            for i in range(valids[0], valids[1]):
                if i in skips:
                    continue

                g, glabels = dataset[i]
                g = g.to(device)
                if glabels['GT_edges'].shape[1] == 0:
                    delg.append(i)
                    print(f'Skipping {i}')
                    continue
                node_features = {
                    'code': g.nodes['code'].data['feat'].view(g.nodes['code'].data['feat'].shape[0], -1).float(),
                    'data': g.nodes['data'].data['feat'].view(g.nodes['data'].data['feat'].shape[0], -1).float()}
                pred = model(g, node_features)

                edge = glabels['GT_edges']
                pos_out = predictor(th.cat((pred['code'][edge[0]], pred['code'][edge[1]]),
                                           dim=1))
                edge = glabels['GT_F_edges']
                neg_out = predictor(th.cat((pred['code'][edge[0]], pred['code'][edge[1]]),
                                           dim=1))
                preds+=pos_out.tolist()
                preds+=neg_out.tolist()
                targets+=[[1]]*pos_out.shape[0]
                targets+=[[0]]*neg_out.shape[0]

            f1.append(metric(th.tensor(preds), th.tensor(targets)).item())
            if bestf1 <f1[-1]:
                bestf1 = f1[-1]
                th.save(model.state_dict(), os.path.join(savePATH, 'model.checkpoint'))
                th.save(predictor.state_dict(), os.path.join(savePATH, 'predictor.checkpoint'))
                with open(os.path.join(savePATH, 'bestf1.txt'), 'w') as f:
                    f.write(str(bestf1))
            with open(os.path.join(savePATH, 'f1s.txt'), 'w') as f:
                for af in f1:
                    f.write(str(af)+'\n')


            print(f"Test time: {time.time()-timetest:.2f} F1: {f1[-1]:.4f} BestF1: {bestf1:.4f}")
            model.train()
            predictor.train()

    print(delg)

def exp_all_sumloss(hidden_features = 128, savePATH = 'D:\\iCallasm\\allsumloss'):
    dataset = iCallds.iCallds()
    #g, rel_names, num_rels = init_nor_palm() #init_r_one()
    rel_names = [('code', 'code2code_edges', 'code'),
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
                 #('code', 'GT_edges', 'code'),
                 #('code', 'GT_F_edges', 'code')
                 ]

    in_features = {'code':40*128+2, 'data': 1}
    #in_features[]
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = Model(in_features, hidden_features, hidden_features, rel_names, dropout = 0.2)
    predictor = LinkPredictor(hidden_features*2, hidden_features, 1, 3, 0)
    model, predictor = map(lambda x: x.to(device), (model, predictor))
    opt = th.optim.Adam(model.parameters())
    model.float()
    predictor.float()
    bestf1 = 0
    f1 = []
    if os.path.exists(os.path.join(savePATH, 'predictor.checkpoint')):
        model.load_state_dict(th.load(os.path.join(savePATH, 'model.checkpoint')))
        predictor.load_state_dict(th.load(os.path.join(savePATH, 'predictor.checkpoint')))
        with open(os.path.join(savePATH, 'bestf1.txt'), 'r') as f:
            bestf1 = float(f.read())
        with open(os.path.join(savePATH, 'f1s.txt'), 'r') as f:
            slist = f.read().split('\n')
            slist.remove('')
            f1 = [float(i) for i in slist]


    model.train()
    predictor.train()
    i = -1
    timep = []
    delg = []
    timep.append(time.time())
    trains = [0, int(dataset.__len__()*0.8)]
    valids = [int(dataset.__len__()*0.8), int(dataset.__len__()*0.9)]
    tests = [int(dataset.__len__()*0.9), dataset.__len__()]
    metric = torchmetrics.F1Score()
    for epoch in range(trains[1]*epochs):
        i = random.randrange(trains[0], trains[1])
        if i in skips:
            continue
        #i=1095
        g, glabels = dataset[i]
        g = g.to(device)
        print(f'{i} Code: {g.num_nodes("code")}, Data: {g.num_nodes("data")}, Edges: {g.num_edges()}, GT: {glabels["GT_edges"].shape[1]}')
        if glabels['GT_edges'].shape[1] == 0:
            delg.append(i)
            print(f'Skipping {i}')
            continue
        #glabel = th.cat((th.ones(glabels['GT_edges'].shape[1]),th.zeros(glabels['GT_F_edges'].shape[1])),0)
        node_features = {'code': g.nodes['code'].data['feat'].view(g.nodes['code'].data['feat'].shape[0],-1).float(),
                         'data': g.nodes['data'].data['feat'].view(g.nodes['data'].data['feat'].shape[0],-1).float()}
        pred = model(g, node_features)
        #loss = ((pred - glabel) ** 2).mean()


        edge = glabels['GT_edges']
        pos_out = predictor(th.cat((pred['code'][edge[0]],pred['code'][edge[1]]),dim=1))#(pred['code'][edge[0]], pred['code'][edge[1]])
        pos_loss = -th.log(pos_out + 1e-15).sum()

        edge = glabels['GT_F_edges']
        neg_out = predictor(th.cat((pred['code'][edge[0]],pred['code'][edge[1]]),dim=1))#(pred['code'][edge[0]], pred['code'][edge[1]])
        neg_loss = -th.log(1 - neg_out + 1e-15).sum()

        loss = pos_loss + neg_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        timep.append(time.time())
        print(f'Total time: {timep[-1]-timep[0]:.2f}, time: {timep[-1]- timep[-2]:.2f}')
        print(f'{epoch} ({epoch/(3777*epochs):.2%}): {loss.item()}')
        pos_loss = None
        neg_loss = None
        loss = None
        g = None
        edge = None
        pos_out = None
        neg_out = None
        neg_loss = None
        loss = None
        if (epoch+1)%4000 == 0:
            preds = []
            targets = []
            model.eval()
            predictor.eval()
            timetest = time.time()
            for i in range(valids[0], valids[1]):
                if i in skips:
                    continue

                g, glabels = dataset[i]
                g = g.to(device)
                if glabels['GT_edges'].shape[1] == 0:
                    delg.append(i)
                    print(f'Skipping {i}')
                    continue
                node_features = {
                    'code': g.nodes['code'].data['feat'].view(g.nodes['code'].data['feat'].shape[0], -1).float(),
                    'data': g.nodes['data'].data['feat'].view(g.nodes['data'].data['feat'].shape[0], -1).float()}
                pred = model(g, node_features)

                edge = glabels['GT_edges']
                pos_out = predictor(th.cat((pred['code'][edge[0]], pred['code'][edge[1]]),
                                           dim=1))
                edge = glabels['GT_F_edges']
                neg_out = predictor(th.cat((pred['code'][edge[0]], pred['code'][edge[1]]),
                                           dim=1))
                preds+=pos_out.tolist()
                preds+=neg_out.tolist()
                targets+=[[1]]*pos_out.shape[0]
                targets+=[[0]]*neg_out.shape[0]

            f1.append(metric(th.tensor(preds), th.tensor(targets)).item())
            if bestf1 <f1[-1]:
                bestf1 = f1[-1]
                th.save(model.state_dict(), os.path.join(savePATH, 'model.checkpoint'))
                th.save(predictor.state_dict(), os.path.join(savePATH, 'predictor.checkpoint'))
                with open(os.path.join(savePATH, 'bestf1.txt'), 'w') as f:
                    f.write(str(bestf1))
            with open(os.path.join(savePATH, 'f1s.txt'), 'w') as f:
                for af in f1:
                    f.write(str(af)+'\n')


            print(f"Test time: {time.time()-timetest:.2f} F1: {f1[-1]:.4f} BestF1: {bestf1:.4f}")
            model.train()
            predictor.train()

    print(delg)

if __name__ == "__main__":
    epochs = 6
    skips = [1095, 2768, 917, 218, 2677, 201, 160, 215, 611, ]
    #exp_all(hidden_features=128, savePATH = 'D:\\iCallasm\\all')
    #exp_nodata(hidden_features = 128, savePATH = 'D:\\iCallasm\\nodata')
    exp_all_sumloss()
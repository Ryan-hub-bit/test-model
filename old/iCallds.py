

import os, sys, dgl, pickle, random
import torch as th

from dgl.data import DGLDataset



class iCallds(DGLDataset):
    directory = 'D:\\iCallasm' #"E:\\iCallds"
    numgraph = 3777
    glabels = {}
    def __init__(self):
        super().__init__(name='iCalls')

    def process(self):
        return


    def __getitem__(self, i):
        #i = '0303db951e1cc967eb1086fafda39e97a6b33158bf57a6aef8f25b02d7a29103'
        #i = 0
        graphfile = os.path.join(self.directory, str(i)+'.rgraph')
        #if os.path.isfile(graphfile) and graphfile.endswith('.ngraph'):
        glist, glabel = dgl.data.utils.load_graphs(graphfile)

        funcaddrfile = os.path.join(self.directory, str(i) + '.funcaddr2')
        with open(funcaddrfile, 'rb') as fp:
            savefunc = pickle.load(fp)

        fcalllist = {}
        calllist = {}
        '''
        for i in range(len(glist[0].edges(etype='GT_edges')[0])):
            if glist[0].edges(etype='GT_edges')[0][i].item() not in calllist:
                calllist[glist[0].edges(etype='GT_edges')[0][i].item()] = []
                fcalllist[glist[0].edges(etype='GT_edges')[0][i].item()] = []
            calllist[glist[0].edges(etype='GT_edges')[0][i].item()].append(glist[0].edges(etype='GT_edges')[1][i].item())
        '''

        for i in range(len(glabel['GT_label'][0])):
            if glabel['GT_label'][0][i].item() not in calllist:
                calllist[glabel['GT_label'][0][i].item()] = []
                fcalllist[glabel['GT_label'][0][i].item()] = []
            calllist[glabel['GT_label'][0][i].item()].append(glabel['GT_label'][1][i].item())

        funcaddrs = list(savefunc.values())
        #GT_F_edges = []
        GT_F_edges_s = []
        GT_F_edges_d = []
        for key, value in calllist.items():
            for _ in range(len(value)):
                r = random.choice(funcaddrs)
                maxtry = 0
                while True:
                    if maxtry > 20:
                        break
                    elif r in calllist[key] or r in fcalllist[key]:
                        r = random.choice(funcaddrs)
                        maxtry += 1
                    else:
                        fcalllist[key].append(r)
                        GT_F_edges_s.append(key)
                        GT_F_edges_d.append(r)
                        break
        #glist[0].add_edges(GT_F_edges_s, GT_F_edges_d, etype='GT_F_edges')
        #glabel['GT_F_edges'] = th.stack([GT_F_edges_s, GT_F_edges_d])
        #self.glabels[i] = {'GT_edges': glabel['GT_label'], 'GT_F_edges': th.tensor([GT_F_edges_s, GT_F_edges_d])}
        return glist[0], {'GT_edges': glabel['GT_label'], 'GT_F_edges': th.tensor([GT_F_edges_s, GT_F_edges_d])}

    def __len__(self):
        return self.numgraph


import angr

import json, random

import os
import pickle, dgl
import matplotlib.pyplot as plt
import torch as th

import eval_utils as utils

fileext='.asm' #.ifcc.json
#directory = "E:\\Research\\binaries"
graphdir = "D:\\iCallasm"
maxasm = 40 #max instruction in each blocks to embedding

def printblockstat():
    numdict = {}
    maxfile = [0, '']
    for files in os.listdir(graphdir):
        filepath = os.path.join(graphdir, files)
        if os.path.isfile(filepath) and filepath.endswith(fileext):

            with open(filepath, 'rb') as fp:
                asms = pickle.load(fp)
    #            if filepath == 'D:\\iCallasm\\00c5f9af405078b00f9ca968b26b319e9d8928877c45aae2615b7f1e358d92da.asm':
    #                for _, block in asms.items():
    #                    if len(block)>75:
    #                        block

                for _, block in asms.items():
                    if len(block) not in numdict:
                        numdict[len(block)] = 1
                    else:
                        numdict[len(block)] += 1
                    if len(block) > maxfile[0]:
                        maxfile[0] = len(block)
                        maxfile[1] = filepath

    total = [0]*100
    for key, value in numdict.items():
        for i in range(key):
            total[i]+=value

    plt.plot(sorted(numdict.items()))
    plt.show()

    print(maxfile)

def buildgraph():
    counts = 0
    palmtree = utils.UsableTransformer(model_path="./palmtree/transformer.ep19", vocab_path="./palmtree/vocab")
    for files in os.listdir(graphdir):
        filepath = os.path.join(graphdir, files)
        if os.path.isfile(filepath) and filepath.endswith(fileext):
            if os.path.exists(filepath[: -len(fileext)] + ".funcaddr2"):
                counts+=1
                print("Skipping",counts, filepath)
                continue

            with open(filepath, 'rb') as fp:
                asms = pickle.load(fp)

            gfilepath = filepath[: -len(fileext)] + ".graph"
            glist, label_dict = dgl.data.utils.load_graphs(gfilepath)
            g = glist[0]

            #savefunc patch
            funcaddrfile = filepath[: -len(fileext)] + '.funcaddr'
            with open(funcaddrfile, 'rb') as fp:
                funcdic = pickle.load(fp)
            savefunc = {}
            for _, v in funcdic.items():
                if v is None:
                    continue
                a = (g.nodes['code'].data['addr'] == v).nonzero(as_tuple=True)[0]
                if a.shape[0] == 0:
                    continue
                r = a.item()
                savefunc[v] = r
            with open(funcaddrfile+'2', 'wb') as fp:
                pickle.dump(savefunc, fp)
            counts += 1
            print("Finished",counts, filepath)
            #not need in savefunc patch
"""
            asm_embedding = {}
            for addr, asm in asms.items():
                if len(asm)>maxasm:
                    asm = asm[:maxasm]
                asm += (maxasm-len(asm)) * ['nop']
                asm_embedding[addr] = palmtree.encode(asm)

            graph_data = {}
            for k in g.canonical_etypes:
                graph_data[k] = g.edges(etype=k)

            num_nodes_dict = {}
            for k in g.ntypes:
                num_nodes_dict[k] =len(g.nodes(k))
            ng = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)


            g.nodes['code'].data['asm'] = th.tensor([asm_embedding[o.item()].ravel() for o in g.nodes['code'].data['addr']])
            #minmax(addr, funcaddr), norm to 0,1
            def cleandata(t, mint = 1001, maxt = 1e8):
                tmp = min(t[t>=mint])
                t[t<mint] = tmp
                tmp = max(t[t<=maxt])
                t[t>maxt] = tmp

            #cleandata(g.nodes['data'].data['addr'])

            addr_min = min(g.nodes['code'].data['addr'].min(), g.nodes['code'].data['func_addr'].min(), g.nodes['data'].data['addr'].min())
            addr_max = max(g.nodes['code'].data['addr'].max(), g.nodes['code'].data['func_addr'].max(), g.nodes['data'].data['addr'].max())
            new_min, new_max = 0, 1

            g.nodes['code'].data['addr'] = (g.nodes['code'].data['addr'] - addr_min) / (addr_max - addr_min) * (new_max - new_min) + new_min
            g.nodes['code'].data['func_addr'] = (g.nodes['code'].data['func_addr'] - addr_min) / (addr_max - addr_min) * (new_max - new_min) + new_min
            g.nodes['data'].data['addr'] = (g.nodes['data'].data['addr'] - addr_min) / (addr_max - addr_min) * (new_max - new_min) + new_min

            ng.nodes['code'].data['feat'] = th.cat((g.nodes['code'].data['addr'].view(-1,1), g.nodes['code'].data['func_addr'].view(-1,1), g.nodes['code'].data['asm'].view(-1,maxasm*128)),1)
            ng.nodes['data'].data['feat'] = g.nodes['data'].data['addr']

            dgl.data.utils.save_graphs(filepath[: -len(fileext)] + ".ngraph", [ng])
            counts+=1
            print("Saved", counts, filepath)
"""

def addrevedges():
    fileext = '.ngraph'
    counts = 0
    for files in os.listdir(graphdir):
        filepath = os.path.join(graphdir, files)
        if os.path.isfile(filepath) and filepath.endswith(fileext):
            if os.path.exists(filepath[: -len(fileext)] + ".rgraph"):
                counts+=1
                print("Skipping",counts, filepath)
                continue

            #gfilepath = filepath[: -len(fileext)] + ".graph"
            glist, label_dict = dgl.data.utils.load_graphs(filepath)
            g = glist[0]

            eg = dgl.edge_type_subgraph(g, [('code', 'code2code_edges', 'code'),
                 ('code', 'codecall_edges', 'code'),
                 ('code', 'codexrefcode_edges', 'code'),
                 ('code', 'codexrefdata_edges', 'data'),
                 ('data', 'dataxrefcode_edges', 'code'),
                 ('data', 'dataxrefdata_edges', 'data')])
            AddReverse =  dgl.transforms.AddReverse(sym_new_etype=True)
            eg = AddReverse(eg)
            graph_labels = {"GT_label": th.stack(g.edges(etype='GT_edges'))}

            dgl.data.utils.save_graphs(filepath[: -len(fileext)] + ".rgraph", [eg], graph_labels)
            counts += 1
            print("Finished",counts, filepath)


if __name__ == "__main__":
    #printblockstat()
    #buildgraph()
    addrevedges()
import os
import json

indirectcall_file = "/home/isec/Desktop/model/Indirectcall_addr.txt"
directcall_file = "/home/isec/Desktop/model/directcall_addr.txt"
sharelib = "/home/isec/Desktop/model/sharelib.txt"
idtoaddr_file = "/home/isec/Documents/angr_experiment_1/dir/idtoaddr.json"

idtoaddr =None

res = []
with open(idtoaddr_file, "r") as f:
    idtoaddrjson = f.read()
    idtoaddr = json.loads(idtoaddrjson)
with open(indirectcall_file, "r") as f:
    predid = f.read().splitlines()
    print(len(predid))
    predidset = list(set(predid))
    print(len(predidset))
    for id in predidset:
        if str(id) in idtoaddr:
            addr = int(idtoaddr[str(id)])
            res.append(addr)

with open(directcall_file, "r") as f:
   directaddr = f.read().splitlines()
   for addr in directaddr:
       addr_int = int(addr)
       if addr_int not in res:
           res.append(addr_int)


with open(sharelib, "r") as f:
    addrs = f.read().splitlines()
    for addr in addrs:
        realaddr = int(addr)
        if realaddr not in res:
            res.append(realaddr)

print(f"len(res): {len(res)}")

target = "/home/isec/Desktop/model/target.txt"
with open(target, "w") as f:
    for item in res:
        f.write(str(item) +"\n")


import os
import json

o0path = "/home/isec/Documents/differentopdata/Reorganized_Dataset/O0/CHANGE_LOGS"
o1path = "/home/isec/Documents/differentopdata/Reorganized_Dataset/O1/CHANGE_LOGS"


o0dict = {}
o1itemtodelete =[]
for root, dir, logfiles in os.walk(o0path):
    for logfile in logfiles:
        f = os.path.join(root, logfile)
        with open(f, 'r') as file:
            content = file.read().split()
            sourcefile  = content[1]
            md5 = content[0]
            if sourcefile in o0dict:
                print("sourcefile in o0dict")
                print(sourcefile)
                o1itemtodelete.append(sourcefile)
            else:
                o0dict[sourcefile] = md5


o1dict = {}
o1itemtodelete = []
for root, dir, logfiles in os.walk(o1path):
    for logfile in logfiles:
        f = os.path.join(root,logfile)
        with open(f, 'r') as file:
            content = file.read().split()
            sourcefile  = content[1]
            md5 = content[0]
            if sourcefile in o1dict:
                print("sourcefile in o1dict")
                print(sourcefile)
                print(md5)
                print(o1dict[sourcefile])
                o1itemtodelete.append(sourcefile)
            else:
                o1dict[sourcefile] = md5

print(f"len(o0dict) ={len(o0dict)} , len(o1dict) = {len(o1dict)}")


# o0too1dict = {}
# cnt = 0
# for key,value in o0dict.items():
#     if key in o1dict:
#         value = value + "->" + o1dict[key]
#         cnt +=1
#         print(value)
#         o0too1dict[key] = value


# path = "/home/isec/Documents/differentopdata/Reorganized_Dataset/Opdict/o0too1dict.json"
# with open(path, 'w') as convert_file:
#      convert_file.write(json.dumps(o0too1dict))
# print(cnt)


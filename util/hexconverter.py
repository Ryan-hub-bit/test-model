

filepath = "/home/isec/Desktop/model/target.txt"
libcpath = "/home/isec/Desktop/model/libc.txt"
addressfilepath = "/home/isec/Desktop/model/hextargetwithoutbase.txt"
lines = open(filepath, "r").read().splitlines()
hexaddr = []

with open(filepath, "r") as f2:
    lines = f2.read().splitlines()
    for line in lines:
        hexaddr.append(hex(int(line)))
    with open(libcpath, "r") as f3:
        libcs = f3.read().splitlines()
        for libc in libcs:
            if libc not in hexaddr:
                hexaddr.append(libc)


with open(addressfilepath, "w") as f1:
    for addr in hexaddr:
        f1.write(addr + "\n")

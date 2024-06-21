import angr

# Path to your binary
binary_path = "/home/isec/Documents/angr_experiment_1/nu/libpcre2-8.so.0.10.4"
func_path = "/home/isec/Desktop/model/target.txt"
addrnotin = "/home/isec/Documents/angr_experiment_1/dir/addrnotin.txt"

func_addr_list = []
with open(func_path, "r") as f:
    func_addr_list = f.read().splitlines()
print(len(func_addr_list))
func_size = 0
# Load the binary
project = angr.Project(binary_path, auto_load_libs=False)

# Get the CFG (Control Flow Graph)
cfg = project.analyses.CFGFast()
total_func_size = 0
functions = cfg.kb.functions
funclist = []
for addr, func in functions.items():
    total_func_size += func.size
    if addr not in func_addr_list:
        funclist.append(func.name)

with open(addrnotin, 'w') as file:
    # Iterate over the items in the list
    for item in funclist:
        # Write each item to a new line in the file
        file.write(str(item) + '\n')




cnt = 0
# for addr, func in functions.items():
#     print(addr)
#     if str(addr) in func_addr_list:
#         func_size +=func.size
# Find the function using the address
for func_addr in func_addr_list:
    #print(int(func_addr,16))
    function = cfg.kb.functions.get(int(func_addr))

    if function:
        cnt +=1
        # Calculate the function size
        func_size += function.size

print(f"total func size:{total_func_size}")
print(f"func size:{func_size}")
print(f"cnt:{cnt}")
print(f"func_size/total func size:{func_size/total_func_size}")
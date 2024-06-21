import angr
import os

binary_dir = "/home/isec/Documents/differentopdata/Reorganized_Dataset/valid_binary_list"

# Initialize counters
numzerototen = 0
numtentotwenty = 0
numtwentytothirty = 0
numthirtytofourty = 0
numfourtytofifty = 0
numfiftytosixty = 0
numsixtytoseventy = 0
nummorethanseventy = 0
cnt = 0
i = 0
for root, dirs, files in os.walk(binary_dir):
    for file in files:
        print(i)
        print(cnt)
        i += 1
        binary = os.path.join(root, file)
        try:
            project = angr.Project(binary, auto_load_libs=False)
            cfg = project.analyses.CFGFast()
            for function_addr, function in cfg.kb.functions.items():
        #         for block in function.blocks:
        #             try:
        #                 num_instructions = block.instructions
        #                 # print(f"num_instructions: {num_instructions}")
        #                 if 0 < num_instructions < 10:
        #                     numzerototen += 1
        #                 elif 10 <= num_instructions < 20:
        #                     numtentotwenty += 1
        #                 elif 20 <= num_instructions < 30:
        #                     numtwentytothirty += 1
        #                 elif 30 <= num_instructions < 40:
        #                     numthirtytofourty += 1
        #                 elif 40 <= num_instructions < 50:
        #                     numfourtytofifty += 1
        #                 elif 50 <= num_instructions < 60:
        #                     numfiftytosixty += 1
        #                 elif 60 <= num_instructions < 70:
        #                     numsixtytoseventy += 1
        #                 else:
        #                     nummorethanseventy += 1
                cnt += 1
        except Exception as e:
            print(f"Failed to process {binary}: {e}")

print(f"funcnum:{cnt}")
# Print final counts
# print(f"numzerototen: {numzerototen}, numtentotwenty: {numtentotwenty}, numtwentytothirty: {numtwentytothirty}, "
    #   f"numthirtytofourty: {numthirtytofourty}, numfourtytofifty: {numfourtytofifty}, numfiftytosixty: {numfiftytosixty}, "
    #   f"numsixtytoseventy: {numsixtytoseventy}, nummorethanseventy: {nummorethanseventy}")

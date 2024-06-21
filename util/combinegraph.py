import shutil
import os
import random

def copy_and_rename(src_path, dest_path):
    # Copy the file

    # Rename the copied file
    old_name = src_path.split("/")[-1]
    # num = int(src_path.split("/")[-1].split(".")[0])
    # new_name = str(num + index) + "." + src_path.split("/")[-1].split(".")[1]
    new_path = f"{dest_path}/{old_name}"
    print(new_path)
    # shutil.move(f"{dest_path}/{old_name}", new_path)
    shutil.copy2(src_path, new_path)
    return new_path

# Example usage
source_folder = "/home/isec/Documents/differentopdata/Reorganized_Dataset/O3/valid_binary_list"
destination_folder = "/home/isec/Documents/attncall/binfolder"
for root, dirs, files in os.walk(source_folder):
    numbers = 0
    for file in files:
        if numbers < 100:
            numbers +=1
            abpath = os.path.join(root, file)
            new_file_path = copy_and_rename(abpath, destination_folder)
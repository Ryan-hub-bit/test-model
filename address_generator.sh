#!/bin/bash
# Specify the directory where executable files are located
# root_directory="/home/isec/Documents/model/nano"
#root_directory="/home/isec/Documents/test_data/NeuTyPro-O2-Set-1-01/binaries"
# root_directory="/home/isec/Documents/data/Reorganized_Dataset/O2/FINAL_BINARIES"
# json_directory="/home/isec/Documents/data/Reorganized_Dataset/O2/TGCFI_JSON/"
binary_directory="/home/isec/Documents/experiment_6/valid_binary_list"
json_directory="/home/isec/Documents/experiment_6/valid_json_list/"
# Specify the Python script you want to run
python_script="/home/isec/model/getAddress.py"

json_str=".tgcfi.json"
#declare -i cnt=0
readDir() {
  local dir=$1
  local files=$(ls "$dir")
  #echo $dir
  for file in $files; do
    local path="$dir/$file"
    #echo $path
    if [ -d "$path" ]; then
      readDir "$path"
    else
      local text_path="$path"
      if [ ! -s "$text_path" ]; then
        echo "$text_path is empty"
        continue
      fi
      local json_path="$json_directory$file$json_str"
      echo $json_path
      if [ -e "$json_path" ]; then
        echo "Running $python_script for $path"
        python3 "$python_script" "$path"
      fi
    fi
  done
}

readDir "$binary_directory"
# echo $cnt
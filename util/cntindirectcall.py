import json
import os

def count_keys_values(data):
    """
    Recursively counts the number of keys and values in a nested JSON object.
    """
    callrelationship = data["tg_targets"]
    value_count = 0
    for key, value in callrelationship.items():
        value_count += len(value)

    return len(callrelationship), value_count

json_dir = "/home/isec/Documents/differentopdata/Reorganized_Dataset/O3/valid_json_list"
totalKey = 0
totalValue = 0
# Load the JSON file
for root, dirs, files in os.walk(json_dir):
    for file in files:
        json_file = os.path.join(root,file)
        with open(json_file, 'r') as file:
            json_data = json.load(file)
            keynumber, valuenumber = count_keys_values(json_data)
            totalKey +=keynumber
            totalValue += valuenumber

print(f"Total number of keys: {totalKey}")
print(f"Total number of values: {totalValue}")

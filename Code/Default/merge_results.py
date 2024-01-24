import os
import pickle
import json
import sys

def read_and_merge_pkl_files(folder_path, output_json_file):
    merged_data = {}

    # Iterate over all files in the given folder
    for file in os.listdir(folder_path):
        if file.endswith('.pkl'):
            file_path = os.path.join(folder_path, file)

            # Read the .pkl file
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

                # Assuming the data is a tuple with the format (key, value)
                if isinstance(data, tuple) and len(data) == 2:
                    merged_data[data[0]] = data[1]

    # Write merged data to a JSON file
    with open(output_json_file, 'w') as json_file:
        json.dump(merged_data, json_file, indent=4)

    print(f'Merged data written to {output_json_file}')

project_name = sys.argv[1]
# Example usage
folder_path = f'crossvalidation/{project_name}'  # Replace with the path to your folder
output_json_file = f'{project_name}_merged_data.json'
read_and_merge_pkl_files(folder_path, output_json_file)

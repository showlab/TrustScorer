import json
import os

input_folder = "../output"
output_folder = "../../../results"
method_name = "trustmodel"

input_filename = method_name + "_ace_scores.json"
output_filename = input_filename.replace("ace_scores.json", "scores.json")

input_file_path = os.path.join(input_folder, input_filename)
output_file_path = os.path.join(output_folder, output_filename)

os.makedirs(output_folder, exist_ok=True)

with open(input_file_path, "r") as file:
    data = json.load(file)

modified_data = {key.replace("___", "_"): value for key, value in data.items() if "_neg_" not in key}

with open(output_file_path, "w", encoding='utf-8') as file:
    json.dump(modified_data, file, indent=4, ensure_ascii=False)

print(f"Saved to {output_file_path}")

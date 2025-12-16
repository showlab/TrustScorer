import json

input_file_path = '../TrustBench/ace_test.json'
output_file_path = '../results/ace_actions.json'

extracted_data = {}

with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

    for key, value in data.items():
        if 'flag' in value:
            for task_key, task_value in value['flag'].items():
                for subtask_key, subtask_value in task_value.items():
                    new_key = f"{key}_{task_key}_{subtask_key}"
                    extracted_data[new_key] = subtask_value

with open(output_file_path, 'w', encoding='utf-8') as outfile:
    json.dump(extracted_data, outfile, indent=4)

print("Data extraction complete.")
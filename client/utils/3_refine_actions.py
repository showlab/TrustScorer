import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

method_name = "prompting_gpt4"
# method_name = "trustmodel"

data1 = load_json('../results/ace_actions.json')
data2 = load_json(f'../results/{method_name}_scores.json')
save_path = f'../results/{method_name}_actions.json'

keys_match = set(data1.keys()) == set(data2.keys())

if not keys_match:
    print("***** data1 and data2 not match in terms of keys *****")
else:
    print("data1 and data2 match in terms of keys")
    data3 = {}
    for key in data1:
        value1 = data1[key]
        value2 = float(data2[key])

        if value2 <= 0.5:
            if value1 == 1:
                # The original action was correct but had a low score, so the GT action was adopted and recorded as 1.4.
                data3[key] = value1 + 0.4
            elif value1 == 0:
                # Since the original action was incorrect and received a low score, the GT action was used and annotated as 1.2.
                data3[key] = value1 + 1.2
        else:
            data3[key] = value1  # If none of the conditions are satisfied, the original value is retained.

    save_json(data3, save_path)
    print(f"data3 saved to {save_path}")

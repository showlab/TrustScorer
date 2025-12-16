import json

ace_actions_path = '../results/ace_actions.json'

method_name = 'prompting_gpt4'
# method_name = 'trustmodel'

method_json_path = f'../results/{method_name}_scores.json'
output_path = f'../results/{method_name}_TF.json'
acc_output_path = f'../results/{method_name}_ACC.txt'

# Load the content of the ace_actions.json file
with open(ace_actions_path, 'r') as file:
    ace_actions = json.load(file)

# Load the content of the {method_name}_scores.json file
with open(method_json_path, 'r') as file:
    method_values = json.load(file)

# Initialize counters for the statistics
total_keys_value_1 = 0
total_keys_value_0 = 0
total_T_value_1 = 0
total_T_value_0 = 0
total_T_results = 0

results = {}
for key, value in ace_actions.items():
    if key in method_values:
        method_value = float(method_values[key])
        if value == 1:
            results[key] = 'T' if method_value > 0.5 else 'F'
            total_keys_value_1 += 1
            if results[key] == 'T':
                total_T_value_1 += 1
        elif value == 0:
            results[key] = 'T' if method_value <= 0.5 else 'F'
            total_keys_value_0 += 1
            if results[key] == 'T':
                total_T_value_0 += 1

# Calculate the percentages based on the counts
total_results = len(results)
total_T_results = sum(1 for result in results.values() if result == 'T')

percentage_T_in_value_1 = "{:.2f}".format((total_T_value_1 / total_keys_value_1 * 100) if total_keys_value_1 else 0)
percentage_T_in_value_0 = "{:.2f}".format((total_T_value_0 / total_keys_value_0 * 100) if total_keys_value_0 else 0)
percentage_T_in_results = "{:.2f}".format((total_T_results / total_results * 100) if total_results else 0)

# Generate formatted output
formatted_output = (
    f"Percentage of T for keys with value 1 in ace_actions: {percentage_T_in_value_1}%\n"
    f"Percentage of T for keys with value 0 in ace_actions: {percentage_T_in_value_0}%\n"
    f"Percentage of T in results: {percentage_T_in_results}%\n"
    f"\n"
    f"Total T for value 1: {total_T_value_1}\n"
    f"Total keys for value 1: {total_keys_value_1}\n"
    f"Total T for value 0: {total_T_value_0}\n"
    f"Total keys for value 0: {total_keys_value_0}\n"
    f"Total T results: {total_T_results}\n"
    f"Total results: {total_results}\n"
)

print(formatted_output)

# Save the results to a new JSON file
with open(output_path, 'w', encoding='utf-8') as file:
    json.dump(results, file, indent=4)

# Save the formatted output, including counts, to a text file
with open(acc_output_path, 'w', encoding='utf-8') as file:
    file.write(formatted_output)

print(f"Results saved to {output_path}")
print(f"Accuracy results saved to {acc_output_path}")

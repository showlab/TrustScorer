import json
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = False

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22

method_name = 'prompting_gpt4'
# method_name = 'trustmodel'

with open('../results/ace_actions.json', 'r') as f:
    data1 = json.load(f)
with open(f'../results/{method_name}_scores.json', 'r') as f:
    data2 = json.load(f)

save_path = f'../results/{method_name}_histogram.pdf'

list_gt1 = []
list_gt0 = []
list1 = []
list0 = []

for key, value in data1.items():
    corresponding_value_in_data2 = float(data2[key])
    if value == 1:
        list_gt1.append(value)
        list1.append(corresponding_value_in_data2)
    elif value == 0:
        list_gt0.append(value)
        list0.append(corresponding_value_in_data2)

bins = 20
range = (0, 1)

plt.hist(list1, bins=bins, range=range, color='#00ced1', alpha=0.65, label='Predicted Scores for Correct Actions', rwidth=0.9)
plt.hist(list0, bins=bins, range=range, color='#ff8c69', alpha=0.65, label='Predicted Scores for Wrong Actions', rwidth=0.9)

plt.xlabel('Trust Score')
plt.ylabel('Number')
plt.ylim(1, 275)  # adjust accordingly
plt.legend(fontsize=18)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_color('gray')
plt.gca().spines['bottom'].set_color('gray')

plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

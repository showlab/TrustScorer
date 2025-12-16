import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import colors

file_path = '../TrustBench/ace_actions.json'
with open(file_path, 'r') as file:
    data = json.load(file)

task_ids = set(key.split('_task')[0] for key in data.keys())
max_subtasks = 0
for task_id in task_ids:
    subtasks = [key for key in data if key.startswith(task_id)]
    max_subtasks = max(max_subtasks, len(subtasks))

matrix = []
labels = []
for task_id in sorted(task_ids, key=lambda x: int(x.split('_')[0]), reverse=True):
    subtasks = [key for key in sorted(data.keys()) if key.startswith(task_id)]
    task_data = [data[key] for key in subtasks]
    labels.append(task_id.split('_')[0])
    task_data += [-1] * (max_subtasks - len(task_data))
    matrix.append(task_data)

matrix = np.array(matrix)
X, Y = np.meshgrid(range(matrix.shape[1] + 1), range(matrix.shape[0] + 1))

cmap = colors.ListedColormap(['white', 'orange', 'lightgreen'])
bounds = [-1.5, -0.5, 0.5, 1.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots()
c = ax.pcolormesh(X, Y, matrix, cmap=cmap, norm=norm, edgecolors='lightgray', linewidth=0.5)

ax.set_title('Original Actor-Critic Embodied (ACE) Agent', fontsize=9, color='black', fontname='Times New Roman', pad=4)

for i, label in enumerate(labels):
    ax.text(-0.15, i + 0.5, label, va='center', ha='right', fontsize=7, color='gray', fontname='Times New Roman')

for j in range(len(matrix[0])):
    ax.text(j + 0.5, -0.3, str(j+1), va='top', ha='center', fontsize=7, color='gray', fontname='Times New Roman')

green_circles = 0
orange_circles = 0
for i, row in enumerate(matrix):
    color = 'orange' if 0 in row[:-1] else 'lightgreen'
    if color == 'orange':
        orange_circles += 1
    else:
        green_circles += 1
    ax.plot(len(matrix[0]) + 0.5, i + 0.5, 'o', color=color, markersize=5)

x_position = len(matrix[0]) + 0.55
offset = 0.3
ax.text(x_position - offset, -0.27, f"{green_circles}", va='top', ha='center', fontsize=8, color='lightgreen', fontstyle='italic', fontname='Times New Roman')
ax.text(x_position + offset, -0.27, f"{orange_circles}", va='top', ha='center', fontsize=8, color='orange', fontstyle='italic', fontname='Times New Roman')
ax.text(x_position, -0.27, "/", va='top', ha='center', fontsize=8, color='lightgray', fontstyle='italic', fontname='Times New Roman')

ax.set_xticks([])
ax.set_yticks([])

plt.savefig('ace_actions.pdf', bbox_inches='tight', pad_inches=0)

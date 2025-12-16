import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import colors

method_name = 'prompting_gpt4'
# method_name = 'trustmodel'

file_path = f'../results/{method_name}_actions.json'
with open(file_path, 'r') as file:
    data = json.load(file)

save_path = f'../results/{method_name}_actions.pdf'

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

fig, ax = plt.subplots(figsize=(8, 16))
c = ax.pcolormesh(X, Y, matrix, cmap=cmap, norm=norm, edgecolors='lightgray', linewidth=0.5)

ax.set_aspect('auto')  # 'auto' adjusts the cells to fit the figure dimensions

ax.set_title(f'Actor-Critic Embodied (ACE) Agent Enhanced by {method_name} Score Method', fontsize=9, color='black', fontname='Times New Roman', pad=4)

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


##################################################################################
################################## Extra Information #############################
##################################################################################
x_position = len(matrix[0]) + 0.5
y_position = -0.22
offset = 0.2
ax.text(x_position - offset, y_position, f"{green_circles}", va='top', ha='center', fontsize=6, color='lightgreen', fontstyle='italic', fontname='Times New Roman')
ax.text(x_position + offset, y_position, f"{orange_circles}", va='top', ha='center', fontsize=6, color='orange', fontstyle='italic', fontname='Times New Roman')
ax.text(x_position, y_position, "/", va='top', ha='center', fontsize=6, color='lightgray', fontstyle='italic', fontname='Times New Roman')

SR = green_circles / (green_circles + orange_circles) * 100
xx_position = len(matrix[0]) + 0.4
yy_position = -0.8
ax.text(xx_position, yy_position, "{:.1f}%↑".format(SR), va='top', ha='center', fontsize=8, color='black', fontstyle='italic', fontname='Times New Roman')

triangle_counts = []

for i, row in enumerate(matrix):
    triangle_count = 0
    for j, value in enumerate(row):
        if value == 1.4:
            ax.plot(j + 0.5, i + 0.5, 'x', color='#8E8BFE', markersize=4)
            triangle_count += 1
        elif value == 1.2:
            ax.plot(j + 0.5, i + 0.5, 'o', color='#8E8BFE', markersize=4)
            triangle_count += 1
    triangle_counts.append(triangle_count)

for i, count in enumerate(triangle_counts):
    ax.text(len(matrix[0]) + 1.2, i + 0.5, f"({count})", va='center', ha='left', fontsize=6, color='black', fontname='Times New Roman')

total_triangles = sum(triangle_counts)
ax.text(len(matrix[0]) + 1.2, -0.22, str(total_triangles), va='top', ha='left', fontsize=6, color='black', fontstyle='italic', fontname='Times New Roman')

total_actions = np.count_nonzero(matrix != -1)
percentage = (total_triangles / total_actions * 100) if total_actions > 0 else 0
ax.text(len(matrix[0]) + 1.15, -0.8, "{:.1f}%↓".format(percentage), va='top', ha='left', fontsize=8, color='black', fontstyle='italic', fontname='Times New Roman')

ax.set_xticks([])
ax.set_yticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

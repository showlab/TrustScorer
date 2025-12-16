import matplotlib.pyplot as plt
from matplotlib import rc

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = False

rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
plt.rcParams['mathtext.fontset'] = 'stix'


save_path = "../results/a_curve_cost_sr.pdf"

data = {
    "(a) AssistGUI": {"SR": 0.211, "Cost": 0.000},
    "(b) Perfect Score": {"SR": 1.000, "Cost": 0.330},
    "(c) Completely Trust": {"SR": 0.211, "Cost": 0.000},
    "(d) Completely Distrust": {"SR": 1.000, "Cost": 1.000},
    "(e) Random": {"SR": 0.366, "Cost": 0.511},
    "(f) Prompting GPT-4 w/o ex.": {"SR": 0.394, "Cost": 0.297},
    "(g) Prompting GPT-4 w/ ex.": {"SR": 0.394, "Cost": 0.330},
    "(h) Prompting Llama3 w/o ex.": {"SR": 0.479, "Cost": 0.485},
    "(i) Prompting Llama3 w/ ex.": {"SR": 0.507, "Cost": 0.528},
    "(j) Prompting GPT-4V w/o ex.": {"SR": 0.423, "Cost": 0.289},
    "(k) Prompting GPT-4V w/ ex.": {"SR": 0.479, "Cost": 0.315},
    "(l) Prompting LLaVA-NeXT w/o ex.": {"SR": 0.451, "Cost": 0.524},
    "(m) Prompting LLaVA-NeXT w/ ex.": {"SR": 0.465, "Cost": 0.423},
    "(n) TrustModel": {"SR": 0.592, "Cost": 0.474},
}

distinct_colors = [
    "#FF0000",  # 0 Red
    "#00FF00",  # 1 Green
    "#0000FF",  # 2 Blue
    "#FFFF00",  # 3 Yellow
    "#FF00FF",  # 4 Pink
    "#00FFFF",  # 5 Cyan
    "#800080",  # 6 Purple
    "#FFA500",  # 7 Orange
    "#008000",  # 8 Dark Green
    "#030303",  # 9 Black
    "#A52A2A",  # 10 Maroon
    "#DEB887",  # 11 Seed Color
    "#708090",  # 12 Slate Gray
    "#6B8E23",  # 13 Olive Green
]

colors = {
    "(a) AssistGUI": distinct_colors[9],
    "(b) Perfect Score": distinct_colors[1],
    "(c) Completely Trust": distinct_colors[8],
    "(d) Completely Distrust": distinct_colors[7],
    "(e) Random": distinct_colors[6],
    "(f) Prompting GPT-4 w/o ex.": distinct_colors[2],
    "(g) Prompting GPT-4 w/ ex.": distinct_colors[5],
    "(h) Prompting Llama3 w/o ex.": distinct_colors[11],
    "(i) Prompting Llama3 w/ ex.": distinct_colors[10],
    "(j) Prompting GPT-4V w/o ex.": distinct_colors[4],
    "(k) Prompting GPT-4V w/ ex.": distinct_colors[3],
    "(l) Prompting LLaVA-NeXT w/o ex.": distinct_colors[12],
    "(m) Prompting LLaVA-NeXT w/ ex.": distinct_colors[13],
    "(n) TrustModel": distinct_colors[0],
}

markers = {
    "(a) AssistGUI": "s",
    "(b) Perfect Score": "*",
    "(c) Completely Trust": "^",
    "(d) Completely Distrust": "^",
    "(e) Random": "^",
    "(f) Prompting GPT-4 w/o ex.": "o",
    "(g) Prompting GPT-4 w/ ex.": "o",
    "(h) Prompting Llama3 w/o ex.": "o",
    "(i) Prompting Llama3 w/ ex.": "o",
    "(j) Prompting GPT-4V w/o ex.": "o",
    "(k) Prompting GPT-4V w/ ex.": "o",
    "(l) Prompting LLaVA-NeXT w/o ex.": "o",
    "(m) Prompting LLaVA-NeXT w/ ex.": "o",
    "(n) TrustModel": "v",
}

fig, ax = plt.subplots()

for method, metrics in data.items():
    ax.scatter(metrics["Cost"]*100, metrics["SR"]*100, label=method, color=colors[method], marker=markers[method])

handles1, labels1 = [], []
handles2, labels2 = [], []

for i, (method, metrics) in enumerate(data.items()):
    marker = ax.scatter(metrics["Cost"]*100, metrics["SR"]*100, label=method, color=colors[method], marker=markers[method])
    if i < 8:
        handles1.append(marker)
        labels1.append(method)
    else:
        handles2.append(marker)
        labels2.append(method)

ax.set_xlabel("Manpower Cost (%)", fontsize=12, fontname='Times New Roman')
ax.set_ylabel("Task Success Rate (%)", fontsize=12, fontname='Times New Roman')

ax.set_ylim(18, 102)
ax.set_xlim(-5, 102)

legend1 = ax.legend(handles1, labels1, fontsize=9, loc='center left', bbox_to_anchor=(0.01, 0.72))
legend2 = ax.legend(handles2, labels2, fontsize=9, loc='center right', bbox_to_anchor=(0.99, 0.79))
ax.add_artist(legend1)

plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02)

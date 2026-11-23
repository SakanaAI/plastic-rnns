import glob
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from matplotlib.lines import Line2D

sns.set_palette("hls")
sns.set_style('darkgrid')

model_order = ["RNN", "PlasticRNN", "PlasticRNNOja", "LSTM", "PlasticLSTM", "PlasticLSTMSimple", "PlasticLSTMRetroactive", "CTM", "PlasticCTM", "PlasticCTMNoNLMs", "FW", "GDN", "HyperLSTM", "STPN"]

model_display_names = {
	"RNN": "RNN",
	"PlasticRNN": "Plastic RNN",
	"PlasticRNNOja": "Plastic RNN\n(Oja)",
	"LSTM": "LSTM",
	"PlasticLSTM": "Plastic LSTM\n(No Mod.)",
	"PlasticLSTMSimple": "Plastic LSTM\n(Simple Mod.)",
	"PlasticLSTMRetroactive": "Plastic LSTM\n(Retro. Mod.)",
	"CTM": "CTM",
	"PlasticCTM": "Plastic CTM",
	"PlasticCTMNoNLMs": "Plastic CTM (No NLMs)",
	"PlasticCTMNoDynamicRescaling": "Plastic CTM\n(No Dyn. Rescale)",
	"FW": "FW",
	"GDN": "GDN",
	"HyperLSTM": "HyperLSTM",
	"STPN": "STPN"
}

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "experiment_dirs.yaml")
with open(config_path) as f:
	experiment_config = yaml.safe_load(f)

# Normalize to lists (support both single dir string and list of dirs)
def normalize_dirs(dirs_config):
	normalized = {}
	for model_name, dirs in dirs_config.items():
		if isinstance(dirs, str):
			normalized[model_name] = [dirs]
		else:
			normalized[model_name] = dirs
	return normalized

model_dirs = normalize_dirs(experiment_config["maze_exploration"])

base_models = ["CTM", "LSTM", "RNN", "FW", "HyperLSTM", "STPN"]
color_palette = sns.color_palette("hls", len(base_models))
# Swap LSTM and HyperLSTM colors: LSTM gets blue (index 4), HyperLSTM gets yellow (index 1)
base_model_colors = {
	"CTM": color_palette[0],        # Red
	"LSTM": color_palette[4],       # Blue (swapped from yellow)
	"RNN": color_palette[2],        # Green
	"FW": color_palette[3],         # Cyan
	"GDN": "#666666",              # Gray
	"HyperLSTM": color_palette[1],  # Yellow (swapped from blue)
	"STPN": color_palette[5]        # Magenta
}

model_colors = {}
model_linestyles = {}

for model in model_order:
	if model == "PlasticLSTM":
		model_colors[model] = base_model_colors["LSTM"]
		model_linestyles[model] = "--"
	elif model == "PlasticLSTMSimple":
		model_colors[model] = base_model_colors["LSTM"]
		model_linestyles[model] = (0, (3, 1, 1, 1))  # densely dashdotted
	elif model == "PlasticLSTMRetroactive":
		model_colors[model] = base_model_colors["LSTM"]
		model_linestyles[model] = ":"  # dotted
	elif model == "PlasticCTM":
		model_colors[model] = base_model_colors["CTM"]
		model_linestyles[model] = "--"
	elif model == "PlasticCTMNoNLMs":
		model_colors[model] = base_model_colors["CTM"]
		model_linestyles[model] = (0, (3, 1, 1, 1))  # densely dashdotted
	elif model == "PlasticCTMNoDynamicRescaling":
		model_colors[model] = base_model_colors["CTM"]
		model_linestyles[model] = ":"  # dotted
	elif model == "PlasticRNN":
		model_colors[model] = base_model_colors["RNN"]
		model_linestyles[model] = "--"
	elif model == "PlasticRNNOja":
		model_colors[model] = base_model_colors["RNN"]
		model_linestyles[model] = (0, (3, 1, 1, 1))  # densely dashdotted
	else:
		model_colors[model] = base_model_colors[model]
		model_linestyles[model] = "-"

def get_eval_return_curves(run_dir):
	"""Get evaluation return curves from metrics.json"""
	metrics_path = f"{run_dir}/metrics.json"
	if not os.path.exists(metrics_path):
		return None

	with open(metrics_path) as f:
		metrics = json.load(f)

	if "eval_return" not in metrics:
		return None

	# Returns (steps, returns)
	steps = [x[0] for x in metrics["eval_return"]]
	returns = [x[1] for x in metrics["eval_return"]]
	return (steps, returns)

eval_return_curves = {}
for model_name, base_dirs in model_dirs.items():
	eval_return_curves[model_name] = []

	for base_dir in base_dirs:
		if os.path.exists(base_dir):
			run_ids = [d for d in os.listdir(base_dir) if d.isdigit()]
			for run_id in run_ids:
				run_dir = f"{base_dir}/{run_id}"
				curves = get_eval_return_curves(run_dir)
				if curves is not None:
					eval_return_curves[model_name].append(curves)

plt.figure(figsize=(16, 6))
legend_handles = []
step_ranges = []
for model_name in model_order:
	if model_name not in model_dirs:
		continue
	if eval_return_curves[model_name]:
		all_curves = eval_return_curves[model_name]

		# Find the minimum length across all seeds to handle variable-length curves
		min_length = min(len(curve[1]) for curve in all_curves)

		# Use steps from first curve, truncated to min_length, convert epochs to episodes
		steps = np.array(all_curves[0][0][:min_length]) * 512
		step_ranges.append((float(steps[0]), float(steps[-1])))

		returns_array = np.array([curve[1][:min_length] for curve in all_curves])
		mean_returns = np.mean(returns_array, axis=0)
		std_returns = np.std(returns_array, axis=0)

		plt.plot(steps, mean_returns, label=model_display_names[model_name], linewidth=2,
				color=model_colors[model_name], linestyle=model_linestyles[model_name], alpha=0.7)
		plt.fill_between(steps, mean_returns - std_returns, mean_returns + std_returns,
						color=model_colors[model_name], alpha=0.2)

		legend_handles.append(Line2D([0], [0], color=model_colors[model_name],
									linestyle=model_linestyles[model_name], linewidth=2,
									label=model_display_names[model_name]))

plt.xlabel("Episodes", fontsize=18)
plt.ylabel("Return", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()

legend = plt.legend(handles=legend_handles, fontsize=16, loc='upper center',
					bbox_to_anchor=(0.5, -0.18), ncol=7, frameon=False,
					labelspacing=1.0, columnspacing=1.5, handletextpad=0.7)
plt.gca().xaxis.get_offset_text().set_fontsize(16)
os.makedirs("outputs/analysis/maze_exploration", exist_ok=True)
plt.savefig("outputs/analysis/maze_exploration/eval_return.png", dpi=300, bbox_inches='tight')
plt.savefig("outputs/analysis/maze_exploration/eval_return.pdf", bbox_inches='tight')
if step_ranges:
	plt.xlim(min(start for start, _ in step_ranges), max(end for _, end in step_ranges))
	plt.savefig("outputs/analysis/maze_exploration/eval_return_tight_x.png", dpi=300, bbox_inches='tight')
	plt.savefig("outputs/analysis/maze_exploration/eval_return_tight_x.pdf", bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))
for model_name in model_order:
	if model_name not in model_dirs:
		continue
	if eval_return_curves[model_name]:
		all_curves = eval_return_curves[model_name]

		# Find the minimum length across all seeds to handle variable-length curves
		min_length = min(len(curve[1]) for curve in all_curves)

		# Use steps from first curve, truncated to min_length, convert epochs to episodes
		steps = np.array(all_curves[0][0][:min_length]) * 512

		returns_array = np.array([curve[1][:min_length] for curve in all_curves])
		mean_returns = np.mean(returns_array, axis=0)
		std_returns = np.std(returns_array, axis=0)

		plt.plot(steps, mean_returns, label=model_display_names[model_name], linewidth=2,
				color=model_colors[model_name], linestyle=model_linestyles[model_name], alpha=0.7)
		plt.fill_between(steps, mean_returns - std_returns, mean_returns + std_returns,
						color=model_colors[model_name], alpha=0.2)

plt.xlabel("Episodes", fontsize=18)
plt.ylabel("Return", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, alpha=0.3)

plt.gca().xaxis.get_offset_text().set_fontsize(16)
# Finalize the axes layout before adding the outside legend so the legend (and
# its column count) does not resize the plot
plt.tight_layout()

# Position the 13-entry legend to the right side in two columns. The column-major
# fill puts the first 7 labels (through Retro. Mod.) in column one.
plt.legend(handles=legend_handles, fontsize=16, loc='center left',
		   bbox_to_anchor=(1.02, 0.5), frameon=False, labelspacing=1.2, ncol=2, columnspacing=1.2)
plt.savefig("outputs/analysis/maze_exploration/eval_return_legend_right.png", dpi=300, bbox_inches='tight')
plt.savefig("outputs/analysis/maze_exploration/eval_return_legend_right.pdf", bbox_inches='tight')
if step_ranges:
	plt.xlim(min(start for start, _ in step_ranges), max(end for _, end in step_ranges))
	plt.savefig("outputs/analysis/maze_exploration/eval_return_legend_right_tight_x.png", dpi=300, bbox_inches='tight')
	plt.savefig("outputs/analysis/maze_exploration/eval_return_legend_right_tight_x.pdf", bbox_inches='tight')
plt.close()

print("Analysis complete! Plots saved to outputs/analysis/maze_exploration/")

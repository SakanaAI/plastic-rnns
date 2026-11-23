import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from matplotlib.lines import Line2D

sns.set_palette("hls")
sns.set_style('darkgrid')

model_order = ["PlasticCTM", "PlasticCTMHebbian", "PlasticCTMNoNLMs", "CTM", "GDN", "FW", "STPN", "HyperLSTM", "PlasticLSTMRetroactive", "PlasticLSTMSimple", "PlasticLSTM", "LSTM", "PlasticRNNOja", "PlasticRNN", "RNN"]

model_display_names = {
	"RNN": "RNN",
	"PlasticRNN": "Plastic RNN",
	"PlasticRNNOja": "Plastic RNN\n(Oja)",
	"LSTM": "LSTM",
	"PlasticLSTM": "Plastic LSTM\n(No Mod.)",
	"PlasticLSTMSimple": "Plastic LSTM\n(Simple Mod.)",
	"PlasticLSTMRetroactive": "Plastic LSTM\n(Retro. Mod.)",
	"CTM": "CTM",
	"PlasticCTM": "Plastic CTM\n(Ours)",
	"PlasticCTMNoNLMs": "Plastic CTM\n(No NLMs)",
	"PlasticCTMHebbian": "Plastic CTM\n(Hebbian)",
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

model_dirs = normalize_dirs(experiment_config["delay_sweep"])
seqlen_dirs = normalize_dirs(experiment_config["seqlen_sweep"])
copy_dim_dirs = normalize_dirs(experiment_config.get("copy_dim_sweep", {}))

delays = [0, 20, 40, 60]
num_seeds = 3

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
	elif model == "PlasticCTMHebbian":
		model_colors[model] = base_model_colors["CTM"]
		model_linestyles[model] = ":"  # dotted
	elif model == "PlasticRNN":
		model_colors[model] = base_model_colors["RNN"]
		model_linestyles[model] = "--"
	elif model == "PlasticRNNOja":
		model_colors[model] = base_model_colors["RNN"]
		model_linestyles[model] = ":"  # dotted
	else:
		model_colors[model] = base_model_colors[model]
		model_linestyles[model] = "-"

def get_test_loss(run_dir):
	with open(f"{run_dir}/metrics.json") as f:
		metrics = json.load(f)
	return metrics["test_loss"][-1][1]

def get_loss_curves(run_dir):
	"""Get train and eval loss curves from metrics.json"""
	with open(f"{run_dir}/metrics.json") as f:
		metrics = json.load(f)
	# Returns (steps, train_losses), (steps, eval_losses)
	train_steps = [x[0] for x in metrics["train_loss"]]
	train_losses = [x[1] for x in metrics["train_loss"]]
	eval_steps = [x[0] for x in metrics["eval_loss"]]
	eval_losses = [x[1] for x in metrics["eval_loss"]]
	return (train_steps, train_losses), (eval_steps, eval_losses)

def get_delay(run_dir):
	with open(f"{run_dir}/.hydra/config.yaml") as f:
		for line in f:
			if "copy_delay:" in line:
				return int(line.split(":")[1].strip())

def get_sequence_length(run_dir):
	with open(f"{run_dir}/.hydra/config.yaml") as f:
		for line in f:
			if "copy_sequence_length:" in line:
				return int(line.split(":")[1].strip())

def get_copy_dim(run_dir):
	with open(f"{run_dir}/.hydra/config.yaml") as f:
		for line in f:
			if "copy_dim:" in line:
				return int(line.split(":")[1].strip())

results = {}
for model_name, base_dirs in model_dirs.items():
	results[model_name] = {delay: [] for delay in delays}

	for base_dir in base_dirs:
		if os.path.exists(base_dir):
			run_ids = [d for d in os.listdir(base_dir) if d.isdigit()]
			for run_id in run_ids:
				run_dir = f"{base_dir}/{run_id}"
				if os.path.exists(f"{run_dir}/.hydra/config.yaml") and os.path.exists(f"{run_dir}/metrics.json"):
					delay = get_delay(run_dir)
					test_loss = get_test_loss(run_dir)
					results[model_name][delay].append(test_loss)

plt.figure(figsize=(8, 6))
legend_handles = []
for model_name in model_order:
	if model_name not in model_dirs:
		continue
	mean_losses = [np.mean(results[model_name][delay]) for delay in delays]
	std_losses = [np.std(results[model_name][delay]) for delay in delays]
	# Clip error bars so they don't go below zero
	lower_errors = [min(std, mean) for std, mean in zip(std_losses, mean_losses)]
	upper_errors = std_losses
	plt.errorbar(delays, mean_losses, yerr=[lower_errors, upper_errors], marker='o', capsize=5, linewidth=2, color=model_colors[model_name], linestyle=model_linestyles[model_name], alpha=0.7)
	legend_handles.append(Line2D([0], [0], color=model_colors[model_name], linestyle=model_linestyles[model_name], linewidth=2, label=model_display_names[model_name]))

plt.xlabel("Delay", fontsize=14)
plt.ylabel("Test Loss", fontsize=14)
plt.xticks(range(0, max(delays) + 1, 20), fontsize=12)
plt.yticks(fontsize=12)
plt.legend(handles=legend_handles, fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
os.makedirs("outputs/analysis/copy", exist_ok=True)
plt.savefig("outputs/analysis/copy/delay.png", dpi=300, bbox_inches='tight')
plt.savefig("outputs/analysis/copy/delay.pdf", bbox_inches='tight')
plt.close()

# Horizontal bar chart for delay sweep: models on y-axis, loss on x-axis, colors for each delay
fig, ax = plt.subplots(figsize=(11, 14))
delay_colors = {delay: sns.color_palette("viridis", len(delays))[i] for i, delay in enumerate(delays)}
bar_height = 0.15  # Reduced to fit multiple bars per model
models_with_delay_data = [m for m in model_order if m in model_dirs]
y_positions = np.arange(len(models_with_delay_data))

# Collect all bars with their losses and std, then assign zorder based on loss (smaller loss = higher zorder)
all_bars = []
for model_idx, model_name in enumerate(models_with_delay_data):
	for delay_idx, delay in enumerate(delays):
		if results[model_name][delay]:
			loss = np.mean(results[model_name][delay])
			std = np.std(results[model_name][delay])
			# Offset y-position based on delay index
			y_offset = (delay_idx - len(delays)/2 + 0.5) * bar_height
			all_bars.append((model_idx, delay, loss, std, y_offset))

# Sort by loss descending and assign zorder (index 0 = lowest zorder, last = highest zorder)
all_bars.sort(key=lambda x: x[2], reverse=True)
for zorder_idx, (model_idx, delay, loss, std, y_offset) in enumerate(all_bars):
	ax.barh(model_idx + y_offset, loss, bar_height, xerr=std, color=delay_colors[delay], 
			ecolor='black', alpha=1.0, zorder=zorder_idx, capsize=3)

ax.set_yticks(y_positions)
ax.set_yticklabels([model_display_names[m] for m in models_with_delay_data], fontsize=20)
for label in ax.get_yticklabels():
	label.set_linespacing(1.4)
ax.set_xlabel("Test Loss", fontsize=20)
ax.set_ylabel("Model", fontsize=20, labelpad=15)
ax.tick_params(axis='x', labelsize=16)
ax.set_xscale('log')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()

legend_handles = [plt.Rectangle((0, 0), 1, 1, color=delay_colors[delay], alpha=1.0) for delay in delays]
legend = ax.legend(legend_handles, [f"{delay}" for delay in delays], title="Delay", fontsize=18, title_fontsize=18, loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=len(delays), columnspacing=3.5, frameon=False)
plt.savefig("outputs/analysis/copy/delay_bar.png", dpi=300, bbox_inches='tight')
plt.savefig("outputs/analysis/copy/delay_bar.pdf", bbox_inches='tight')
plt.close()

# Sequence length sweep
seq_lengths = [5, 10, 20, 30, 40]

seqlen_results = {}
for model_name, base_dirs in seqlen_dirs.items():
	seqlen_results[model_name] = {length: [] for length in seq_lengths}

	for base_dir in base_dirs:
		if os.path.exists(base_dir):
			run_ids = [d for d in os.listdir(base_dir) if d.isdigit()]

			for run_id in run_ids:
				run_dir = f"{base_dir}/{run_id}"
				if os.path.exists(f"{run_dir}/.hydra/config.yaml") and os.path.exists(f"{run_dir}/metrics.json"):
					seq_length = get_sequence_length(run_dir)
					test_loss = get_test_loss(run_dir)
					seqlen_results[model_name][seq_length].append(test_loss)

plt.figure(figsize=(8, 6))
legend_handles = []
for model_name in model_order:
	if model_name not in seqlen_dirs:
		continue
	mean_losses = [np.mean(seqlen_results[model_name][length]) if seqlen_results[model_name][length] else np.nan for length in seq_lengths]
	std_losses = [np.std(seqlen_results[model_name][length]) if len(seqlen_results[model_name][length]) > 0 else 0 for length in seq_lengths]
	# Clip error bars so they don't go below zero
	lower_errors = [min(std, mean) if not np.isnan(mean) else 0 for std, mean in zip(std_losses, mean_losses)]
	upper_errors = std_losses
	plt.errorbar(seq_lengths, mean_losses, yerr=[lower_errors, upper_errors], marker='o', capsize=5, linewidth=2, color=model_colors[model_name], linestyle=model_linestyles[model_name], alpha=0.7)
	legend_handles.append(Line2D([0], [0], color=model_colors[model_name], linestyle=model_linestyles[model_name], linewidth=2, label=model_display_names[model_name]))

plt.xlabel("Sequence Length", fontsize=14)
plt.ylabel("Test Loss", fontsize=14)
plt.xticks(seq_lengths, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(handles=legend_handles, fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.savefig("outputs/analysis/copy/seq_len.png", dpi=300, bbox_inches='tight')
plt.savefig("outputs/analysis/copy/seq_len.pdf", bbox_inches='tight')
plt.close()

# Horizontal bar chart: models on y-axis, loss on x-axis, colors for each sequence length
fig, ax = plt.subplots(figsize=(11, 14))
seqlen_colors = {length: sns.color_palette("viridis", len(seq_lengths))[i] for i, length in enumerate(seq_lengths)}
bar_height = 0.15  # Reduced to fit multiple bars per model
models_with_data = [m for m in model_order if m in seqlen_dirs]
y_positions = np.arange(len(models_with_data))

# Collect all bars with their losses and std, then assign zorder based on loss (smaller loss = higher zorder)
all_bars = []
for model_idx, model_name in enumerate(models_with_data):
	for length_idx, length in enumerate(seq_lengths):
		if seqlen_results[model_name][length]:
			loss = np.mean(seqlen_results[model_name][length])
			std = np.std(seqlen_results[model_name][length])
			# Offset y-position based on sequence length index
			y_offset = (length_idx - len(seq_lengths)/2 + 0.5) * bar_height
			all_bars.append((model_idx, length, loss, std, y_offset))

# Sort by loss descending and assign zorder (index 0 = lowest zorder, last = highest zorder)
all_bars.sort(key=lambda x: x[2], reverse=True)
for zorder_idx, (model_idx, length, loss, std, y_offset) in enumerate(all_bars):
	ax.barh(model_idx + y_offset, loss, bar_height, xerr=std, color=seqlen_colors[length],
			ecolor='black', alpha=1.0, zorder=zorder_idx, capsize=3)

ax.set_yticks(y_positions)
ax.set_yticklabels([model_display_names[m] for m in models_with_data], fontsize=20)
for label in ax.get_yticklabels():
	label.set_linespacing(1.4)
ax.set_xlabel("Test Loss", fontsize=20)
ax.set_ylabel("Model", fontsize=20, labelpad=15)
ax.tick_params(axis='x', labelsize=16)
ax.set_xscale('log')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()

legend_handles = [plt.Rectangle((0, 0), 1, 1, color=seqlen_colors[length], alpha=1.0) for length in seq_lengths]
legend = ax.legend(legend_handles, [f"{length}" for length in seq_lengths], title="Sequence Length", fontsize=18, title_fontsize=18, loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=len(seq_lengths), columnspacing=3.5, frameon=False)
plt.savefig("outputs/analysis/copy/seq_len_bar.png", dpi=300, bbox_inches='tight')
plt.savefig("outputs/analysis/copy/seq_len_bar.pdf", bbox_inches='tight')
plt.close()

delay_loss_curves = {}
for model_name, base_dirs in model_dirs.items():
	delay_loss_curves[model_name] = {delay: {'train': [], 'eval': []} for delay in delays}

	for base_dir in base_dirs:
		if os.path.exists(base_dir):
			run_ids = [d for d in os.listdir(base_dir) if d.isdigit()]
			for run_id in run_ids:
				run_dir = f"{base_dir}/{run_id}"
				if os.path.exists(f"{run_dir}/.hydra/config.yaml") and os.path.exists(f"{run_dir}/metrics.json"):
					delay = get_delay(run_dir)
					(train_steps, train_losses), (eval_steps, eval_losses) = get_loss_curves(run_dir)
					delay_loss_curves[model_name][delay]['train'].append((train_steps, train_losses))
					delay_loss_curves[model_name][delay]['eval'].append((eval_steps, eval_losses))

for delay in delays:
	plt.figure(figsize=(10, 6))
	for model_name in model_order:
		if model_name not in model_dirs:
			continue
		if delay_loss_curves[model_name][delay]['train']:
			all_train_curves = delay_loss_curves[model_name][delay]['train']
			steps = all_train_curves[0][0]
			losses_array = np.array([curve[1] for curve in all_train_curves])
			mean_losses = np.mean(losses_array, axis=0)
			std_losses = np.std(losses_array, axis=0)
			plt.plot(steps, mean_losses, label=model_display_names[model_name], linewidth=2, color=model_colors[model_name], linestyle=model_linestyles[model_name], alpha=0.7)
			plt.fill_between(steps, np.maximum(mean_losses - std_losses, 0), mean_losses + std_losses, color=model_colors[model_name], alpha=0.2)
	plt.xlabel("Training Steps", fontsize=14)
	plt.ylabel("Train Loss", fontsize=14)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.legend(fontsize=12, loc='best')
	plt.grid(True, alpha=0.3)
	plt.savefig(f"outputs/analysis/copy/train_loss_delay_{delay}.png", dpi=300, bbox_inches='tight')
	plt.savefig(f"outputs/analysis/copy/train_loss_delay_{delay}.pdf", bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(10, 6))
	for model_name in model_order:
		if model_name not in model_dirs:
			continue
		if delay_loss_curves[model_name][delay]['eval']:
			all_eval_curves = delay_loss_curves[model_name][delay]['eval']
			steps = all_eval_curves[0][0]
			losses_array = np.array([curve[1] for curve in all_eval_curves])
			mean_losses = np.mean(losses_array, axis=0)
			std_losses = np.std(losses_array, axis=0)
			plt.plot(steps, mean_losses, label=model_display_names[model_name], linewidth=2, color=model_colors[model_name], linestyle=model_linestyles[model_name], alpha=0.7)
			plt.fill_between(steps, np.maximum(mean_losses - std_losses, 0), mean_losses + std_losses, color=model_colors[model_name], alpha=0.2)
	plt.xlabel("Training Steps", fontsize=14)
	plt.ylabel("Test Loss", fontsize=14)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.legend(fontsize=12, loc='best')
	plt.grid(True, alpha=0.3)
	plt.savefig(f"outputs/analysis/copy/test_loss_delay_{delay}.png", dpi=300, bbox_inches='tight')
	plt.savefig(f"outputs/analysis/copy/test_loss_delay_{delay}.pdf", bbox_inches='tight')
	plt.close()

seqlen_loss_curves = {}
for model_name, base_dirs in seqlen_dirs.items():
	seqlen_loss_curves[model_name] = {length: {'train': [], 'eval': []} for length in seq_lengths}

	for base_dir in base_dirs:
		if os.path.exists(base_dir):
			run_ids = [d for d in os.listdir(base_dir) if d.isdigit()]
			for run_id in run_ids:
				run_dir = f"{base_dir}/{run_id}"
				if os.path.exists(f"{run_dir}/.hydra/config.yaml") and os.path.exists(f"{run_dir}/metrics.json"):
					seq_length = get_sequence_length(run_dir)
					(train_steps, train_losses), (eval_steps, eval_losses) = get_loss_curves(run_dir)
					seqlen_loss_curves[model_name][seq_length]['train'].append((train_steps, train_losses))
					seqlen_loss_curves[model_name][seq_length]['eval'].append((eval_steps, eval_losses))

for seq_length in seq_lengths:
	plt.figure(figsize=(10, 6))
	for model_name in model_order:
		if model_name not in seqlen_dirs:
			continue
		if seqlen_loss_curves[model_name][seq_length]['train']:
			all_train_curves = seqlen_loss_curves[model_name][seq_length]['train']
			steps = all_train_curves[0][0]
			losses_array = np.array([curve[1] for curve in all_train_curves])
			mean_losses = np.mean(losses_array, axis=0)
			std_losses = np.std(losses_array, axis=0)
			plt.plot(steps, mean_losses, label=model_display_names[model_name], linewidth=2, color=model_colors[model_name], linestyle=model_linestyles[model_name], alpha=0.7)
			plt.fill_between(steps, np.maximum(mean_losses - std_losses, 0), mean_losses + std_losses, color=model_colors[model_name], alpha=0.2)
	plt.xlabel("Training Steps", fontsize=14)
	plt.ylabel("Train Loss", fontsize=14)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.legend(fontsize=12, loc='best')
	plt.grid(True, alpha=0.3)
	plt.savefig(f"outputs/analysis/copy/train_loss_seqlen_{seq_length}.png", dpi=300, bbox_inches='tight')
	plt.savefig(f"outputs/analysis/copy/train_loss_seqlen_{seq_length}.pdf", bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(10, 6))
	for model_name in model_order:
		if model_name not in seqlen_dirs:
			continue
		if seqlen_loss_curves[model_name][seq_length]['eval']:
			all_eval_curves = seqlen_loss_curves[model_name][seq_length]['eval']
			steps = all_eval_curves[0][0]
			losses_array = np.array([curve[1] for curve in all_eval_curves])
			mean_losses = np.mean(losses_array, axis=0)
			std_losses = np.std(losses_array, axis=0)
			plt.plot(steps, mean_losses, label=model_display_names[model_name], linewidth=2, color=model_colors[model_name], linestyle=model_linestyles[model_name], alpha=0.7)
			plt.fill_between(steps, np.maximum(mean_losses - std_losses, 0), mean_losses + std_losses, color=model_colors[model_name], alpha=0.2)
	plt.xlabel("Training Steps", fontsize=14)
	plt.ylabel("Test Loss", fontsize=14)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.legend(fontsize=12, loc='best')
	plt.grid(True, alpha=0.3)
	plt.savefig(f"outputs/analysis/copy/test_loss_seqlen_{seq_length}.png", dpi=300, bbox_inches='tight')
	plt.savefig(f"outputs/analysis/copy/test_loss_seqlen_{seq_length}.pdf", bbox_inches='tight')
	plt.close()

# Copy dimension sweep analysis
copy_dims = [5, 10]

copy_dim_results = {}
for model_name, base_dirs in copy_dim_dirs.items():
	copy_dim_results[model_name] = {dim: [] for dim in copy_dims}

	for base_dir in base_dirs:
		if os.path.exists(base_dir):
			run_ids = [d for d in os.listdir(base_dir) if d.isdigit()]

			for run_id in run_ids:
				run_dir = f"{base_dir}/{run_id}"
				if os.path.exists(f"{run_dir}/.hydra/config.yaml") and os.path.exists(f"{run_dir}/metrics.json"):
					copy_dim = get_copy_dim(run_dir)
					test_loss = get_test_loss(run_dir)
					copy_dim_results[model_name][copy_dim].append(test_loss)

if copy_dim_dirs:
	plt.figure(figsize=(8, 6))
	legend_handles = []
	for model_name in model_order:
		if model_name not in copy_dim_dirs:
			continue
		mean_losses = [np.mean(copy_dim_results[model_name][dim]) if copy_dim_results[model_name][dim] else np.nan for dim in copy_dims]
		std_losses = [np.std(copy_dim_results[model_name][dim]) if len(copy_dim_results[model_name][dim]) > 0 else 0 for dim in copy_dims]
		lower_errors = [min(std, mean) if not np.isnan(mean) else 0 for std, mean in zip(std_losses, mean_losses)]
		upper_errors = std_losses
		plt.errorbar(copy_dims, mean_losses, yerr=[lower_errors, upper_errors], marker='o', capsize=5, linewidth=2, color=model_colors[model_name], linestyle=model_linestyles[model_name], alpha=0.7)
		legend_handles.append(Line2D([0], [0], color=model_colors[model_name], linestyle=model_linestyles[model_name], linewidth=2, label=model_display_names[model_name]))

	plt.xlabel("Copy Dimension", fontsize=14)
	plt.ylabel("Test Loss", fontsize=14)
	plt.xticks(copy_dims, fontsize=12)
	plt.yticks(fontsize=12)
	plt.legend(handles=legend_handles, fontsize=12, loc='best')
	plt.grid(True, alpha=0.3)
	plt.savefig("outputs/analysis/copy/copy_dim.png", dpi=300, bbox_inches='tight')
	plt.savefig("outputs/analysis/copy/copy_dim.pdf", bbox_inches='tight')
	plt.close()

	# Horizontal bar chart for copy dim sweep: models on y-axis, loss on x-axis, colors for each copy dim
	fig, ax = plt.subplots(figsize=(10, 8))
	copy_dim_colors = {dim: sns.color_palette("viridis", len(copy_dims))[i] for i, dim in enumerate(copy_dims)}
	bar_height = 0.6
	models_with_copy_dim_data = [m for m in model_order if m in copy_dim_dirs]
	y_positions = np.arange(len(models_with_copy_dim_data))

	# Collect all bars with their losses, then assign zorder based on loss (smaller loss = higher zorder)
	all_bars = []
	for model_idx, model_name in enumerate(models_with_copy_dim_data):
		for dim in copy_dims:
			if copy_dim_results[model_name][dim]:
				loss = np.mean(copy_dim_results[model_name][dim])
				all_bars.append((model_idx, dim, loss))

	# Sort by loss descending and assign zorder (index 0 = lowest zorder, last = highest zorder)
	all_bars.sort(key=lambda x: x[2], reverse=True)
	for zorder_idx, (model_idx, dim, loss) in enumerate(all_bars):
		ax.barh(model_idx, loss, bar_height, color=copy_dim_colors[dim], alpha=1.0, zorder=zorder_idx)

	legend_handles = [plt.Rectangle((0, 0), 1, 1, color=copy_dim_colors[dim], alpha=1.0) for dim in copy_dims]
	legend = ax.legend(legend_handles, [f"{dim}" for dim in copy_dims], title="Copy Dim.", fontsize=16, title_fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(copy_dims), frameon=False)

	ax.set_yticks(y_positions)
	ax.set_yticklabels([model_display_names[m] for m in models_with_copy_dim_data], fontsize=20)
	ax.set_xlabel("Test Loss", fontsize=20)
	ax.set_ylabel("Model", fontsize=20, labelpad=15)
	ax.tick_params(axis='x', labelsize=16)
	ax.set_xscale('log')
	ax.grid(True, alpha=0.3, axis='x')
	plt.tight_layout()
	plt.savefig("outputs/analysis/copy/copy_dim_bar.png", dpi=300, bbox_inches='tight')
	plt.savefig("outputs/analysis/copy/copy_dim_bar.pdf", bbox_inches='tight')
	plt.close()

# Copy dimension loss curves
copy_dim_loss_curves = {}
for model_name, base_dirs in copy_dim_dirs.items():
	copy_dim_loss_curves[model_name] = {dim: {'train': [], 'eval': []} for dim in copy_dims}

	for base_dir in base_dirs:
		if os.path.exists(base_dir):
			run_ids = [d for d in os.listdir(base_dir) if d.isdigit()]
			for run_id in run_ids:
				run_dir = f"{base_dir}/{run_id}"
				if os.path.exists(f"{run_dir}/.hydra/config.yaml") and os.path.exists(f"{run_dir}/metrics.json"):
					copy_dim = get_copy_dim(run_dir)
					(train_steps, train_losses), (eval_steps, eval_losses) = get_loss_curves(run_dir)
					copy_dim_loss_curves[model_name][copy_dim]['train'].append((train_steps, train_losses))
					copy_dim_loss_curves[model_name][copy_dim]['eval'].append((eval_steps, eval_losses))

for copy_dim in copy_dims:
	if not copy_dim_dirs:
		continue
	plt.figure(figsize=(10, 6))
	for model_name in model_order:
		if model_name not in copy_dim_dirs:
			continue
		if copy_dim_loss_curves[model_name][copy_dim]['train']:
			all_train_curves = copy_dim_loss_curves[model_name][copy_dim]['train']
			steps = all_train_curves[0][0]
			losses_array = np.array([curve[1] for curve in all_train_curves])
			mean_losses = np.mean(losses_array, axis=0)
			std_losses = np.std(losses_array, axis=0)
			plt.plot(steps, mean_losses, label=model_display_names[model_name], linewidth=2, color=model_colors[model_name], linestyle=model_linestyles[model_name], alpha=0.7)
			plt.fill_between(steps, np.maximum(mean_losses - std_losses, 0), mean_losses + std_losses, color=model_colors[model_name], alpha=0.2)
	plt.xlabel("Training Steps", fontsize=14)
	plt.ylabel("Train Loss", fontsize=14)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.legend(fontsize=12, loc='best')
	plt.grid(True, alpha=0.3)
	plt.savefig(f"outputs/analysis/copy/train_loss_copy_dim_{copy_dim}.png", dpi=300, bbox_inches='tight')
	plt.savefig(f"outputs/analysis/copy/train_loss_copy_dim_{copy_dim}.pdf", bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(10, 6))
	for model_name in model_order:
		if model_name not in copy_dim_dirs:
			continue
		if copy_dim_loss_curves[model_name][copy_dim]['eval']:
			all_eval_curves = copy_dim_loss_curves[model_name][copy_dim]['eval']
			steps = all_eval_curves[0][0]
			losses_array = np.array([curve[1] for curve in all_eval_curves])
			mean_losses = np.mean(losses_array, axis=0)
			std_losses = np.std(losses_array, axis=0)
			plt.plot(steps, mean_losses, label=model_display_names[model_name], linewidth=2, color=model_colors[model_name], linestyle=model_linestyles[model_name], alpha=0.7)
			plt.fill_between(steps, np.maximum(mean_losses - std_losses, 0), mean_losses + std_losses, color=model_colors[model_name], alpha=0.2)
	plt.xlabel("Training Steps", fontsize=14)
	plt.ylabel("Test Loss", fontsize=14)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.legend(fontsize=12, loc='best')
	plt.grid(True, alpha=0.3)
	plt.savefig(f"outputs/analysis/copy/test_loss_copy_dim_{copy_dim}.png", dpi=300, bbox_inches='tight')
	plt.savefig(f"outputs/analysis/copy/test_loss_copy_dim_{copy_dim}.pdf", bbox_inches='tight')
	plt.close()

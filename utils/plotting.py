import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks")  # Globally set plotting style
plt.rcParams["axes.xmargin"] = 0  # Globally remove x-axis margins


def plot_metrics(metrics: dict):
	for name, step_value_pairs in metrics.items():
		plt.plot([pair[0] for pair in step_value_pairs], [pair[1] for pair in step_value_pairs], label=name)
		plt.xlabel("epoch")
		plt.ylabel(name)
		plt.savefig(f"{name}.png", bbox_inches="tight")
		plt.close()

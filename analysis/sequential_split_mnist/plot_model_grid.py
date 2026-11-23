import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.lines import Line2D

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Assemble the paper's PCA comparison from analysis output directories.")
for model in ("lstm", "plastic-lstm", "stpn", "ctm", "plastic-ctm"):
    parser.add_argument(f"--{model}", type=Path, required=True,
                        help="Directory containing this model's PCA trajectory PNGs")
parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/sequential_split_mnist"))
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

# Model name, plastic weights, first hidden state, second hidden state.
grid_data = [
    ["LSTM", "blank", None,
     args.lstm / "activation_h_trajectory_pca_3d.png",
     args.lstm / "activation_c_trajectory_pca_3d.png"],
    ["Plastic\nLSTM", "weights",
     args.plastic_lstm / "weight_trajectory_pca_3d.png",
     args.plastic_lstm / "activation_h_trajectory_pca_3d.png",
     args.plastic_lstm / "activation_c_trajectory_pca_3d.png"],
    ["STPN", "weights", args.stpn / "weight_trajectory_pca_3d.png",
     args.stpn / "activation_h_trajectory_pca_3d.png", None],
    ["CTM", "blank", None,
     args.ctm / "activation_z_trajectory_pca_3d.png",
     args.ctm / "activation_a_trajectory_pca_3d.png"],
    ["Plastic\nCTM", "weights",
     args.plastic_ctm / "weight_trajectory_pca_3d.png",
     args.plastic_ctm / "activation_z_trajectory_pca_3d.png",
     args.plastic_ctm / "activation_a_trajectory_pca_3d.png"],
]

fig = plt.figure(figsize=(28, 52))

left_margin = 0.02
col_widths = [0.08, 0.30, 0.30, 0.30]  # Model name, col1, col2, col3
col_spacing = 0.0001  # Small gap between columns
col_starts = [left_margin]
for w in col_widths[:-1]:
    col_starts.append(col_starts[-1] + w + col_spacing)

row_height = 0.15
row_spacing = 0.001  # Small gap between rows
top_start = 0.90
header_height = 0.03

for i, (model, col2_type, col2_path, col3_path, col4_path) in enumerate(grid_data):
    row_bottom = top_start - header_height - (i + 1) * row_height - i * row_spacing

    if col2_type == "weights" and col2_path:
        ax1 = fig.add_axes([col_starts[1], row_bottom, col_widths[1], row_height])
        img = imread(col2_path)
        ax1.imshow(img)
        ax1.axis('off')

    if col3_path:
        ax2 = fig.add_axes([col_starts[2], row_bottom, col_widths[2], row_height])
        img = imread(col3_path)
        ax2.imshow(img)
        ax2.axis('off')

    if col4_path:
        ax3 = fig.add_axes([col_starts[3], row_bottom, col_widths[3], row_height])
        img = imread(col4_path)
        ax3.imshow(img)
        ax3.axis('off')

legend_elements = [
    Line2D([0], [0], color='#2E86AB', linewidth=5, label='Train A'),      # Blue
    Line2D([0], [0], color='#A23B72', linewidth=5, label='Test A'),       # Magenta/Purple
    Line2D([0], [0], color='#F18F01', linewidth=5, label='Train B'),      # Orange
    Line2D([0], [0], color='#06A77D', linewidth=5, label='Test B'),       # Teal/Green
]
fig.legend(
    handles=legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.0),
    ncol=4,
    fontsize=48,
    frameon=False
)

plt.savefig(args.output_dir / 'model_comparison_grid.png', dpi=300, bbox_inches='tight')
print("Grid plot saved as 'model_comparison_grid.png'")
plt.close()


models = []
for model_name, col2_type, col2_path, col3_path, col4_path in grid_data:
    models.append({
        'name': model_name,
        'weights': col2_path if col2_type == "weights" else None,
        'hidden1': col3_path,
        'hidden2': col4_path
    })

fig_t = plt.figure(figsize=(52, 28))

left_margin_t = 0.02
row_widths_t = [0.08, 0.30, 0.30, 0.30]
row_spacing_t = 0.0001
row_starts_t = [0.02]  # Start from bottom
for w in row_widths_t[:-1]:
    row_starts_t.append(row_starts_t[-1] + w + row_spacing_t)

col_width_t = 0.18
col_spacing_t = 0.001
left_start_t = 0.10
header_width_t = 0.05

for j, model in enumerate(models):
    col_left = left_start_t + j * (col_width_t + col_spacing_t)

    if model['weights']:
        ax = fig_t.add_axes([col_left, row_starts_t[3], col_width_t, row_widths_t[3]])
        img = imread(model['weights'])
        ax.imshow(img)
        ax.axis('off')

    if model['hidden1']:
        ax = fig_t.add_axes([col_left, row_starts_t[2], col_width_t, row_widths_t[2]])
        img = imread(model['hidden1'])
        ax.imshow(img)
        ax.axis('off')

    if model['hidden2']:
        ax = fig_t.add_axes([col_left, row_starts_t[1], col_width_t, row_widths_t[1]])
        img = imread(model['hidden2'])
        ax.imshow(img)
        ax.axis('off')

fig_t.legend(
    handles=legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.0),
    ncol=4,
    fontsize=48,
    frameon=False
)

plt.savefig(args.output_dir / 'model_comparison_grid_transposed.png', dpi=300, bbox_inches='tight')
print("Transposed grid plot saved as 'model_comparison_grid_transposed.png'")
plt.close()

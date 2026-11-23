"""Visualize Sequential Split-MNIST weight and hidden-state trajectories.

Extract PCA/UMAP trajectories, align matching plastic/non-plastic hidden states
with Procrustes analysis, and generate static plots or animations. See README.md
for checkpoint examples; use --help for plotting options.
"""

import argparse
import csv
import os
import pickle
import sys

import imageio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import umap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from omegaconf import OmegaConf
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tasks.sequential_split_mnist import SequentialSplitMNISTModel, SequentialSplitMNISTDataset, SequentialSplitMNISTDataloader


MODEL_DISPLAY_NAMES = {
    "PlasticLSTM": "Plastic LSTM",
    "PlasticRNN": "Plastic RNN",
    "CTM": "CTM",
    "PlasticCTM": "Plastic CTM",
    "STPN": "STPN",
    "LSTM": "LSTM",
    "RNN": "RNN",
}


def get_phase_colors():
    """Use the same color for immediate and retention queries on Set A."""
    return {
        'train_a': '#1f77b4',  # Blue
        'test_a': '#aec7e8',   # Light blue
        'train_b': '#ff7f0e',  # Orange
        'test_b': '#ffbb78',   # Light orange
    }


def get_husl_colors(n_classes=10):
    """Get HUSL color palette for digit classes."""
    return sns.color_palette("husl", n_colors=n_classes)


def compute_phase_boundaries(n_ticks, n_exemplars_per_class=1, queries_per_loss=5,
                             internal_ticks=1, num_sequences=1):
    """Compute tick indices for each phase boundary."""
    n_classes = 5
    n_exemplars_total = n_classes * n_exemplars_per_class

    n_train_a = n_exemplars_total
    n_test_a_imm = queries_per_loss
    n_train_b = n_exemplars_total
    n_test_a_bwd = queries_per_loss
    n_test_b_fwd = queries_per_loss

    ticks_per_sequence = (n_train_a + n_test_a_imm + n_train_b + n_test_a_bwd + n_test_b_fwd) * internal_ticks

    phases = []
    for seq_idx in range(num_sequences):
        seq_offset = seq_idx * ticks_per_sequence

        start = seq_offset
        end = start + n_train_a * internal_ticks
        phases.append(('train_a', start, end))

        start = end
        end = start + n_test_a_imm * internal_ticks
        phases.append(('test_a', start, end))

        start = end
        end = start + n_train_b * internal_ticks
        phases.append(('train_b', start, end))

        start = end
        end = start + n_test_a_bwd * internal_ticks
        phases.append(('test_a', start, end))

        start = end
        end = start + n_test_b_fwd * internal_ticks
        phases.append(('test_b', start, end))

    return phases


def get_phase_for_tick(tick, phases):
    """Get phase name for a given tick index."""
    for phase_name, start, end in phases:
        if start <= tick < end:
            return phase_name
    return phases[-1][0]


def fit_umap_and_extract(weights_by_tick, checkpoint_path=None, reload=False, n_components=2):
    """Fit UMAP on all weights across all ticks and return embeddings.

    Args:
        weights_by_tick: List of weight arrays per tick
        checkpoint_path: Path to save/load UMAP embeddings (optional)
        reload: If True and checkpoint exists, reload from checkpoint
        n_components: Number of dimensions for UMAP (2 or 3)
    """
    if reload and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Reloading UMAP embeddings from: {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            embeddings_by_tick = pickle.load(f)
        return embeddings_by_tick

    n_ticks = len(weights_by_tick)

    all_weights = []
    tick_indices = []
    for tick_idx, weights in enumerate(weights_by_tick):
        weights_flat = weights.reshape(len(weights), -1)
        all_weights.append(weights_flat)
        tick_indices.extend([tick_idx] * len(weights))

    all_weights = np.vstack(all_weights)
    tick_indices = np.array(tick_indices)

    print(f"Fitting UMAP ({n_components}D) on {all_weights.shape[0]} samples...")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=50,
        min_dist=0.0,
        spread=0.5,
        metric='cosine',
        random_state=42
    )
    embedding = reducer.fit_transform(all_weights)

    embeddings_by_tick = []
    for tick_idx in range(n_ticks):
        mask = tick_indices == tick_idx
        embeddings_by_tick.append(embedding[mask])

    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(embeddings_by_tick, f)
        print(f"UMAP {n_components}D embeddings saved to: {checkpoint_path}")

    return embeddings_by_tick


def fit_pca_and_extract(weights_by_tick, checkpoint_path=None, reload=False, n_components=2):
    """Fit PCA on all weights across all ticks and return embeddings.

    Args:
        weights_by_tick: List of weight arrays per tick
        checkpoint_path: Path to save/load PCA embeddings (optional)
        reload: If True and checkpoint exists, reload from checkpoint
        n_components: Number of dimensions for PCA (2 or 3)
    """
    if reload and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Reloading PCA embeddings from: {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            embeddings_by_tick = pickle.load(f)
        return embeddings_by_tick

    n_ticks = len(weights_by_tick)

    all_weights = []
    tick_indices = []
    for tick_idx, weights in enumerate(weights_by_tick):
        weights_flat = weights.reshape(len(weights), -1)
        all_weights.append(weights_flat)
        tick_indices.extend([tick_idx] * len(weights))

    all_weights = np.vstack(all_weights)
    tick_indices = np.array(tick_indices)

    print(f"Fitting PCA ({n_components}D) on {all_weights.shape[0]} samples...")
    pca = PCA(n_components=n_components, random_state=42)
    embedding = pca.fit_transform(all_weights)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"PCA total variance explained: {pca.explained_variance_ratio_.sum():.4f}")

    embeddings_by_tick = []
    for tick_idx in range(n_ticks):
        mask = tick_indices == tick_idx
        embeddings_by_tick.append(embedding[mask])

    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(embeddings_by_tick, f)
        print(f"PCA {n_components}D embeddings saved to: {checkpoint_path}")

    return embeddings_by_tick


def procrustes_align_embeddings(reference_embeddings, target_embeddings):
    """Apply Procrustes alignment to align target embeddings to reference embeddings.

    Procrustes alignment finds the optimal rotation, translation, and scaling
    to align two sets of points in the same space.

    This function aligns the ENTIRE trajectory at once (all ticks together)
    to preserve the trajectory structure with a consistent transformation.

    Args:
        reference_embeddings: List of reference embedding arrays per tick [tick][n_samples, n_components]
        target_embeddings: List of target embedding arrays per tick [tick][n_samples, n_components]

    Returns:
        aligned_embeddings: List of aligned target embeddings per tick
        disparity: Overall disparity value (how well aligned)
    """
    # Concatenate all ticks together to align the entire trajectory at once
    ref_all = np.vstack(reference_embeddings)
    tgt_all = np.vstack(target_embeddings)

    # Returns: (mtx1_standardized, mtx2_transformed, disparity)
    ref_aligned, tgt_aligned, disparity = procrustes(ref_all, tgt_all)

    print(f"Procrustes alignment complete. Disparity: {disparity:.4f}")

    n_ticks = len(reference_embeddings)
    aligned_embeddings = []
    start_idx = 0
    for tick in range(n_ticks):
        n_samples = reference_embeddings[tick].shape[0]
        aligned_embeddings.append(tgt_aligned[start_idx:start_idx + n_samples])
        start_idx += n_samples

    return aligned_embeddings, disparity


def create_gif_by_phase(embeddings_by_tick, phases, output_path, fps=5):
    """Create animated GIF with points colored by current phase."""
    n_ticks = len(embeddings_by_tick)
    phase_colors = get_phase_colors()

    all_embeddings = np.vstack(embeddings_by_tick)
    x_min, x_max = all_embeddings[:, 0].min(), all_embeddings[:, 0].max()
    y_min, y_max = all_embeddings[:, 1].min(), all_embeddings[:, 1].max()
    x_range, y_range = x_max - x_min, y_max - y_min
    padding = 0.1
    x_min -= padding * x_range
    x_max += padding * x_range
    y_min -= padding * y_range
    y_max += padding * y_range

    frames = []
    for tick_idx in range(n_ticks):
        fig, ax = plt.subplots(figsize=(10, 10))

        emb = embeddings_by_tick[tick_idx]
        phase_name = get_phase_for_tick(tick_idx, phases)
        color = phase_colors[phase_name]

        ax.scatter(emb[:, 0], emb[:, 1], c=color, s=20, alpha=0.8, edgecolors='none')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Tick {tick_idx + 1}/{n_ticks} - {phase_name.replace('_', ' ').title()}", fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout(pad=0)

        fig.canvas.draw()
        try:
            buf = fig.canvas.buffer_rgba()
            image = np.asarray(buf)[:, :, :3]
        except AttributeError:
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = np.reshape(image, fig.canvas.get_width_height()[::-1] + (3,))

        frames.append(image)
        plt.close(fig)

    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"GIF saved: {output_path} ({len(frames)} frames)")


def create_gif_by_phase_3d(embeddings_by_tick, phases, output_path, fps=5):
    """Create animated 3D GIF with points colored by current phase."""
    n_ticks = len(embeddings_by_tick)
    phase_colors = get_phase_colors()

    all_embeddings = np.vstack(embeddings_by_tick)
    x_min, x_max = all_embeddings[:, 0].min(), all_embeddings[:, 0].max()
    y_min, y_max = all_embeddings[:, 1].min(), all_embeddings[:, 1].max()
    z_min, z_max = all_embeddings[:, 2].min(), all_embeddings[:, 2].max()
    x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min
    padding = 0.1
    x_min -= padding * x_range
    x_max += padding * x_range
    y_min -= padding * y_range
    y_max += padding * y_range
    z_min -= padding * z_range
    z_max += padding * z_range

    frames = []
    for tick_idx in range(n_ticks):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        emb = embeddings_by_tick[tick_idx]
        phase_name = get_phase_for_tick(tick_idx, phases)
        color = phase_colors[phase_name]

        ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=color, s=20, alpha=0.8, edgecolors='none')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_title(f"Tick {tick_idx + 1}/{n_ticks} - {phase_name.replace('_', ' ').title()}", fontsize=16)
        ax.set_xlabel('PC1' if 'pca' in output_path.lower() else 'UMAP1')
        ax.set_ylabel('PC2' if 'pca' in output_path.lower() else 'UMAP2')
        ax.set_zlabel('PC3' if 'pca' in output_path.lower() else 'UMAP3')

        plt.tight_layout(pad=0)

        fig.canvas.draw()
        try:
            buf = fig.canvas.buffer_rgba()
            image = np.asarray(buf)[:, :, :3]
        except AttributeError:
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = np.reshape(image, fig.canvas.get_width_height()[::-1] + (3,))

        frames.append(image)
        plt.close(fig)

    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"3D GIF saved: {output_path} ({len(frames)} frames)")


def create_trajectory_image_by_phase(embeddings_by_tick, phases, output_path,
                                     figsize=(10, 10), line_width=0.5, alpha=0.3):
    """Create static image showing weight trajectories with dots at each position.

    Uses distinct HLS colors for each phase, with opacity indicating temporal progress within phase.
    """
    n_ticks = len(embeddings_by_tick)
    n_samples = embeddings_by_tick[0].shape[0]

    phase_colors_hls = {
        'train_a': '#2E86AB',    # Blue
        'test_a': '#A23B72',     # Magenta/Purple
        'train_b': '#F18F01',    # Orange
        'test_b': '#06A77D',     # Teal/Green
    }

    all_embeddings = np.vstack(embeddings_by_tick)
    x_min, x_max = all_embeddings[:, 0].min(), all_embeddings[:, 0].max()
    y_min, y_max = all_embeddings[:, 1].min(), all_embeddings[:, 1].max()
    x_range, y_range = x_max - x_min, y_max - y_min
    padding = 0.1
    x_min -= padding * x_range
    x_max += padding * x_range
    y_min -= padding * y_range
    y_max += padding * y_range

    fig, ax = plt.subplots(figsize=figsize)

    initial_embeddings = embeddings_by_tick[0]
    start_center = np.mean(initial_embeddings, axis=0)

    phase_tick_ranges = {}
    for phase_name, start_tick, end_tick in phases:
        if phase_name not in phase_tick_ranges:
            phase_tick_ranges[phase_name] = []
        phase_tick_ranges[phase_name].append((start_tick, end_tick))

    for sample_idx in range(n_samples):
        trajectory = np.array([embeddings_by_tick[t][sample_idx] for t in range(n_ticks)])

        for t in range(n_ticks - 1):
            ax.plot(
                [trajectory[t, 0], trajectory[t + 1, 0]],
                [trajectory[t, 1], trajectory[t + 1, 1]],
                color='gray', alpha=alpha, linewidth=line_width, zorder=1
            )

        for t in range(n_ticks):
            phase_name = get_phase_for_tick(t, phases)
            color = phase_colors_hls[phase_name]

            phase_start, phase_end = None, None
            for start, end in phase_tick_ranges[phase_name]:
                if start <= t < end:
                    phase_start, phase_end = start, end
                    break

            if phase_start is not None and phase_end > phase_start:
                phase_progress = (t - phase_start) / (phase_end - phase_start)
                # Exponential mapping: starts at 0.25, increases slowly at first, then quickly
                exp_progress = phase_progress ** 3
                dot_alpha = 0.25 + 0.75 * exp_progress
            else:
                dot_alpha = 1.0

            ax.scatter(
                trajectory[t, 0], trajectory[t, 1],
                c=[color], s=30, alpha=dot_alpha, edgecolors='none', zorder=2
            )

    ax.scatter(start_center[0], start_center[1], marker='*', s=300, c='black',
               edgecolors='white', linewidths=1, zorder=10)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Phase trajectory image saved: {output_path}")


def create_trajectory_image_by_phase_3d(embeddings_by_tick, phases, output_path,
                                        figsize=(12, 10), line_width=0.5, alpha=0.3):
    """Create static 3D image showing weight trajectories with dots at each position.

    Uses distinct HLS colors for each phase, with opacity indicating temporal progress within phase.
    """
    n_ticks = len(embeddings_by_tick)
    n_samples = embeddings_by_tick[0].shape[0]

    phase_colors_hls = {
        'train_a': '#2E86AB',    # Blue
        'test_a': '#A23B72',     # Magenta/Purple
        'train_b': '#F18F01',    # Orange
        'test_b': '#06A77D',     # Teal/Green
    }

    all_embeddings = np.vstack(embeddings_by_tick)
    x_min, x_max = all_embeddings[:, 0].min(), all_embeddings[:, 0].max()
    y_min, y_max = all_embeddings[:, 1].min(), all_embeddings[:, 1].max()
    z_min, z_max = all_embeddings[:, 2].min(), all_embeddings[:, 2].max()
    x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min
    padding = 0.1
    x_min -= padding * x_range
    x_max += padding * x_range
    y_min -= padding * y_range
    y_max += padding * y_range
    z_min -= padding * z_range
    z_max += padding * z_range

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    initial_embeddings = embeddings_by_tick[0]
    start_center = np.mean(initial_embeddings, axis=0)

    phase_tick_ranges = {}
    for phase_name, start_tick, end_tick in phases:
        if phase_name not in phase_tick_ranges:
            phase_tick_ranges[phase_name] = []
        phase_tick_ranges[phase_name].append((start_tick, end_tick))

    for sample_idx in range(n_samples):
        trajectory = np.array([embeddings_by_tick[t][sample_idx] for t in range(n_ticks)])

        for t in range(n_ticks - 1):
            ax.plot(
                [trajectory[t, 0], trajectory[t + 1, 0]],
                [trajectory[t, 1], trajectory[t + 1, 1]],
                [trajectory[t, 2], trajectory[t + 1, 2]],
                color='gray', alpha=alpha, linewidth=line_width, zorder=1
            )

        for t in range(n_ticks):
            phase_name = get_phase_for_tick(t, phases)
            color = phase_colors_hls[phase_name]

            phase_start, phase_end = None, None
            for start, end in phase_tick_ranges[phase_name]:
                if start <= t < end:
                    phase_start, phase_end = start, end
                    break

            if phase_start is not None and phase_end > phase_start:
                phase_progress = (t - phase_start) / (phase_end - phase_start)
                # Exponential mapping: starts at 0.25, increases slowly at first, then quickly
                exp_progress = phase_progress ** 3
                dot_alpha = 0.25 + 0.75 * exp_progress
            else:
                dot_alpha = 1.0

            ax.scatter(
                trajectory[t, 0], trajectory[t, 1], trajectory[t, 2],
                c=[color], s=30, alpha=dot_alpha, edgecolors='none', zorder=2
            )

    ax.scatter(start_center[0], start_center[1], start_center[2], marker='*', s=300, c='black',
               edgecolors='white', linewidths=1, zorder=10)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.tick_params(labelbottom=False, labelleft=False, labeltop=False, labelright=False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"3D phase trajectory image saved: {output_path}")


def create_combined_trajectory_figure(model_data, output_path, figsize_per_model=5, line_width=0.5, alpha=0.3):
    """Create combined figure with trajectories for multiple models.

    Uses distinct HLS colors for each phase, with opacity indicating temporal progress within phase.

    Args:
        model_data: List of dicts with keys 'name', 'embeddings_by_tick', 'phases'
        output_path: Path to save the figure
        figsize_per_model: Size of each subplot
        line_width: Width of trajectory lines
        alpha: Alpha for lines
    """
    n_models = len(model_data)

    phase_colors_hls = {
        'train_a': '#2E86AB',    # Blue
        'test_a': '#A23B72',     # Magenta/Purple
        'train_b': '#F18F01',    # Orange
        'test_b': '#06A77D',     # Teal/Green
    }

    if n_models <= 3:
        n_cols = n_models
        n_rows = 1
    elif n_models <= 6:
        n_cols = 3
        n_rows = 2
    else:
        n_cols = 4
        n_rows = (n_models + 3) // 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_model * n_cols, figsize_per_model * n_rows))

    if n_models == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, data in enumerate(model_data):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        embeddings_by_tick = data['embeddings_by_tick']
        phases = data['phases']
        model_name = data['name']
        unique_name = data.get('unique_name', model_name)
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)

        if '_' in unique_name and unique_name != model_name:
            timestamp = unique_name.split('_', 1)[1]
            display_name = f"{display_name}\n({timestamp})"

        n_ticks = len(embeddings_by_tick)
        n_samples = embeddings_by_tick[0].shape[0]

        all_embeddings = np.vstack(embeddings_by_tick)
        x_min, x_max = all_embeddings[:, 0].min(), all_embeddings[:, 0].max()
        y_min, y_max = all_embeddings[:, 1].min(), all_embeddings[:, 1].max()
        x_range, y_range = x_max - x_min, y_max - y_min
        padding = 0.1
        x_min -= padding * x_range
        x_max += padding * x_range
        y_min -= padding * y_range
        y_max += padding * y_range

        initial_embeddings = embeddings_by_tick[0]
        start_center = np.mean(initial_embeddings, axis=0)

        phase_tick_ranges = {}
        for phase_name, start_tick, end_tick in phases:
            if phase_name not in phase_tick_ranges:
                phase_tick_ranges[phase_name] = []
            phase_tick_ranges[phase_name].append((start_tick, end_tick))

        for sample_idx in range(n_samples):
            trajectory = np.array([embeddings_by_tick[t][sample_idx] for t in range(n_ticks)])

            for t in range(n_ticks - 1):
                ax.plot(
                    [trajectory[t, 0], trajectory[t + 1, 0]],
                    [trajectory[t, 1], trajectory[t + 1, 1]],
                    color='gray', alpha=alpha, linewidth=line_width, zorder=1
                )

            for t in range(n_ticks):
                phase_name = get_phase_for_tick(t, phases)
                color = phase_colors_hls[phase_name]

                phase_start, phase_end = None, None
                for start, end in phase_tick_ranges[phase_name]:
                    if start <= t < end:
                        phase_start, phase_end = start, end
                        break

                if phase_start is not None and phase_end > phase_start:
                    phase_progress = (t - phase_start) / (phase_end - phase_start)
                    # Opacity ranges from 0.2 (start) to 1.0 (end)
                    dot_alpha = 0.2 + 0.8 * phase_progress
                else:
                    dot_alpha = 1.0

                ax.scatter(
                    trajectory[t, 0], trajectory[t, 1],
                    c=[color], s=20, alpha=dot_alpha, edgecolors='none', zorder=2
                )

        ax.scatter(start_center[0], start_center[1], marker='*', s=300, c='black',
                   edgecolors='white', linewidths=1, zorder=10)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(display_name, fontsize=14)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=phase_colors_hls['train_a'],
               markersize=10, label='Train A'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=phase_colors_hls['test_a'],
               markersize=10, label='Test A'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=phase_colors_hls['train_b'],
               markersize=10, label='Train B'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=phase_colors_hls['test_b'],
               markersize=10, label='Test B'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=12,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Combined trajectory figure saved: {output_path}")


def create_procrustes_comparison_figure(reference_data, target_data, aligned_data, output_path,
                                       reference_name, target_name, figsize_per_model=5,
                                       line_width=0.5, alpha=0.3):
    """Create side-by-side comparison of unaligned and aligned trajectories.

    Args:
        reference_data: Dict with 'embeddings_by_tick', 'phases', 'name'
        target_data: Dict with 'embeddings_by_tick', 'phases', 'name' (original unaligned)
        aligned_data: Dict with 'embeddings_by_tick', 'phases', 'name' (Procrustes aligned)
        output_path: Path to save figure
        reference_name: Display name for reference model
        target_name: Display name for target model
        figsize_per_model: Size of each subplot
        line_width: Width of trajectory lines
        alpha: Alpha for lines
    """
    phase_colors_hls = {
        'train_a': '#2E86AB',    # Blue
        'test_a': '#A23B72',     # Magenta/Purple
        'train_b': '#F18F01',    # Orange
        'test_b': '#06A77D',     # Teal/Green
    }

    fig, axes = plt.subplots(1, 3, figsize=(figsize_per_model * 3, figsize_per_model))

    data_list = [reference_data, target_data, aligned_data]
    titles = [f"{reference_name}\n(Reference)", f"{target_name}\n(Unaligned)", f"{target_name}\n(Procrustes Aligned)"]

    for ax, data, title in zip(axes, data_list, titles):
        embeddings_by_tick = data['embeddings_by_tick']
        phases = data['phases']

        n_ticks = len(embeddings_by_tick)
        n_samples = embeddings_by_tick[0].shape[0]

        all_embeddings = np.vstack(embeddings_by_tick)
        x_min, x_max = all_embeddings[:, 0].min(), all_embeddings[:, 0].max()
        y_min, y_max = all_embeddings[:, 1].min(), all_embeddings[:, 1].max()
        x_range, y_range = x_max - x_min, y_max - y_min
        padding = 0.1
        x_min -= padding * x_range
        x_max += padding * x_range
        y_min -= padding * y_range
        y_max += padding * y_range

        initial_embeddings = embeddings_by_tick[0]
        start_center = np.mean(initial_embeddings, axis=0)

        phase_tick_ranges = {}
        for phase_name, start_tick, end_tick in phases:
            if phase_name not in phase_tick_ranges:
                phase_tick_ranges[phase_name] = []
            phase_tick_ranges[phase_name].append((start_tick, end_tick))

        for sample_idx in range(n_samples):
            trajectory = np.array([embeddings_by_tick[t][sample_idx] for t in range(n_ticks)])

            for t in range(n_ticks - 1):
                ax.plot(
                    [trajectory[t, 0], trajectory[t + 1, 0]],
                    [trajectory[t, 1], trajectory[t + 1, 1]],
                    color='gray', alpha=alpha, linewidth=line_width, zorder=1
                )

            for t in range(n_ticks):
                phase_name = get_phase_for_tick(t, phases)
                color = phase_colors_hls[phase_name]

                phase_start, phase_end = None, None
                for start, end in phase_tick_ranges[phase_name]:
                    if start <= t < end:
                        phase_start, phase_end = start, end
                        break

                if phase_start is not None and phase_end > phase_start:
                    phase_progress = (t - phase_start) / (phase_end - phase_start)
                    dot_alpha = 0.2 + 0.8 * phase_progress
                else:
                    dot_alpha = 1.0

                ax.scatter(
                    trajectory[t, 0], trajectory[t, 1],
                    c=[color], s=20, alpha=dot_alpha, edgecolors='none', zorder=2
                )

        ax.scatter(start_center[0], start_center[1], marker='*', s=300, c='black',
                   edgecolors='white', linewidths=1, zorder=10)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=14)
        for spine in ax.spines.values():
            spine.set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=phase_colors_hls['train_a'],
               markersize=10, label='Train A'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=phase_colors_hls['test_a'],
               markersize=10, label='Test A'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=phase_colors_hls['train_b'],
               markersize=10, label='Train B'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=phase_colors_hls['test_b'],
               markersize=10, label='Test B'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=12,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Procrustes comparison figure saved: {output_path}")


def create_procrustes_comparison_figure_3d(reference_data, target_data, aligned_data, output_path,
                                           reference_name, target_name, figsize=(18, 6),
                                           line_width=0.5, alpha=0.3):
    """Create side-by-side comparison of unaligned and aligned trajectories in 3D.

    Args:
        reference_data: Dict with 'embeddings_by_tick', 'phases', 'name'
        target_data: Dict with 'embeddings_by_tick', 'phases', 'name' (original unaligned)
        aligned_data: Dict with 'embeddings_by_tick', 'phases', 'name' (Procrustes aligned)
        output_path: Path to save figure
        reference_name: Display name for reference model
        target_name: Display name for target model
        figsize: Overall figure size
        line_width: Width of trajectory lines
        alpha: Alpha for lines
    """
    phase_colors_hls = {
        'train_a': '#2E86AB',    # Blue
        'test_a': '#A23B72',     # Magenta/Purple
        'train_b': '#F18F01',    # Orange
        'test_b': '#06A77D',     # Teal/Green
    }

    fig = plt.figure(figsize=figsize)

    data_list = [reference_data, target_data, aligned_data]
    titles = [f"{reference_name}\n(Reference)", f"{target_name}\n(Unaligned)", f"{target_name}\n(Procrustes Aligned)"]

    for idx, (data, title) in enumerate(zip(data_list, titles)):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')

        embeddings_by_tick = data['embeddings_by_tick']
        phases = data['phases']

        n_ticks = len(embeddings_by_tick)
        n_samples = embeddings_by_tick[0].shape[0]

        all_embeddings = np.vstack(embeddings_by_tick)
        x_min, x_max = all_embeddings[:, 0].min(), all_embeddings[:, 0].max()
        y_min, y_max = all_embeddings[:, 1].min(), all_embeddings[:, 1].max()
        z_min, z_max = all_embeddings[:, 2].min(), all_embeddings[:, 2].max()
        x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min
        padding = 0.1
        x_min -= padding * x_range
        x_max += padding * x_range
        y_min -= padding * y_range
        y_max += padding * y_range
        z_min -= padding * z_range
        z_max += padding * z_range

        initial_embeddings = embeddings_by_tick[0]
        start_center = np.mean(initial_embeddings, axis=0)

        phase_tick_ranges = {}
        for phase_name, start_tick, end_tick in phases:
            if phase_name not in phase_tick_ranges:
                phase_tick_ranges[phase_name] = []
            phase_tick_ranges[phase_name].append((start_tick, end_tick))

        for sample_idx in range(n_samples):
            trajectory = np.array([embeddings_by_tick[t][sample_idx] for t in range(n_ticks)])

            for t in range(n_ticks - 1):
                ax.plot(
                    [trajectory[t, 0], trajectory[t + 1, 0]],
                    [trajectory[t, 1], trajectory[t + 1, 1]],
                    [trajectory[t, 2], trajectory[t + 1, 2]],
                    color='gray', alpha=alpha, linewidth=line_width, zorder=1
                )

            for t in range(n_ticks):
                phase_name = get_phase_for_tick(t, phases)
                color = phase_colors_hls[phase_name]

                phase_start, phase_end = None, None
                for start, end in phase_tick_ranges[phase_name]:
                    if start <= t < end:
                        phase_start, phase_end = start, end
                        break

                if phase_start is not None and phase_end > phase_start:
                    phase_progress = (t - phase_start) / (phase_end - phase_start)
                    dot_alpha = 0.2 + 0.8 * phase_progress
                else:
                    dot_alpha = 1.0

                ax.scatter(
                    trajectory[t, 0], trajectory[t, 1], trajectory[t, 2],
                    c=[color], s=20, alpha=dot_alpha, edgecolors='none', zorder=2
                )

        ax.scatter(start_center[0], start_center[1], start_center[2], marker='*', s=300, c='black',
                   edgecolors='white', linewidths=1, zorder=10)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.tick_params(labelbottom=False, labelleft=False, labeltop=False, labelright=False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"3D Procrustes comparison figure saved: {output_path}")


def infer_model_type_from_path(checkpoint_path):
    """Infer model type from checkpoint directory path."""
    path_parts = checkpoint_path.split('/')
    for part in path_parts:
        if 'PlasticLSTM' in part:
            return 'PlasticLSTM'
        elif 'PlasticRNN' in part:
            return 'PlasticRNN'
        elif 'PlasticCTM' in part:
            return 'PlasticCTM'
        elif 'CTM' in part:
            return 'CTM'
        elif 'STPN' in part:
            return 'STPN'
        elif 'LSTM' in part:
            return 'LSTM'
        elif 'RNN' in part:
            return 'RNN'
    return None


def extract_weights(model, dataloader, device, max_samples=100):
    """Extract plastic weights across the SplitMNIST sequence."""
    model.eval()

    rnn_module = model.rnn.recurrent_module
    model_type = type(rnn_module).__name__

    all_weights_by_tick = None
    total_collected = 0

    with torch.no_grad():
        for batch_idx, ((inputs, aux_inputs), targets) in enumerate(dataloader):
            if total_collected >= max_samples:
                break

            batch_size = min(inputs.size(0), max_samples - total_collected)
            inputs = inputs[:batch_size].to(device)
            aux_inputs = aux_inputs[:batch_size].to(device)

            print(f"  Processing batch {batch_idx}: {total_collected}/{max_samples} samples")

            B, T, C, H, W = inputs.shape

            inputs_t = inputs.transpose(0, 1).contiguous()
            aux_inputs_t = aux_inputs.transpose(0, 1).contiguous()

            inputs_flat = inputs_t.reshape(T * B, C, H, W)
            aux_inputs_flat = aux_inputs_t.reshape(T * B)

            label_embeddings = torch.nn.functional.one_hot(
                aux_inputs_flat, num_classes=model.num_classes + 1
            ).float()
            embeddings = model.backbone(inputs_flat, label_embeddings)
            embeddings = embeddings.view(T, B, -1)

            if all_weights_by_tick is None:
                all_weights_by_tick = [[] for _ in range(T)]

            state = model.rnn.get_initial_state()

            if len(state) >= 1 and state[0].size(0) == 1 and B > 1:
                state = tuple(s.expand(B, *s.shape[1:]).contiguous() if s is not None else None
                             for s in state)

            for t in range(T):
                input_t = embeddings[t]
                weights = _extract_weights_for_tick(model, rnn_module, model_type, state, B, device)
                all_weights_by_tick[t].append(weights)
                _, state = rnn_module(input_t, state)

            total_collected += batch_size

    for t in range(len(all_weights_by_tick)):
        all_weights_by_tick[t] = np.concatenate(all_weights_by_tick[t], axis=0)[:max_samples]

    return all_weights_by_tick


def extract_activations(model, dataloader, device, max_samples=100):
    """Extract activations across the SplitMNIST sequence.

    Returns:
        Dict mapping activation type (e.g., 'a', 'z', 'h', 'c') to list of arrays by tick
    """
    model.eval()

    rnn_module = model.rnn.recurrent_module
    model_type = type(rnn_module).__name__

    all_activations_by_type = None  # Will be {type: [tick_0, tick_1, ...]}
    total_collected = 0

    with torch.no_grad():
        for batch_idx, ((inputs, aux_inputs), targets) in enumerate(dataloader):
            if total_collected >= max_samples:
                break

            batch_size = min(inputs.size(0), max_samples - total_collected)
            inputs = inputs[:batch_size].to(device)
            aux_inputs = aux_inputs[:batch_size].to(device)

            print(f"  Processing batch {batch_idx}: {total_collected}/{max_samples} samples")

            B, T, C, H, W = inputs.shape

            inputs_t = inputs.transpose(0, 1).contiguous()
            aux_inputs_t = aux_inputs.transpose(0, 1).contiguous()

            inputs_flat = inputs_t.reshape(T * B, C, H, W)
            aux_inputs_flat = aux_inputs_t.reshape(T * B)

            label_embeddings = torch.nn.functional.one_hot(
                aux_inputs_flat, num_classes=model.num_classes + 1
            ).float()
            embeddings = model.backbone(inputs_flat, label_embeddings)
            embeddings = embeddings.view(T, B, -1)

            state = model.rnn.get_initial_state()

            if len(state) >= 1 and state[0].size(0) == 1 and B > 1:
                state = tuple(s.expand(B, *s.shape[1:]).contiguous() if s is not None else None
                             for s in state)

            for t in range(T):
                input_t = embeddings[t]
                activations_dict = _extract_activations_for_tick(model, rnn_module, model_type, state, B, device)

                if all_activations_by_type is None:
                    all_activations_by_type = {key: [[] for _ in range(T)] for key in activations_dict.keys()}

                for key, value in activations_dict.items():
                    all_activations_by_type[key][t].append(value)

                _, state = rnn_module(input_t, state)

            total_collected += batch_size

    for act_type in all_activations_by_type:
        for t in range(len(all_activations_by_type[act_type])):
            all_activations_by_type[act_type][t] = np.concatenate(
                all_activations_by_type[act_type][t], axis=0
            )[:max_samples]

    return all_activations_by_type


def _extract_activations_for_tick(model, rnn_module, model_type, state, batch_size, device):
    """Extract activations from state based on model type.

    Returns:
        Dict with separate activation arrays for each type:
        - CTM: {'a': state (output of synapses), 'z': activated_state (output of NLMs)}
        - LSTM: {'h': hidden state, 'c': cell state}
        - STPN: {'h': hidden state}
    """

    if 'ContinuousThoughtMachine' in model_type:
        # CTM: state[0] is activated_state (z), need to get state (a) from state_trace
        activations = {}
        if len(state) >= 2:
            activated_state = state[0]  # z [B, d_model]
            state_trace = state[1]  # [B, d_model, memory_length]
            state_a = state_trace[:, :, -1]  # [B, d_model]

            activations['a'] = state_a.cpu().numpy()
            activations['z'] = activated_state.cpu().numpy()
        else:
            dummy = torch.zeros(batch_size, 1).to(device).cpu().numpy()
            activations['a'] = dummy
            activations['z'] = dummy
        return activations

    elif 'STPN' in model_type:
        # STPN: extract h (state[0])
        activations = {}
        if len(state) >= 1:
            h = state[0]  # hidden state [B, d_model]
            activations['h'] = h.cpu().numpy()
        else:
            dummy = torch.zeros(batch_size, 1).to(device).cpu().numpy()
            activations['h'] = dummy
        return activations

    elif 'LSTM' in model_type or 'PlasticLSTM' in model_type:
        # LSTM: extract h and c
        activations = {}
        if len(state) >= 2:
            h = state[0]  # hidden state [B, d_model]
            c = state[1]  # cell state [B, d_model]
            activations['h'] = h.cpu().numpy()
            activations['c'] = c.cpu().numpy()
        else:
            dummy = torch.zeros(batch_size, 1).to(device).cpu().numpy()
            activations['h'] = dummy
            activations['c'] = dummy
        return activations

    else:
        # Default: just use first state element
        activations = {}
        if len(state) >= 1:
            activations['state'] = state[0].cpu().numpy()
        else:
            dummy = torch.zeros(batch_size, 1).to(device).cpu().numpy()
            activations['state'] = dummy
        return activations


def _extract_weights_for_tick(model, rnn_module, model_type, state, batch_size, device):
    """Extract weights from state based on model type."""

    if 'ContinuousThoughtMachine' in model_type:
        synapses = rnn_module.synapses
        weights = synapses.weights_fixed.unsqueeze(0).expand(batch_size, -1, -1)
        if rnn_module.plastic:
            if hasattr(synapses, 'alpha'):
                weights = weights + synapses.alpha * state[5]
            else:
                state_index = 3 if 'NoNLMs' in model_type else 4
                trace = state[state_index].view(batch_size, rnn_module.d_model, rnn_module.d_model)
                if synapses.adaptive_scaling:
                    trace_norm = torch.linalg.vector_norm(trace, dim=(-2, -1), keepdim=True)
                    fixed_norm = torch.linalg.vector_norm(weights, dim=(-2, -1), keepdim=True)
                    scale = fixed_norm * synapses.beta / (trace_norm + 1e-6)
                else:
                    scale = synapses.fixed_scale_factor
                weights = weights + scale * trace
        if getattr(synapses, 'normalize_weights', True):
            weights = weights / (torch.linalg.vector_norm(weights, dim=(-2, -1), keepdim=True) + 1e-6)
        return weights.cpu().numpy()

    elif 'STPN' in model_type:
        fixed_weights = rnn_module.weight
        if len(state) >= 2:
            F_state = state[1]
            fixed_expanded = fixed_weights.unsqueeze(0).expand(batch_size, -1, -1).to(device)
            G = fixed_expanded + F_state
            G_norm = torch.linalg.norm(G, ord=2, dim=1, keepdim=True)
            G_normalized = G / (G_norm + 1e-6)
            return G_normalized.cpu().numpy()
        else:
            fixed_expanded = fixed_weights.unsqueeze(0).expand(batch_size, -1, -1)
            G_norm = torch.linalg.norm(fixed_expanded, ord=2, dim=1, keepdim=True)
            return (fixed_expanded / (G_norm + 1e-6)).cpu().numpy()

    elif 'PlasticLSTM' in model_type:
        fixed_weights = rnn_module.weight_h.cpu().numpy()
        alpha = rnn_module.alpha.cpu().numpy()
        d_model = rnn_module.d_model

        if len(state) >= 3:
            hebb_state = state[2]
            hebb_weights = hebb_state.cpu().numpy()
            cell_gate_fixed = fixed_weights[:, 3*d_model:4*d_model]
            plastic_component = alpha * hebb_weights
            combined = cell_gate_fixed[np.newaxis, :, :] + plastic_component
            return combined
        else:
            cell_gate_fixed = fixed_weights[:, 3*d_model:4*d_model]
            return np.tile(cell_gate_fixed, (batch_size, 1, 1))

    elif 'PlasticRNN' in model_type:
        fixed_weights = rnn_module.weight_h.cpu().numpy()
        alpha = rnn_module.alpha.cpu().numpy()

        if len(state) >= 2:
            hebb_state = state[1]
            hebb_weights = hebb_state.cpu().numpy()
            plastic_component = alpha * hebb_weights
            combined = fixed_weights[np.newaxis, :, :] + plastic_component
            return combined
        else:
            return np.tile(fixed_weights, (batch_size, 1, 1))

    else:
        return state[0].cpu().numpy()[:, :, np.newaxis]


def resolve_checkpoint_path(path):
    """Resolve a path to a checkpoint file.

    If path is a directory, finds the most recent checkpoint file in it.
    If path is already a .pt file, returns it as-is.
    """
    if os.path.isfile(path) and path.endswith('.pt'):
        return path

    if os.path.isdir(path):
        checkpoint_files = [f for f in os.listdir(path) if f.endswith('.pt')]
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {path}")

        def get_epoch(filename):
            try:
                parts = filename.replace('.pt', '').split('_')
                for i, part in enumerate(parts):
                    if part == 'epoch' and i + 1 < len(parts):
                        return int(parts[i + 1])
                return 0
            except (ValueError, IndexError):
                return 0

        checkpoint_files.sort(key=get_epoch, reverse=True)
        resolved = os.path.join(path, checkpoint_files[0])
        print(f"Resolved checkpoint: {resolved}")
        return resolved

    raise FileNotFoundError(f"Path does not exist or is not a valid checkpoint: {path}")


def load_from_checkpoint(checkpoint_path, max_samples=100, phase='test', device='cpu', num_sequences=1):
    """Load checkpoint, extract weights and activations, and return data needed for visualization."""
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, '.hydra', 'config.yaml')
    cfg = OmegaConf.load(config_path)
    state_dict = {key.removeprefix('_orig_mod.'): value
                  for key, value in checkpoint['model_state_dict'].items()}
    # Hydra saves the width before parameter matching; the checkpoint holds the fitted width.
    cfg.model.d_model = int(state_dict['output_proj.weight'].shape[1])

    device = torch.device(device)

    model_name = None
    hydra_config_path = os.path.join(checkpoint_dir, '.hydra', 'hydra.yaml')
    if os.path.exists(hydra_config_path):
        hydra_cfg = OmegaConf.load(hydra_config_path)
        runtime_cfg = OmegaConf.select(hydra_cfg, 'runtime')
        if runtime_cfg is not None and 'choices' in runtime_cfg:
            model_name = runtime_cfg.choices.get('model', None)

    if model_name is None:
        model_name = infer_model_type_from_path(checkpoint_path)

    model = SequentialSplitMNISTModel(
        model_cfg=cfg.model,
        task_cfg=cfg.task,
        model_type=model_name
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    dataset = SequentialSplitMNISTDataset(phase=phase, dataset_mode=cfg.task.get('dataset_mode', 'split_mnist'))

    dataloader = SequentialSplitMNISTDataloader(
        dataset=dataset,
        internal_ticks=cfg.task.get('iterations', 1),
        n_exemplars_per_class=cfg.task.get('n_exemplars_per_class', 1),
        queries_per_loss=cfg.task.get('queries_per_loss', 5),
        batch_size=min(32, max_samples),
        epoch_size=max_samples
    )

    rnn_module = model.rnn.recurrent_module
    model_type = type(rnn_module).__name__

    is_plastic_model = False
    if model_name in ['PlasticLSTM', 'PlasticRNN', 'STPN']:
        is_plastic_model = True
    elif 'ContinuousThoughtMachine' in model_type:
        is_plastic_model = rnn_module.plastic

    if is_plastic_model:
        print("Extracting weights...")
        weights_by_tick = extract_weights(model, dataloader, device, max_samples=max_samples)
    else:
        print(f"Skipping weight extraction for non-plastic model: {model_name}")
        weights_by_tick = None

    print("Extracting activations...")
    activations_by_type = extract_activations(model, dataloader, device, max_samples=max_samples)

    task_params = {
        'n_exemplars_per_class': cfg.task.get('n_exemplars_per_class', 1),
        'queries_per_loss': cfg.task.get('queries_per_loss', 5),
        'internal_ticks': cfg.task.get('iterations', 1),
    }

    return weights_by_tick, activations_by_type, task_params, model_name


def main():
    parser = argparse.ArgumentParser(
        description='Extract weights and create phase-colored UMAP visualizations'
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--checkpoint', type=str,
                             help='Path to single model checkpoint')
    input_group.add_argument('--checkpoints', type=str, nargs='+',
                             help='Paths to multiple model checkpoints (creates combined figure)')
    input_group.add_argument('--weights-file', type=str,
                             help='Path to pre-saved weights .pkl file')

    parser.add_argument('--output-dir', type=str, default='outputs/analysis',
                        help='Output directory')
    parser.add_argument('--max-samples', type=int, default=100,
                        help='Maximum samples to extract')
    parser.add_argument('--phase', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset phase')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for extraction')

    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second for GIF')
    parser.add_argument('--figsize', type=float, default=10,
                        help='Figure size (square) for single model')
    parser.add_argument('--figsize-per-model', type=float, default=5,
                        help='Figure size per model for combined figure')
    parser.add_argument('--no-gif', action='store_true',
                        help='Skip GIF creation')
    parser.add_argument('--no-trajectory', action='store_true',
                        help='Skip trajectory image creation')
    parser.add_argument('--num-sequences', type=int, default=1,
                        help='Number of sequences to run back-to-back')
    parser.add_argument('--reload', action='store_true',
                        help='Reload UMAP embeddings from checkpoint if available')

    args = parser.parse_args()

    # Multiple checkpoints mode
    if args.checkpoints:
        output_dir = os.path.join(args.output_dir, 'sequential_split_mnist', 'combined')
        os.makedirs(output_dir, exist_ok=True)

        model_order = ['PlasticLSTM', 'STPN', 'PlasticCTM']

        model_data_dict = {}
        for ckpt_path in args.checkpoints:
            print(f"\n{'='*60}")
            print(f"Processing: {ckpt_path}")
            print('='*60)

            weights_by_tick, activations_by_type, task_params, model_name = load_from_checkpoint(
                ckpt_path,
                max_samples=args.max_samples,
                phase=args.phase,
                device=args.device,
                num_sequences=args.num_sequences
            )

            if weights_by_tick is not None:
                n_ticks = len(weights_by_tick)
            else:
                first_act_type = next(iter(activations_by_type.keys()))
                n_ticks = len(activations_by_type[first_act_type])

            phases = compute_phase_boundaries(
                n_ticks=n_ticks,
                n_exemplars_per_class=task_params['n_exemplars_per_class'],
                queries_per_loss=task_params['queries_per_loss'],
                internal_ticks=task_params['internal_ticks'],
                num_sequences=args.num_sequences
            )

            path_parts = ckpt_path.split('/')
            timestamp = None
            for i, part in enumerate(path_parts):
                if model_name in part and i + 1 < len(path_parts):
                    timestamp = path_parts[i + 1]
                    break

            if timestamp:
                unique_model_name = f"{model_name}_{timestamp}"
            else:
                unique_model_name = model_name

            model_output_dir = os.path.join(output_dir, unique_model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            weights_embeddings_umap_2d = None
            weights_embeddings_umap_3d = None
            weights_embeddings_pca_2d = None
            weights_embeddings_pca_3d = None

            if weights_by_tick is not None:
                print(f"Fitting UMAP 2D for {model_name} weights...")
                weights_umap_2d_path = os.path.join(model_output_dir, 'weights_umap_2d.pkl')
                weights_embeddings_umap_2d = fit_umap_and_extract(
                    weights_by_tick,
                    checkpoint_path=weights_umap_2d_path,
                    reload=args.reload,
                    n_components=2
                )

                print(f"Fitting UMAP 3D for {model_name} weights...")
                weights_umap_3d_path = os.path.join(model_output_dir, 'weights_umap_3d.pkl')
                weights_embeddings_umap_3d = fit_umap_and_extract(
                    weights_by_tick,
                    checkpoint_path=weights_umap_3d_path,
                    reload=args.reload,
                    n_components=3
                )

                print(f"Fitting PCA 2D for {model_name} weights...")
                weights_pca_2d_path = os.path.join(model_output_dir, 'weights_pca_2d.pkl')
                weights_embeddings_pca_2d = fit_pca_and_extract(
                    weights_by_tick,
                    checkpoint_path=weights_pca_2d_path,
                    reload=args.reload,
                    n_components=2
                )

                print(f"Fitting PCA 3D for {model_name} weights...")
                weights_pca_3d_path = os.path.join(model_output_dir, 'weights_pca_3d.pkl')
                weights_embeddings_pca_3d = fit_pca_and_extract(
                    weights_by_tick,
                    checkpoint_path=weights_pca_3d_path,
                    reload=args.reload,
                    n_components=3
                )

            activations_embeddings_umap_2d = {}
            activations_embeddings_umap_3d = {}
            activations_embeddings_pca_2d = {}
            activations_embeddings_pca_3d = {}

            for act_type, act_by_tick in activations_by_type.items():
                print(f"Fitting UMAP 2D for {model_name} activations ({act_type})...")
                act_umap_2d_path = os.path.join(model_output_dir, f'activations_{act_type}_umap_2d.pkl')
                activations_embeddings_umap_2d[act_type] = fit_umap_and_extract(
                    act_by_tick,
                    checkpoint_path=act_umap_2d_path,
                    reload=args.reload,
                    n_components=2
                )

                print(f"Fitting UMAP 3D for {model_name} activations ({act_type})...")
                act_umap_3d_path = os.path.join(model_output_dir, f'activations_{act_type}_umap_3d.pkl')
                activations_embeddings_umap_3d[act_type] = fit_umap_and_extract(
                    act_by_tick,
                    checkpoint_path=act_umap_3d_path,
                    reload=args.reload,
                    n_components=3
                )

                print(f"Fitting PCA 2D for {model_name} activations ({act_type})...")
                act_pca_2d_path = os.path.join(model_output_dir, f'activations_{act_type}_pca_2d.pkl')
                activations_embeddings_pca_2d[act_type] = fit_pca_and_extract(
                    act_by_tick,
                    checkpoint_path=act_pca_2d_path,
                    reload=args.reload,
                    n_components=2
                )

                print(f"Fitting PCA 3D for {model_name} activations ({act_type})...")
                act_pca_3d_path = os.path.join(model_output_dir, f'activations_{act_type}_pca_3d.pkl')
                activations_embeddings_pca_3d[act_type] = fit_pca_and_extract(
                    act_by_tick,
                    checkpoint_path=act_pca_3d_path,
                    reload=args.reload,
                    n_components=3
                )

            model_data_dict[unique_model_name] = {
                'name': model_name,
                'unique_name': unique_model_name,
                'embeddings_umap_2d': weights_embeddings_umap_2d,
                'embeddings_umap_3d': weights_embeddings_umap_3d,
                'embeddings_pca_2d': weights_embeddings_pca_2d,
                'embeddings_pca_3d': weights_embeddings_pca_3d,
                'activations_embeddings_umap_2d': activations_embeddings_umap_2d,
                'activations_embeddings_umap_3d': activations_embeddings_umap_3d,
                'activations_embeddings_pca_2d': activations_embeddings_pca_2d,
                'activations_embeddings_pca_3d': activations_embeddings_pca_3d,
                'phases': phases,
                'weights_by_tick': weights_by_tick,
                'activations_by_type': activations_by_type,
                'task_params': task_params
            }

            weights_path = os.path.join(model_output_dir, 'weights.pkl')
            with open(weights_path, 'wb') as f:
                pickle.dump({
                    'weights_by_tick': weights_by_tick,
                    'activations_by_type': activations_by_type,
                    'task_params': task_params,
                    'model_name': model_name
                }, f)

            if not args.no_gif:
                if weights_embeddings_umap_2d is not None:
                    gif_path = os.path.join(model_output_dir, 'weight_evolution_umap_2d.gif')
                    create_gif_by_phase(weights_embeddings_umap_2d, phases, gif_path, fps=args.fps)

                if weights_embeddings_umap_3d is not None:
                    gif_path = os.path.join(model_output_dir, 'weight_evolution_umap_3d.gif')
                    create_gif_by_phase_3d(weights_embeddings_umap_3d, phases, gif_path, fps=args.fps)

                if weights_embeddings_pca_2d is not None:
                    gif_path = os.path.join(model_output_dir, 'weight_evolution_pca_2d.gif')
                    create_gif_by_phase(weights_embeddings_pca_2d, phases, gif_path, fps=args.fps)

                if weights_embeddings_pca_3d is not None:
                    gif_path = os.path.join(model_output_dir, 'weight_evolution_pca_3d.gif')
                    create_gif_by_phase_3d(weights_embeddings_pca_3d, phases, gif_path, fps=args.fps)

            if not args.no_trajectory:
                if weights_embeddings_umap_2d is not None:
                    traj_path = os.path.join(model_output_dir, 'weight_trajectory_umap_2d.png')
                    create_trajectory_image_by_phase(weights_embeddings_umap_2d, phases, traj_path,
                                                     figsize=(args.figsize_per_model, args.figsize_per_model))

                if weights_embeddings_umap_3d is not None:
                    traj_path = os.path.join(model_output_dir, 'weight_trajectory_umap_3d.png')
                    create_trajectory_image_by_phase_3d(weights_embeddings_umap_3d, phases, traj_path,
                                                        figsize=(12, 10))

                if weights_embeddings_pca_2d is not None:
                    traj_path = os.path.join(model_output_dir, 'weight_trajectory_pca_2d.png')
                    create_trajectory_image_by_phase(weights_embeddings_pca_2d, phases, traj_path,
                                                     figsize=(args.figsize_per_model, args.figsize_per_model))

                if weights_embeddings_pca_3d is not None:
                    traj_path = os.path.join(model_output_dir, 'weight_trajectory_pca_3d.png')
                    create_trajectory_image_by_phase_3d(weights_embeddings_pca_3d, phases, traj_path,
                                                        figsize=(12, 10))

            if not args.no_gif:
                for act_type in activations_by_type.keys():
                    act_gif_path = os.path.join(model_output_dir, f'activation_{act_type}_evolution_umap_2d.gif')
                    create_gif_by_phase(activations_embeddings_umap_2d[act_type], phases, act_gif_path, fps=args.fps)

                    act_gif_path = os.path.join(model_output_dir, f'activation_{act_type}_evolution_umap_3d.gif')
                    create_gif_by_phase_3d(activations_embeddings_umap_3d[act_type], phases, act_gif_path, fps=args.fps)

                    act_gif_path = os.path.join(model_output_dir, f'activation_{act_type}_evolution_pca_2d.gif')
                    create_gif_by_phase(activations_embeddings_pca_2d[act_type], phases, act_gif_path, fps=args.fps)

                    act_gif_path = os.path.join(model_output_dir, f'activation_{act_type}_evolution_pca_3d.gif')
                    create_gif_by_phase_3d(activations_embeddings_pca_3d[act_type], phases, act_gif_path, fps=args.fps)

            if not args.no_trajectory:
                for act_type in activations_by_type.keys():
                    act_traj_path = os.path.join(model_output_dir, f'activation_{act_type}_trajectory_umap_2d.png')
                    create_trajectory_image_by_phase(activations_embeddings_umap_2d[act_type], phases, act_traj_path,
                                                     figsize=(args.figsize_per_model, args.figsize_per_model))

                    act_traj_path = os.path.join(model_output_dir, f'activation_{act_type}_trajectory_umap_3d.png')
                    create_trajectory_image_by_phase_3d(activations_embeddings_umap_3d[act_type], phases, act_traj_path,
                                                        figsize=(12, 10))

                    act_traj_path = os.path.join(model_output_dir, f'activation_{act_type}_trajectory_pca_2d.png')
                    create_trajectory_image_by_phase(activations_embeddings_pca_2d[act_type], phases, act_traj_path,
                                                     figsize=(args.figsize_per_model, args.figsize_per_model))

                    act_traj_path = os.path.join(model_output_dir, f'activation_{act_type}_trajectory_pca_3d.png')
                    create_trajectory_image_by_phase_3d(activations_embeddings_pca_3d[act_type], phases, act_traj_path,
                                                        figsize=(12, 10))

        # Second pass: Perform Procrustes alignment for non-plastic models
        print(f"\n{'='*60}")
        print("PROCRUSTES ALIGNMENT")
        print('='*60)

        plastic_models = {}
        non_plastic_models = {}

        for unique_name, data in model_data_dict.items():
            model_name = data['name']
            if model_name == 'PlasticLSTM':
                plastic_models['LSTM'] = (unique_name, data)
            elif model_name == 'LSTM':
                non_plastic_models['LSTM'] = (unique_name, data)
            elif model_name in ['CTM', 'PlasticCTM']:
                if data['weights_by_tick'] is not None:
                    plastic_models['CTM'] = (unique_name, data)
                else:
                    non_plastic_models['CTM'] = (unique_name, data)

        alignment_results = []

        # Align non-plastic models to their plastic counterparts
        for model_type in ['LSTM', 'CTM']:
            if model_type in plastic_models and model_type in non_plastic_models:
                plastic_unique_name, plastic_data = plastic_models[model_type]
                non_plastic_unique_name, non_plastic_data = non_plastic_models[model_type]

                print(f"\nAligning {non_plastic_unique_name} to {plastic_unique_name}")

                activation_types = list(non_plastic_data['activations_embeddings_umap_2d'].keys())

                for act_type in activation_types:
                    print(f"  Aligning activation type: {act_type}")

                    ref_umap_2d = plastic_data['activations_embeddings_umap_2d'][act_type]
                    ref_umap_3d = plastic_data['activations_embeddings_umap_3d'][act_type]
                    ref_pca_2d = plastic_data['activations_embeddings_pca_2d'][act_type]
                    ref_pca_3d = plastic_data['activations_embeddings_pca_3d'][act_type]

                    tgt_umap_2d = non_plastic_data['activations_embeddings_umap_2d'][act_type]
                    tgt_umap_3d = non_plastic_data['activations_embeddings_umap_3d'][act_type]
                    tgt_pca_2d = non_plastic_data['activations_embeddings_pca_2d'][act_type]
                    tgt_pca_3d = non_plastic_data['activations_embeddings_pca_3d'][act_type]

                    aligned_umap_2d, disp_umap_2d = procrustes_align_embeddings(ref_umap_2d, tgt_umap_2d)
                    aligned_umap_3d, disp_umap_3d = procrustes_align_embeddings(ref_umap_3d, tgt_umap_3d)
                    aligned_pca_2d, disp_pca_2d = procrustes_align_embeddings(ref_pca_2d, tgt_pca_2d)
                    aligned_pca_3d, disp_pca_3d = procrustes_align_embeddings(ref_pca_3d, tgt_pca_3d)

                    print(f"    UMAP 2D disparity: {disp_umap_2d:.4f}")
                    print(f"    UMAP 3D disparity: {disp_umap_3d:.4f}")
                    print(f"    PCA 2D disparity: {disp_pca_2d:.4f}")
                    print(f"    PCA 3D disparity: {disp_pca_3d:.4f}")

                    alignment_results.append({
                        'reference_model': plastic_unique_name,
                        'target_model': non_plastic_unique_name,
                        'model_type': model_type,
                        'activation_type': act_type,
                        'umap_2d_disparity': disp_umap_2d,
                        'umap_3d_disparity': disp_umap_3d,
                        'pca_2d_disparity': disp_pca_2d,
                        'pca_3d_disparity': disp_pca_3d
                    })

                    non_plastic_data['activations_embeddings_umap_2d'][act_type] = aligned_umap_2d
                    non_plastic_data['activations_embeddings_umap_3d'][act_type] = aligned_umap_3d
                    non_plastic_data['activations_embeddings_pca_2d'][act_type] = aligned_pca_2d
                    non_plastic_data['activations_embeddings_pca_3d'][act_type] = aligned_pca_3d

                    model_output_dir = os.path.join(output_dir, non_plastic_unique_name)

                    aligned_umap_2d_path = os.path.join(model_output_dir, f'activations_{act_type}_umap_2d.pkl')
                    with open(aligned_umap_2d_path, 'wb') as f:
                        pickle.dump(aligned_umap_2d, f)

                    aligned_umap_3d_path = os.path.join(model_output_dir, f'activations_{act_type}_umap_3d.pkl')
                    with open(aligned_umap_3d_path, 'wb') as f:
                        pickle.dump(aligned_umap_3d, f)

                    aligned_pca_2d_path = os.path.join(model_output_dir, f'activations_{act_type}_pca_2d.pkl')
                    with open(aligned_pca_2d_path, 'wb') as f:
                        pickle.dump(aligned_pca_2d, f)

                    aligned_pca_3d_path = os.path.join(model_output_dir, f'activations_{act_type}_pca_3d.pkl')
                    with open(aligned_pca_3d_path, 'wb') as f:
                        pickle.dump(aligned_pca_3d, f)

                    print(f"    Aligned embeddings saved to: {model_output_dir}")

                    phases = non_plastic_data['phases']

                    if not args.no_gif:
                        act_gif_path = os.path.join(model_output_dir, f'activation_{act_type}_evolution_umap_2d.gif')
                        create_gif_by_phase(aligned_umap_2d, phases, act_gif_path, fps=args.fps)

                        act_gif_path = os.path.join(model_output_dir, f'activation_{act_type}_evolution_umap_3d.gif')
                        create_gif_by_phase_3d(aligned_umap_3d, phases, act_gif_path, fps=args.fps)

                        act_gif_path = os.path.join(model_output_dir, f'activation_{act_type}_evolution_pca_2d.gif')
                        create_gif_by_phase(aligned_pca_2d, phases, act_gif_path, fps=args.fps)

                        act_gif_path = os.path.join(model_output_dir, f'activation_{act_type}_evolution_pca_3d.gif')
                        create_gif_by_phase_3d(aligned_pca_3d, phases, act_gif_path, fps=args.fps)

                    if not args.no_trajectory:
                        act_traj_path = os.path.join(model_output_dir, f'activation_{act_type}_trajectory_umap_2d.png')
                        create_trajectory_image_by_phase(aligned_umap_2d, phases, act_traj_path,
                                                         figsize=(args.figsize_per_model, args.figsize_per_model))

                        act_traj_path = os.path.join(model_output_dir, f'activation_{act_type}_trajectory_umap_3d.png')
                        create_trajectory_image_by_phase_3d(aligned_umap_3d, phases, act_traj_path,
                                                            figsize=(12, 10))

                        act_traj_path = os.path.join(model_output_dir, f'activation_{act_type}_trajectory_pca_2d.png')
                        create_trajectory_image_by_phase(aligned_pca_2d, phases, act_traj_path,
                                                         figsize=(args.figsize_per_model, args.figsize_per_model))

                        act_traj_path = os.path.join(model_output_dir, f'activation_{act_type}_trajectory_pca_3d.png')
                        create_trajectory_image_by_phase_3d(aligned_pca_3d, phases, act_traj_path,
                                                            figsize=(12, 10))

                print(f"  Completed alignment for {non_plastic_unique_name}")

        if alignment_results:
            csv_path = os.path.join(output_dir, 'alignment_disparities.csv')
            csv_columns = ['reference_model', 'target_model', 'model_type', 'activation_type',
                          'umap_2d_disparity', 'umap_3d_disparity', 'pca_2d_disparity', 'pca_3d_disparity']

            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                writer.writerows(alignment_results)

            print(f"\nAlignment disparity values saved to: {csv_path}")

        print(f"\n{'='*60}")
        print("PROCRUSTES ALIGNMENT COMPLETE")
        print('='*60)

        print(f"\nAll outputs saved to: {output_dir}")

    # Single checkpoint mode
    elif args.checkpoint:
        checkpoint_name = os.path.basename(os.path.dirname(args.checkpoint))
        output_dir = os.path.join(args.output_dir, 'sequential_split_mnist', checkpoint_name)
        os.makedirs(output_dir, exist_ok=True)

        weights_by_tick, activations_by_type, task_params, model_name = load_from_checkpoint(
            args.checkpoint,
            max_samples=args.max_samples,
            phase=args.phase,
            device=args.device,
            num_sequences=args.num_sequences
        )

        weights_path = os.path.join(output_dir, 'weights.pkl')
        with open(weights_path, 'wb') as f:
            pickle.dump({
                'weights_by_tick': weights_by_tick,
                'activations_by_type': activations_by_type,
                'task_params': task_params
            }, f)
        print(f"Weights and activations saved: {weights_path}")

        if weights_by_tick is not None:
            n_ticks = len(weights_by_tick)
            n_samples = weights_by_tick[0].shape[0]
        else:
            first_act_type = next(iter(activations_by_type.keys()))
            n_ticks = len(activations_by_type[first_act_type])
            n_samples = activations_by_type[first_act_type][0].shape[0]

        print(f"\nExtracted {n_samples} samples across {n_ticks} ticks")

        phases = compute_phase_boundaries(
            n_ticks=n_ticks,
            n_exemplars_per_class=task_params['n_exemplars_per_class'],
            queries_per_loss=task_params['queries_per_loss'],
            internal_ticks=task_params['internal_ticks'],
            num_sequences=args.num_sequences
        )

        weights_embeddings_umap_2d = None
        weights_embeddings_umap_3d = None
        weights_embeddings_pca_2d = None
        weights_embeddings_pca_3d = None

        if weights_by_tick is not None:
            print("\nFitting UMAP 2D for weights...")
            weights_umap_2d_path = os.path.join(output_dir, 'weights_umap_2d.pkl')
            weights_embeddings_umap_2d = fit_umap_and_extract(
                weights_by_tick,
                checkpoint_path=weights_umap_2d_path,
                reload=args.reload,
                n_components=2
            )

            print("\nFitting UMAP 3D for weights...")
            weights_umap_3d_path = os.path.join(output_dir, 'weights_umap_3d.pkl')
            weights_embeddings_umap_3d = fit_umap_and_extract(
                weights_by_tick,
                checkpoint_path=weights_umap_3d_path,
                reload=args.reload,
                n_components=3
            )

            print("\nFitting PCA 2D for weights...")
            weights_pca_2d_path = os.path.join(output_dir, 'weights_pca_2d.pkl')
            weights_embeddings_pca_2d = fit_pca_and_extract(
                weights_by_tick,
                checkpoint_path=weights_pca_2d_path,
                reload=args.reload,
                n_components=2
            )

            print("\nFitting PCA 3D for weights...")
            weights_pca_3d_path = os.path.join(output_dir, 'weights_pca_3d.pkl')
            weights_embeddings_pca_3d = fit_pca_and_extract(
                weights_by_tick,
                checkpoint_path=weights_pca_3d_path,
                reload=args.reload,
                n_components=3
            )

        activations_embeddings_umap_2d = {}
        activations_embeddings_umap_3d = {}
        activations_embeddings_pca_2d = {}
        activations_embeddings_pca_3d = {}

        for act_type, act_by_tick in activations_by_type.items():
            print(f"\nFitting UMAP 2D for activations ({act_type})...")
            act_umap_2d_path = os.path.join(output_dir, f'activations_{act_type}_umap_2d.pkl')
            activations_embeddings_umap_2d[act_type] = fit_umap_and_extract(
                act_by_tick,
                checkpoint_path=act_umap_2d_path,
                reload=args.reload,
                n_components=2
            )

            print(f"\nFitting UMAP 3D for activations ({act_type})...")
            act_umap_3d_path = os.path.join(output_dir, f'activations_{act_type}_umap_3d.pkl')
            activations_embeddings_umap_3d[act_type] = fit_umap_and_extract(
                act_by_tick,
                checkpoint_path=act_umap_3d_path,
                reload=args.reload,
                n_components=3
            )

            print(f"\nFitting PCA 2D for activations ({act_type})...")
            act_pca_2d_path = os.path.join(output_dir, f'activations_{act_type}_pca_2d.pkl')
            activations_embeddings_pca_2d[act_type] = fit_pca_and_extract(
                act_by_tick,
                checkpoint_path=act_pca_2d_path,
                reload=args.reload,
                n_components=2
            )

            print(f"\nFitting PCA 3D for activations ({act_type})...")
            act_pca_3d_path = os.path.join(output_dir, f'activations_{act_type}_pca_3d.pkl')
            activations_embeddings_pca_3d[act_type] = fit_pca_and_extract(
                act_by_tick,
                checkpoint_path=act_pca_3d_path,
                reload=args.reload,
                n_components=3
            )

        figsize = (args.figsize, args.figsize)

        if not args.no_gif:
            if weights_embeddings_umap_2d is not None:
                gif_path = os.path.join(output_dir, 'weight_evolution_umap_2d.gif')
                create_gif_by_phase(weights_embeddings_umap_2d, phases, gif_path, fps=args.fps)

            if weights_embeddings_umap_3d is not None:
                gif_path = os.path.join(output_dir, 'weight_evolution_umap_3d.gif')
                create_gif_by_phase_3d(weights_embeddings_umap_3d, phases, gif_path, fps=args.fps)

            if weights_embeddings_pca_2d is not None:
                gif_path = os.path.join(output_dir, 'weight_evolution_pca_2d.gif')
                create_gif_by_phase(weights_embeddings_pca_2d, phases, gif_path, fps=args.fps)

            if weights_embeddings_pca_3d is not None:
                gif_path = os.path.join(output_dir, 'weight_evolution_pca_3d.gif')
                create_gif_by_phase_3d(weights_embeddings_pca_3d, phases, gif_path, fps=args.fps)

            for act_type in activations_by_type.keys():
                act_gif_path = os.path.join(output_dir, f'activation_{act_type}_evolution_umap_2d.gif')
                create_gif_by_phase(activations_embeddings_umap_2d[act_type], phases, act_gif_path, fps=args.fps)

                act_gif_path = os.path.join(output_dir, f'activation_{act_type}_evolution_umap_3d.gif')
                create_gif_by_phase_3d(activations_embeddings_umap_3d[act_type], phases, act_gif_path, fps=args.fps)

                act_gif_path = os.path.join(output_dir, f'activation_{act_type}_evolution_pca_2d.gif')
                create_gif_by_phase(activations_embeddings_pca_2d[act_type], phases, act_gif_path, fps=args.fps)

                act_gif_path = os.path.join(output_dir, f'activation_{act_type}_evolution_pca_3d.gif')
                create_gif_by_phase_3d(activations_embeddings_pca_3d[act_type], phases, act_gif_path, fps=args.fps)

        if not args.no_trajectory:
            if weights_embeddings_umap_2d is not None:
                traj_path = os.path.join(output_dir, 'trajectory_umap_2d.png')
                create_trajectory_image_by_phase(weights_embeddings_umap_2d, phases, traj_path, figsize=figsize)

            if weights_embeddings_umap_3d is not None:
                traj_path = os.path.join(output_dir, 'trajectory_umap_3d.png')
                create_trajectory_image_by_phase_3d(weights_embeddings_umap_3d, phases, traj_path, figsize=(12, 10))

            if weights_embeddings_pca_2d is not None:
                traj_path = os.path.join(output_dir, 'trajectory_pca_2d.png')
                create_trajectory_image_by_phase(weights_embeddings_pca_2d, phases, traj_path, figsize=figsize)

            if weights_embeddings_pca_3d is not None:
                traj_path = os.path.join(output_dir, 'trajectory_pca_3d.png')
                create_trajectory_image_by_phase_3d(weights_embeddings_pca_3d, phases, traj_path, figsize=(12, 10))

            for act_type in activations_by_type.keys():
                act_traj_path = os.path.join(output_dir, f'activation_{act_type}_trajectory_umap_2d.png')
                create_trajectory_image_by_phase(activations_embeddings_umap_2d[act_type], phases, act_traj_path, figsize=figsize)

                act_traj_path = os.path.join(output_dir, f'activation_{act_type}_trajectory_umap_3d.png')
                create_trajectory_image_by_phase_3d(activations_embeddings_umap_3d[act_type], phases, act_traj_path, figsize=(12, 10))

                act_traj_path = os.path.join(output_dir, f'activation_{act_type}_trajectory_pca_2d.png')
                create_trajectory_image_by_phase(activations_embeddings_pca_2d[act_type], phases, act_traj_path, figsize=figsize)

                act_traj_path = os.path.join(output_dir, f'activation_{act_type}_trajectory_pca_3d.png')
                create_trajectory_image_by_phase_3d(activations_embeddings_pca_3d[act_type], phases, act_traj_path, figsize=(12, 10))

        print(f"\nAll outputs saved to: {output_dir}")

    # Weights file mode
    else:
        with open(args.weights_file, 'rb') as f:
            data = pickle.load(f)
        weights_by_tick = data['weights_by_tick']
        task_params = data.get('task_params', {
            'n_exemplars_per_class': 1,
            'queries_per_loss': 5,
            'internal_ticks': 1,
        })
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Loaded weights from: {args.weights_file}")

        n_ticks = len(weights_by_tick)
        n_samples = weights_by_tick[0].shape[0]
        print(f"\nLoaded {n_samples} samples across {n_ticks} ticks")

        phases = compute_phase_boundaries(
            n_ticks=n_ticks,
            n_exemplars_per_class=task_params['n_exemplars_per_class'],
            queries_per_loss=task_params['queries_per_loss'],
            internal_ticks=task_params['internal_ticks'],
            num_sequences=args.num_sequences
        )

        print("\nFitting UMAP...")
        embeddings_by_tick = fit_umap_and_extract(weights_by_tick)

        figsize = (args.figsize, args.figsize)

        if not args.no_gif:
            gif_path = os.path.join(output_dir, 'weight_evolution_by_phase.gif')
            create_gif_by_phase(embeddings_by_tick, phases, gif_path, fps=args.fps)

        if not args.no_trajectory:
            traj_path = os.path.join(output_dir, 'trajectory_by_phase.png')
            create_trajectory_image_by_phase(embeddings_by_tick, phases, traj_path, figsize=figsize)

        print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

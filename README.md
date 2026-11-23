# Differentiable Plastic Recurrent Neural Networks

Code for **Differentiable Plastic Recurrent Neural Networks: A Systematic Evaluation**, including the Plastic Continuous Thought Machine (Plastic CTM), recurrent baselines, and the experiments and analyses in the TMLR paper.

## Setup

Use Python 3.10 or later and install the locked dependencies:

```bash
uv sync --locked
```

Commands below run from the repository root. Training defaults to GPU 0; pass `device=-1` for CPU or `device=1` for another GPU. Metrics are saved locally; Weights & Biases is disabled by default.

### Data

Copy, regression, and maze tasks generate their own data. MNIST and CIFAR-10 download automatically. For the few-shot image tasks, download and extract the datasets:

```bash
mkdir -p data/raw_data data/cifar_fs data/miniImageNet

# CIFAR-FS
uv run gdown 1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS -O data/raw_data/CIFAR-FS.zip
unzip data/raw_data/CIFAR-FS.zip -d data/cifar_fs/

# MiniImageNet
uv run gdown 12V7qi-AjrYi6OoJdYcN_k502BM_jcP8D -O data/raw_data/miniImageNet.zip
unzip data/raw_data/miniImageNet.zip -d data/miniImageNet/
```

The loaders expect `CIFAR_FS_{train,val,test}.pickle` directly in `data/cifar_fs/`, and `miniImageNet_category_split_*.pickle` directly in `data/miniImageNet/`. Move the files up a level if an archive creates an extra enclosing directory.

## Experiments

```bash
uv run python main.py task=Copy model=CTM model.plastic=true
```

| Paper experiment | Task configuration |
| --- | --- |
| Memory and recall | `Copy` |
| One-shot CIFAR-100 classification | `FewShotCIFAR` |
| One-shot MiniImageNet classification | `FewShotMiniImageNet` |
| In-context function learning | `FewShotRegression` |
| Maze exploration | `MazeExploration` |
| Sequential Split-MNIST | `SequentialSplitMNIST` |
| CIFAR-10 classification (appendix) | `ImageClassificationCIFAR10` |

Recurrent models are `RNN`, `PlasticRNN`, `LSTM`, `PlasticLSTM`, `HyperLSTM`, `STPN`, `FW`, `GDN`, and `CTM`. The paper's variants use these overrides:

| Variant | Model and overrides |
| --- | --- |
| Plastic CTM | `model=CTM model.plastic=true` |
| Non-plastic CTM | `model=CTM model.plastic=false` |
| Plastic CTM without NLMs | `model=CTMNoNLMs model.plastic=true` |
| Plastic CTM with Hebbian plasticity | `model=CTMHebbian` |
| Plastic RNN with Oja's rule | `model=PlasticRNN model.rule=oja` |
| Plastic LSTM with neuromodulation | `model=PlasticLSTM model.neuromodulation=simple` or `retroactive` |
| CTM with image cross-attention (CIFAR-10) | `model=CrossAttentionCTM model.plastic=true` or `false` |
| Fixed-scale Plastic CTM (maze appendix) | `model=PlasticCTMNoDynamicRescaling` |

`MAML` and `ProtoNet` are also available for the two few-shot image tasks reported in the paper.

Configurations in `conf/` are starting points. Use the paper's appendix for the final task conditions, widths, learning rates, training lengths, and seed counts. `+max_parameters=<budget>` fits the model width to a parameter budget before training.

```bash
# Five seeded few-shot runs, with checkpoints for later analysis
uv run python main.py -m task=FewShotCIFAR model=CTM model.plastic=true \
  seed=0,1,2,3,4 evaluation.evaluation_interval=1 checkpoint.enabled=true

# MLP regression; use n_var=12 and task_mode=linear for linear functions
uv run python main.py task=FewShotRegression model=CTM \
  task.task_mode=mlp task.n_var=6 model.nlm_init_identity=true

# Appendix CIFAR-10 experiment
uv run python main.py task=ImageClassificationCIFAR10 model=CrossAttentionCTM
```

Hydra saves each run's configuration, `metrics.json`, and `summary.json` under `outputs/<task>/<model>/`. Multiruns execute sequentially with Hydra's standard local launcher. Enable checkpointing with `checkpoint.enabled=true`; resume a run using `checkpoint.resume_from=<run-directory-or-checkpoint>` and the same task/model configuration.

### Sequential Split-MNIST

`task=SequentialSplitMNIST` uses the paper's five-way, 25-image episodes: one labeled exemplar per class for A, five A queries, one exemplar per class for B, five A retention queries, and five B queries. Training defaults are 1,000 epochs, batch size 64, and gradient clipping at 10.

Use the appendix's architecture-specific widths, learning rates, and dropout. For example, the Plastic CTM configuration is:

```bash
uv run python main.py -m task=SequentialSplitMNIST model=CTM \
  model.plastic=true model.d_input=256 model.d_model=358 \
  training.learning_rate=0.00048120 model.dropout=0.0017 \
  seed=0,1,2,3,4 checkpoint.enabled=true
```

The task was previously named `ContinualLearningACL`. Analysis scripts accept checkpoints in directories with either name; existing run configurations and results remain valid. See [Sequential Split-MNIST analysis](analysis/sequential_split_mnist/README.md).

### Hyperparameter search

The optional search dependency uses Hydra's standard Optuna sweeper and the search spaces in `conf/search_space/`:

```bash
uv run --extra search python main.py -m --config-name=search task=Copy model=CTM
```

Searches run 20 TPE trials. Sequential Split-MNIST trials use 500 epochs; other tasks retain the general 20-epoch search default. Training lengths and other settings can be supplied as ordinary overrides.

### Optional W&B logging

Authenticate with your own W&B account, then enable logging:

```bash
uv run --extra wandb wandb login
uv run --extra wandb python main.py task=Copy model=CTM logging.wandb.enabled=true
```

## Paper analyses

See [analysis/README.md](analysis/README.md) for result tables, plasticity interventions, transfer metrics, PCA plots, and efficiency profiling. These scripts consume your experiment outputs and checkpoints; datasets and trained checkpoints are not bundled with the code.

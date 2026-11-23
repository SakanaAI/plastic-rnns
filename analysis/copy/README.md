# Copy Task Analysis

## Setup

Create `experiment_dirs.yaml` with paths to your experiment outputs:

```yaml
delay_sweep:
  PlasticLSTM: outputs/Copy/PlasticLSTM/multirun/...
  # ... other models

seqlen_sweep:
  PlasticLSTM: outputs/Copy/PlasticLSTM/multirun/...
  # ... other models
```

## Usage

Generate plots:

```bash
uv run python analysis/copy/analyze.py
```

Outputs PNG and PDF plots to `outputs/analysis/copy/`.

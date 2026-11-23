import argparse
import csv
import os
import sys
import time
from statistics import mean, stdev

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.cells.ctm import (
    ContinuousThoughtMachineCell,
    ContinuousThoughtMachineCellNoNLMs,
    ContinuousThoughtMachineHebbianCell,
)
from models.cells.fw import FW
from models.cells.gated_delta_net import GatedDeltaNetCell
from models.cells.lstm import HyperLSTMCell, LSTMCell, PlasticLSTMCell
from models.cells.rnn import PlasticRNNCell, RNNCell
from models.cells.stpn import STPNCell

D_INPUT = 256

MODEL_SPECS = [
    ("RNN",                  RNNCell,                            lambda d: dict(d_input=D_INPUT, d_model=d, dropout=0.0, layer_norm=True)),
    ("LSTM",                 LSTMCell,                           lambda d: dict(d_input=D_INPUT, d_model=d, dropout=0.0, layer_norm=True, T_max=None)),
    ("PlasticLSTM",          PlasticLSTMCell,                    lambda d: dict(d_input=D_INPUT, d_model=d, dropout=0.0, layer_norm=True, T_max=None, neuromodulation=None)),
    ("PlasticLSTM (simple)", PlasticLSTMCell,                    lambda d: dict(d_input=D_INPUT, d_model=d, dropout=0.0, layer_norm=True, T_max=None, neuromodulation="simple")),
    ("PlasticLSTM (retroactive)", PlasticLSTMCell,               lambda d: dict(d_input=D_INPUT, d_model=d, dropout=0.0, layer_norm=True, T_max=None, neuromodulation="retroactive")),
    ("HyperLSTM",            HyperLSTMCell,                     lambda d: dict(d_input=D_INPUT, d_model=d, hyper_size=64, embedding_size=32, dropout=0.0, layer_norm=True)),
    ("PlasticRNN",           PlasticRNNCell,                     lambda d: dict(d_input=D_INPUT, d_model=d, dropout=0.0, layer_norm=True)),
    ("PlasticRNN (Oja)",      PlasticRNNCell,                     lambda d: dict(d_input=D_INPUT, d_model=d, dropout=0.0, layer_norm=True, rule="oja")),
    ("STPN",                 STPNCell,                           lambda d: dict(d_input=D_INPUT, d_model=d, dropout=0.0, layer_norm=True)),
    ("FW",                   FW,                                 lambda d: dict(d_input=D_INPUT, d_model=d, lmbda=0.9, eta=0.5, steps=1, layer_norm=True, dropout=0.0)),
    ("GDN",                  GatedDeltaNetCell,                  lambda d: dict(d_input=D_INPUT, d_model=d, num_heads=4, expand_k=0.75, expand_v=1.5, conv_size=4, qk_norm="l2", dropout=0.0)),
    ("CTM",                  ContinuousThoughtMachineCell,       lambda d: dict(d_input=D_INPUT, d_model=d, memory_length=10, dropout=0.0, plastic=False)),
    ("Plastic CTM",          ContinuousThoughtMachineCell,       lambda d: dict(d_input=D_INPUT, d_model=d, memory_length=10, dropout=0.0, plastic=True)),
    ("CTM (no NLMs)",        ContinuousThoughtMachineCellNoNLMs, lambda d: dict(d_input=D_INPUT, d_model=d, memory_length=10, dropout=0.0, plastic=True)),
    ("CTM Hebbian",          ContinuousThoughtMachineHebbianCell, lambda d: dict(d_input=d, d_model=d, memory_length=10, dropout=0.0, plastic=True)),
]

PARAM_MATCHED_CONFIGS = {
    "RNN": 464, "LSTM": 188, "PlasticLSTM": 175, "PlasticLSTM (simple)": 175,
    "PlasticLSTM (retroactive)": 175,
    "HyperLSTM": 110, "PlasticRNN": 350, "PlasticRNN (Oja)": 350, "STPN": 230, "FW": 464,
    "GDN": 223,
    "CTM": 340, "Plastic CTM": 256, "CTM (no NLMs)": 260, "CTM Hebbian": 290,
}

CSV_FIELDS = [
    "model", "d_model", "batch_size", "run", "params",
    "fwd_ms", "fwd_ms_iter_std", "fwd_bwd_ms", "fwd_bwd_ms_iter_std",
    "state_mem_kb", "peak_mem_mb",
]
SUMMARY_CSV_FIELDS = [
    "model", "d_model", "batch_size", "n_runs", "params",
    "fwd_ms_mean", "fwd_ms_std", "fwd_bwd_ms_mean", "fwd_bwd_ms_std",
    "state_mem_kb", "peak_mem_mb_mean", "peak_mem_mb_std",
]


def _expand_state(state, batch_size, device):
    return tuple(
        s.to(device).expand(batch_size, *s.shape[1:]).contiguous() if s.shape[0] == 1 and batch_size > 1 else s.to(device)
        for s in state
    )


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _time_steps(fn, device, n_warmup, n_measure):
    for _ in range(n_warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(n_measure):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return times


def profile_cell(cell: nn.Module, d_input: int, batch_size: int, device: str,
                 n_warmup: int = 10, n_measure: int = 100):
    cell = cell.to(device)
    cell.train()
    x = torch.randn(batch_size, d_input, device=device)

    def fresh_state():
        return _expand_state(cell.initialize_state(), batch_size, device)

    with torch.no_grad():
        cell(x, fresh_state())

    def fwd_step():
        with torch.no_grad():
            cell(x, fresh_state())

    def fwd_bwd_step():
        cell.zero_grad()
        out, _ = cell(x, fresh_state())
        out.sum().backward()

    fwd_times = _time_steps(fwd_step, device, n_warmup, n_measure)
    fwd_bwd_times = _time_steps(fwd_bwd_step, device, n_warmup, n_measure)

    state = fresh_state()
    state_floats = sum(s.numel() // max(s.shape[0], 1) for s in state)

    peak_mem_mb = 0.0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        fwd_bwd_step()
        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "params": sum(p.numel() for p in cell.parameters()),
        "fwd_ms": mean(fwd_times) * 1000,
        "fwd_ms_iter_std": stdev(fwd_times) * 1000 if len(fwd_times) > 1 else 0.0,
        "fwd_bwd_ms": mean(fwd_bwd_times) * 1000,
        "fwd_bwd_ms_iter_std": stdev(fwd_bwd_times) * 1000 if len(fwd_bwd_times) > 1 else 0.0,
        "state_mem_kb": state_floats * 4 / 1024,
        "peak_mem_mb": peak_mem_mb,
    }


def _std(values):
    return stdev(values) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Profile recurrent model efficiency")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--param-matched", action="store_true",
                        help="Use parameter-matched d_model values")
    parser.add_argument("--d-models", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 64])
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Number of independent profiling runs per model/config")
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-measure", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/analysis/efficiency")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    specs_by_name = {name: (cls, kw_fn) for name, cls, kw_fn in MODEL_SPECS}

    if args.param_matched:
        configs = [(name, PARAM_MATCHED_CONFIGS[name]) for name in specs_by_name if name in PARAM_MATCHED_CONFIGS]
        csv_name = "profile_param_matched.csv"
        summary_csv_name = "profile_param_matched_summary.csv"
    else:
        configs = [(name, d) for d in args.d_models for name in specs_by_name]
        csv_name = "profile_results.csv"
        summary_csv_name = "profile_results_summary.csv"

    results = []
    for batch_size in args.batch_sizes:
        for name, d_model in configs:
            for run_idx in range(1, args.n_runs + 1):
                cls, kw_fn = specs_by_name[name]
                try:
                    kwargs = kw_fn(d_model)
                    cell = cls(**kwargs)
                    result = profile_cell(cell, kwargs["d_input"], batch_size, args.device,
                                          n_warmup=args.n_warmup, n_measure=args.n_measure)
                    results.append({
                        "model": name,
                        "d_model": d_model,
                        "batch_size": batch_size,
                        "run": run_idx,
                        **result,
                    })
                    print(f"  {name:<25s} d={d_model:<4d} bs={batch_size:<4d} run={run_idx:<2d} "
                          f"params={result['params']:>10,d} "
                          f"fwd={result['fwd_ms']:.2f}ms fwd+bwd={result['fwd_bwd_ms']:.2f}ms")
                except Exception as e:
                    print(f"  {name:<25s} d={d_model:<4d} bs={batch_size:<4d} run={run_idx:<2d} FAILED: {e}")
                finally:
                    if args.device == "cuda":
                        torch.cuda.empty_cache()

    csv_path = os.path.join(args.output_dir, csv_name)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {csv_path}")

    summary_rows = []
    for batch_size in args.batch_sizes:
        for name, d_model in configs:
            rows = [
                row for row in results
                if row["model"] == name and row["d_model"] == d_model and row["batch_size"] == batch_size
            ]
            if not rows:
                continue
            summary_rows.append({
                "model": name,
                "d_model": d_model,
                "batch_size": batch_size,
                "n_runs": len(rows),
                "params": rows[0]["params"],
                "fwd_ms_mean": mean(row["fwd_ms"] for row in rows),
                "fwd_ms_std": _std([row["fwd_ms"] for row in rows]),
                "fwd_bwd_ms_mean": mean(row["fwd_bwd_ms"] for row in rows),
                "fwd_bwd_ms_std": _std([row["fwd_bwd_ms"] for row in rows]),
                "state_mem_kb": rows[0]["state_mem_kb"],
                "peak_mem_mb_mean": mean(row["peak_mem_mb"] for row in rows),
                "peak_mem_mb_std": _std([row["peak_mem_mb"] for row in rows]),
            })
    summary_csv_path = os.path.join(args.output_dir, summary_csv_name)
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved {summary_csv_path}")


if __name__ == "__main__":
    main()

"""Merge separately evaluated per-model transfer metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_transfer_metrics import write_csv, write_latex_table


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("outputs/analysis/sequential_split_mnist/transfer_chunks"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/sequential_split_mnist"),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main():
    args = parse_args()
    chunks_dir = resolve(args.chunks_dir)
    output_dir = resolve(args.output_dir)
    payloads = [
        json.loads((chunks_dir / str(index) / "transfer_metrics.json").read_text())
        for index in range(15)
    ]
    results = [payload["models"][0] for payload in payloads]
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = payloads[0]["protocol"]
    protocol["fwt"] = "R_A,B - bar_b_B"
    protocol["matrix_entries"] = {
        "bar_b_B": "set_b_baseline",
        "R_A,A": "accuracy_immediate",
        "R_A,B": "accuracy_set_b_before_exemplars",
        "R_B,A": "accuracy_retention",
        "R_B,B": "accuracy_set_b",
    }
    payload = {"protocol": protocol, "models": results}
    (output_dir / "transfer_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    write_csv(results, output_dir / "transfer_metrics.csv")
    write_latex_table(results, output_dir / "transfer_metrics_table.tex")


if __name__ == "__main__":
    main()

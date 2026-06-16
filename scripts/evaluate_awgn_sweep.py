from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_name(ratio: float, snr: int, seed: int) -> str:
    return f"cifar10_awgn_ratio-{ratio:.6f}_snr-{snr:g}_seed-{seed}"


def ratio_label(ratio: float) -> str:
    if abs(ratio - 1 / 12) < 1e-9:
        return "1-12"
    if abs(ratio - 1 / 6) < 1e-9:
        return "1-6"
    return f"{ratio:.6f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all AWGN models used in paper Fig. 4")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/fig4"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--steps", type=int, default=600_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    for ratio in (1 / 12, 1 / 6):
        for train_snr in (1, 4, 7, 13, 19):
            checkpoint = (
                args.runs_dir
                / run_name(ratio, train_snr, args.seed)
                / f"checkpoint-{args.steps}.pt"
            )
            if not checkpoint.exists():
                raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
            output = (
                args.results_dir
                / f"ratio-{ratio_label(ratio)}_train-snr-{train_snr}.csv"
            )
            command = [
                sys.executable,
                "evaluate.py",
                str(checkpoint),
                "--test-snrs",
                "1",
                "4",
                "7",
                "10",
                "13",
                "16",
                "19",
                "22",
                "25",
                "--repeats",
                str(args.repeats),
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
                "--data-dir",
                str(args.data_dir),
                "--output",
                str(output),
            ]
            print(" ".join(command), flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

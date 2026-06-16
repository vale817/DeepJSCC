from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt


def read_curve(path: Path) -> tuple[list[float], list[float]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return (
        [float(row["test_snr_db"]) for row in rows],
        [float(row["mean_psnr_db"]) for row in rows],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the CIFAR-10 AWGN Fig. 4 reproduction")
    parser.add_argument("--results-dir", type=Path, default=Path("results/fig4"))
    parser.add_argument("--output", type=Path, default=Path("results/fig4-awgn.png"))
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    for axis, ratio_key, title in [
        (axes[0], "1-12", "AWGN channel (k/n=1/12)"),
        (axes[1], "1-6", "AWGN channel (k/n=1/6)"),
    ]:
        for train_snr in (1, 4, 7, 13, 19):
            path = args.results_dir / f"ratio-{ratio_key}_train-snr-{train_snr}.csv"
            x, y = read_curve(path)
            axis.plot(x, y, marker="o", label=f"SNRtrain={train_snr} dB")
        axis.set_title(title)
        axis.set_xlabel("SNRtest (dB)")
        axis.set_ylabel("PSNR (dB)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()

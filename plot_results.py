from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt


def parse_series(value: str) -> tuple[str, Path]:
    try:
        label, path = value.rsplit("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("series must be LABEL=CSV_PATH") from error
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DeepJSCC PSNR versus test SNR")
    parser.add_argument("--series", action="append", type=parse_series, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/figure-4-reproduction.png"))
    parser.add_argument("--title", default="DeepJSCC over AWGN")
    args = parser.parse_args()

    for label, path in args.series:
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        x = [float(row["test_snr_db"]) for row in rows]
        y = [float(row["mean_psnr_db"]) for row in rows]
        plt.plot(x, y, marker="o", label=label)

    plt.xlabel("Test SNR (dB)")
    plt.ylabel("Average PSNR (dB)")
    plt.title(args.title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()

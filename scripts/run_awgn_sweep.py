from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the models needed for paper Fig. 4")
    parser.add_argument("--steps", type=int, default=600_000)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    for ratio in (1 / 12, 1 / 6):
        for snr in (1, 4, 7, 13, 19):
            command = [
                sys.executable,
                "train.py",
                "--ratio",
                str(ratio),
                "--snr",
                str(snr),
                "--steps",
                str(args.steps),
                "--data-dir",
                str(args.data_dir),
                "--output-dir",
                str(args.output_dir),
            ]
            if args.download:
                command.append("--download")
            print(" ".join(command), flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

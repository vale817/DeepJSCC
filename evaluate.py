from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from deepjscc.data import cifar10_loaders
from deepjscc.metrics import per_image_psnr
from deepjscc.model import DeepJSCC
from train import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DeepJSCC across test SNRs")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--test-snrs", type=float, nargs="+", default=[1, 4, 7, 10, 13, 16, 19, 22, 25])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results.csv"))
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = choose_device()
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = saved["args"]
    model = DeepJSCC(train_args["ratio"], train_args["channel"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    _, test_loader = cifar10_loaders(
        args.data_dir, args.batch_size, args.workers, args.download
    )

    rows = []
    for snr in args.test_snrs:
        psnr_sum = 0.0
        count = 0
        for images, _ in test_loader:
            images = images.to(device)
            for _ in range(args.repeats):
                reconstruction = model(images, snr)
                psnr = per_image_psnr(images, reconstruction)
                psnr_sum += psnr.sum().item()
                count += psnr.numel()
        mean_psnr = psnr_sum / count
        rows.append({"test_snr_db": snr, "mean_psnr_db": mean_psnr})
        print(f"test_snr_db={snr:g} mean_psnr_db={mean_psnr:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

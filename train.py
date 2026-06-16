from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from deepjscc.data import cifar10_loaders
from deepjscc.model import DeepJSCC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the original CIFAR-10 DeepJSCC model")
    parser.add_argument("--ratio", type=float, default=1 / 12)
    parser.add_argument("--snr", type=float, default=7.0)
    parser.add_argument("--channel", choices=["awgn", "rayleigh"], default="awgn")
    parser.add_argument("--steps", type=int, default=600_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-drop-step", type=int, default=500_000)
    parser.add_argument("--lr-after-drop", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device()

    train_loader, _ = cifar10_loaders(
        args.data_dir, args.batch_size, args.workers, args.download
    )
    model = DeepJSCC(args.ratio, args.channel).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    start_step = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = saved["step"]

    run_name = f"cifar10_{args.channel}_ratio-{args.ratio:.6f}_snr-{args.snr:g}_seed-{args.seed}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(vars(args) | {"device": str(device)}, default=str, indent=2)
    )

    model.train()
    iterator = iter(train_loader)
    for step in range(start_step + 1, args.steps + 1):
        if step == args.lr_drop_step:
            for group in optimizer.param_groups:
                group["lr"] = args.lr_after_drop
        try:
            images, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, _ = next(iterator)
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        reconstruction = model(images, args.snr)
        # The paper denormalizes decoder outputs to [0, 255] before MSE.
        loss = loss_fn(reconstruction * 255.0, images * 255.0)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 1000 == 0:
            print(f"step={step} mse={loss.item():.8f} lr={optimizer.param_groups[0]['lr']}")
        if step % 50_000 == 0 or step == args.steps:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "step": step,
                },
                run_dir / f"checkpoint-{step}.pt",
            )


if __name__ == "__main__":
    main()

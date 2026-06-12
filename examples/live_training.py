#live_training
#author: Facundo Franchino
"""
hear a flamo model change in the daw while it trains.

setup, once:
  1. build the hot-reload clap plugin (faust/architecture/clap) and
     load it on a track in your daw
  2. run this script inside an environment with flamo, torch and
     rt-fdn installed

every optimiser step re-emits the model to faust and atomically
rewrites the watched .dsp; the plugin reloads it in ~100 ms. the
macro-control knobs keep their positions across reloads (the plugin
maps parameters by address), so you can ride rt60 and dry/wet while
the optimisation runs.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict

import torch
from flamo.processor import dsp, system

from rt_fdn import HotReload


def build_fdn(fs: int, nfft: int, n: int = 4):
    """small fdn: B (n x 1) -> delays -> orthogonal feedback -> C (1 x n)."""
    delays = dsp.parallelDelay(size=(n,), max_len=3000, nfft=nfft,
                               isint=True, unit=1, fs=fs)
    delays.assign_value(torch.tensor(
        [887, 1109, 1361, 1693][:n], dtype=torch.float32) / fs)
    feedback = dsp.Matrix(size=(n, n), nfft=nfft, matrix_type="orthogonal")
    in_gain = dsp.Gain(size=(n, 1), nfft=nfft)
    in_gain.assign_value(torch.ones(n, 1))
    out_gain = dsp.Gain(size=(1, n), nfft=nfft, requires_grad=True)
    loop = system.Recursion(
        fF=system.Series(OrderedDict({"delays": delays})), fB=feedback)
    model = system.Series(OrderedDict(
        {"in": in_gain, "loop": loop, "out": out_gain}))
    return model, out_gain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs", type=int, default=48000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--dsp-path", default="/tmp/rt-fdn-live.dsp",
                        help="where the watched .dsp is written")
    args = parser.parse_args()

    nfft = 2**12
    model, out_gain = build_fdn(args.fs, nfft)

    live = HotReload(
        args.fs, name="LiveTraining", dsp_path=args.dsp_path,
        controls={"rt60": True, "dry_wet": True},
    )

    #toy objective: drive the broadband output level towards a target.
    #replace with your actual loss (target ir, edc match, ...)
    optimiser = torch.optim.Adam(out_gain.parameters(), lr=args.lr)
    x = torch.zeros(1, nfft // 2 + 1, 1, dtype=torch.complex64)
    x[:, :, 0] = 1.0
    target = torch.full((1, nfft // 2 + 1, 1), 0.1, dtype=torch.complex64)

    for step in range(args.steps):
        optimiser.zero_grad()
        loss = torch.mean(torch.abs(model(x) - target) ** 2)
        loss.backward()
        optimiser.step()
        published = live.update(model)
        if step % 10 == 0:
            mark = " -> published" if published else ""
            print(f"step {step:4d}  loss {loss.item():.6f}{mark}")

    live.update(model, force=True)
    print(f"final model published to {args.dsp_path}")


if __name__ == "__main__":
    main()

#export_plugin
#author: Facundo Franchino
"""
one call from a flamo model to an installed, certified vst3/au plugin.

requires (macos): the faust distribution (faust2juce), a juce checkout,
projucer, and xcode. run inside an environment with flamo, torch and
rt-fdn installed:

    python export_plugin.py --name MyReverb \
        --juce-modules ~/JUCE/modules --build

the exporter writes the .dsp, computes the stability certificate
(refusing to build anything not provably stable), generates the juce
project, compiles vst3 + au in release, and juce's copy step installs
them into ~/Library/Audio/Plug-Ins. rescan plugins in the daw.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict

import torch
from flamo.processor import dsp, system

from rt_fdn import flamo_to_json, export_juce


def build_fdn(fs: int, nfft: int, n: int = 4):
    """stereo fdn: B (n x 2) -> delays -> orthogonal feedback -> C (2 x n).

    stands in for your trained model: load a checkpoint instead.
    """
    delays = dsp.parallelDelay(size=(n,), max_len=3000, nfft=nfft,
                               isint=True, unit=1, fs=fs)
    delays.assign_value(torch.tensor(
        [887, 1109, 1361, 1693][:n], dtype=torch.float32) / fs)
    ortho = dsp.Matrix(size=(n, n), nfft=nfft, matrix_type="orthogonal")
    b_in = dsp.Gain(size=(n, 2), nfft=nfft)
    b_in.assign_value(torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.3], [0.3, 0.7]][:n]))
    c_out = dsp.Gain(size=(2, n), nfft=nfft)
    c_out.assign_value(torch.tensor(
        [[0.5, 0.0, 0.5, 0.0], [0.0, 0.5, 0.0, 0.5]])[:, :n])
    loop = system.Recursion(
        fF=system.Series(OrderedDict({"delays": delays})), fB=ortho)
    return system.Series(OrderedDict(
        {"b": b_in, "loop": loop, "c": c_out}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="FlamoReverb")
    parser.add_argument("--fs", type=int, default=48000)
    parser.add_argument("--out", default="exported")
    parser.add_argument("--juce-modules", default="~/JUCE/modules",
                        help="path to the juce modules folder")
    parser.add_argument("--manufacturer", default="rt-fdn")
    parser.add_argument("--build", action="store_true",
                        help="also compile and install vst3 + au (macos)")
    args = parser.parse_args()

    model = build_fdn(args.fs, nfft=2**15)

    result = export_juce(
        flamo_to_json(model, args.fs, name=args.name),
        args.out,
        name=args.name,
        controls={"rt60": True, "dry_wet": True, "pre_delay": True},
        juce_modules=args.juce_modules,
        manufacturer=args.manufacturer,
        build=args.build,
    )

    print("verdict:    ", result["verdict"])
    print("dsp:        ", result["dsp"])
    print("certificate:", result["certificate"])
    print("project:    ", result["project"])
    if args.build:
        for p in result["installed"]:
            print("installed:  ", p)
        print("rescan plugins in your daw "
              "(logic: restart, or killall -9 AudioComponentRegistrar)")


if __name__ == "__main__":
    main()

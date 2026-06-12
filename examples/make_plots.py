#make_plots
#author: Facundo Franchino
"""
generate the equivalence and rt60-validation figures in plots/.

two figures:
  ir_match.png        flamo vs compiled faust on one representative
                      path: overlay on top (exact match = traces
                      coincide), difference below in db re peak (the
                      gap is the story: ~100 db = float32 noise)
  rt60_validation.png schroeder energy decay of the compiled plugin
                      with the rt60 knob at 0.5 s, against the ideal
                      -120 db/s line and the lossless reference

requires faust2plot on PATH and an environment with flamo, torch,
matplotlib and rt-fdn installed.

alignment: the emitted code compensates the ~ operator inside the
loop and re-delays the recursion output by one sample, so arrivals
match flamo sample-exactly and no alignment is applied here.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from flamo.processor import dsp, system

from rt_fdn import flamo_to_json, json_to_faust

FS = 48000
NFFT = 2**15
N = 4
OUTDIR = Path(__file__).resolve().parent.parent / "plots"


def build_model(fb_scale: float):
    torch.manual_seed(3)
    delays = dsp.parallelDelay(size=(N,), max_len=3000, nfft=NFFT,
                               isint=True, unit=1, fs=FS)
    delays.assign_value(torch.tensor(
        [887, 1109, 1361, 1693], dtype=torch.float32) / FS)
    ortho = dsp.Matrix(size=(N, N), nfft=NFFT, matrix_type="orthogonal")
    fb = dsp.Gain(size=(N, N), nfft=NFFT)
    fb.assign_value(fb_scale * ortho.map(ortho.param).detach())
    b_in = dsp.Gain(size=(N, 2), nfft=NFFT)
    b_in.assign_value(torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.3], [0.3, 0.7]]))
    c_out = dsp.Gain(size=(2, N), nfft=NFFT)
    c_out.assign_value(torch.tensor(
        [[0.5, 0.0, 0.5, 0.0], [0.0, 0.5, 0.0, 0.5]]))
    loop = system.Recursion(
        fF=system.Series(OrderedDict({"delays": delays})), fB=fb)
    return system.Series(OrderedDict(
        {"b": b_in, "loop": loop, "c": c_out}))


def flamo_irs(model) -> np.ndarray:
    """2x2 ir matrix [input][output][time] via the frequency domain."""
    irs = np.zeros((2, 2, NFFT))
    for j in range(2):
        x = torch.zeros(1, NFFT // 2 + 1, 2, dtype=torch.complex64)
        x[:, :, j] = 1.0
        y = model(x).detach()
        irs[j] = torch.fft.irfft(y, n=NFFT, dim=1).numpy()[0].T
    return irs


def run_faust(code: str, feed: str, label: str, workdir: Path,
              n: int) -> np.ndarray:
    """wrap with an impulse source, compile, return (channels, n)."""
    m = re.search(r"process\s*=\s*(.+?);", code, re.DOTALL)
    wrapped = code[:m.start()] + (
        f"fdn = {m.group(1).strip()};\nimpulse = 1 - 1';\n"
        f"process = {feed} : fdn;"
    ) + code[m.end():]
    p = workdir / f"{label}.dsp"
    p.write_text(wrapped)
    subprocess.run(["faust2plot", str(p)], check=True,
                   capture_output=True, timeout=120)
    r = subprocess.run([str(p.with_suffix("")), "-n", str(n), "-r", str(FS)],
                       check=True, capture_output=True, text=True, timeout=60)
    #parse only the matlab vector body, the boilerplate contains digits
    body = r.stdout.split("faustout = [", 1)[1].split("];", 1)[0]
    rows = [[float(v) for v in
             re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", ln)]
            for ln in body.strip().splitlines()]
    return np.array([r_ for r_ in rows if r_]).T


def edc_db(ir: np.ndarray) -> np.ndarray:
    e = np.cumsum(ir[::-1] ** 2)[::-1]
    return 10.0 * np.log10(np.maximum(e, 1e-30) / e[0])


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="rt-fdn-plots-"))

    model = build_model(fb_scale=0.5)
    fl = flamo_irs(model)
    code = json_to_faust(flamo_to_json(model, FS, name="StereoFDN"))

    #no alignment needed, see module docstring
    fa = run_faust(code, "(impulse, 0.0)", "in0", workdir, NFFT)

    #figure 1: one representative path (input 1 to output 1), overlay
    #on top, difference below. the other three stereo paths match
    #identically and are pinned by the test suite; one readable panel
    #beats four cluttered ones.
    a, b = fl[0, 0], fa[0]
    peak = np.max(np.abs(a))
    t_ms = np.arange(NFFT) / FS * 1000.0
    sl = slice(0, int(0.15 * FS))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0]},
    )
    ax1.plot(t_ms[sl], a[sl], lw=1.0, color="#1f77b4",
             label="FLAMO (frequency domain)")
    ax1.plot(t_ms[sl], b[sl], lw=1.0, ls="--", color="#ff7f0e",
             label="generated FAUST (compiled)")
    ax1.set_ylabel("amplitude")
    ax1.set_title("impulse response: the traces coincide", fontsize=11)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(alpha=0.3)

    ax2.plot(t_ms[sl], 20 * np.log10(np.maximum(np.abs(a[sl]) / peak, 1e-12)),
             lw=0.7, color="#1f77b4", alpha=0.7, label="response level")
    ax2.plot(t_ms[sl], 20 * np.log10(np.maximum(np.abs(a[sl] - b[sl]) / peak, 1e-12)),
             lw=0.7, color="#d62728", label="difference FLAMO vs FAUST")
    ax2.axhline(-80, color="grey", ls=":", lw=1)
    ax2.text(2, -75, "-80 dB", fontsize=8, color="grey")
    ax2.set_ylim(-140, 5)
    ax2.set_ylabel("dB re peak")
    ax2.set_xlabel("time (ms)")
    ax2.set_title("the difference sits about 100 dB below the response",
                  fontsize=11)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "ir_match.png", dpi=150)
    plt.close(fig)

    #figure 3: rt60 validation, lossless prototype for both variants
    model_ll = build_model(fb_scale=1.0)
    cfg = flamo_to_json(model_ll, FS, name="RT")
    ir_rt = run_faust(json_to_faust(cfg, controls={"rt60": {"init": 0.5}}),
                      "(impulse, 0.0)", "rt05", workdir, FS)[0]
    ir_ll = run_faust(json_to_faust(cfg), "(impulse, 0.0)",
                      "lossless", workdir, FS)[0]

    tt = np.arange(FS) / FS
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tt, edc_db(ir_ll), lw=1.2, color="#1f77b4",
            label="lossless prototype: reference, no decay expected")
    ax.plot(tt, edc_db(ir_rt), lw=1.2, color="#d62728",
            label="rt60 control at 0.5 s: should follow the dashed line")
    ax.plot(tt, -120.0 * tt, ls="--", lw=1, color="black",
            label="ideal -120 dB/s (RT60 = 0.5 s)")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-120, 3)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("energy decay (dB)")
    ax.set_title("rt60 macro-control: Schroeder decay on compiled FAUST")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / "rt60_validation.png", dpi=150)
    plt.close(fig)

    for f in sorted(OUTDIR.glob("*.png")):
        print("written", f)


if __name__ == "__main__":
    main()

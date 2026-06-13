#make_plots
#author: Facundo Franchino
"""
generate the equivalence and rt60-validation figures in plots/.

two figures, styled for the paper (serif, no internal titles)
  edc_match.png       schroeder energy decay of flamo vs the compiled
                      faust, overlaid. the curves coincide, which is
                      the whole message. sample-level agreement is
                      stated numerically in the text, not plotted
  rt60_validation.png energy decay of the compiled plugin with the
                      rt60 knob at 0.5 s, against the ideal
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

#paper styling, matches a times-set document
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "legend.fontsize": 9.5,
    "axes.linewidth": 0.8,
})
import numpy as np
import torch
from flamo.processor import dsp, system

import rt_fdn

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
    code = rt_fdn.json_to_faust(rt_fdn.flamo_to_json(model, FS, name="StereoFDN"))

    #no alignment needed, see module docstring
    fa = run_faust(code, "(impulse, 0.0)", "in0", workdir, NFFT)

    #figure 1, energy decay overlay. one concept, two curves that
    #coincide. readable at first sight by anyone who reads edcs.
    a, b = fl[0, 0], fa[0]
    n_cmp = 24000
    tt = np.arange(n_cmp) / FS

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(tt, edc_db(a[:n_cmp]), lw=1.4, color="black",
            label="FLAMO (frequency domain)")
    ax.plot(tt, edc_db(b[:n_cmp]), lw=1.4, ls=(0, (4, 3)), color="#c1272d",
            label="generated FAUST (compiled)")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-80, 2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("energy decay (dB)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "edc_match.png", dpi=300)
    plt.close(fig)

    #figure 2, rt60 validation, lossless prototype for both variants
    model_ll = build_model(fb_scale=1.0)
    cfg = rt_fdn.flamo_to_json(model_ll, FS, name="RT")
    ir_rt = run_faust(rt_fdn.json_to_faust(cfg, controls={"rt60": {"init": 0.5}}),
                      "(impulse, 0.0)", "rt05", workdir, FS)[0]
    ir_ll = run_faust(rt_fdn.json_to_faust(cfg), "(impulse, 0.0)",
                      "lossless", workdir, FS)[0]

    tt = np.arange(FS) / FS
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(tt, edc_db(ir_rt), lw=1.4, color="black",
            label="control at 0.5 s")
    ax.plot(tt, -120.0 * tt, lw=1.2, ls=(0, (4, 3)), color="#c1272d",
            label="ideal, RT60 of 0.5 s")
    ax.plot(tt, edc_db(ir_ll), lw=1.2, color="0.65",
            label="lossless prototype")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-100, 2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("energy decay (dB)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "rt60_validation.png", dpi=300)
    plt.close(fig)

    for f in sorted(OUTDIR.glob("*.png")):
        print("written", f)


if __name__ == "__main__":
    main()

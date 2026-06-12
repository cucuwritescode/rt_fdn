<div align="center">

# rt-fdn

**real-time deployment of differentiable audio graphs via FAUST**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/licence-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-195%20passing-brightgreen.svg)](#testing)
![Status](https://img.shields.io/badge/status-beta-yellow.svg)

*bridge the gap between differentiable audio research and deployable real-time plugins*

</div>

---

## the problem

researchers design and optimise FDNs using [FLAMO](https://github.com/gdalsanto/flamo)'s differentiable audio framework, but deploying these as real-time plugins requires manual reimplementation. this is error-prone and creates a gap between research prototypes and usable tools.

```
before:   FLAMO model (PyTorch)  →  ???  →  real-time plugin
                                     ↑
                                manual rewrite

after:    FLAMO model (PyTorch)  →  rt_fdn  →  FAUST  →  plugin
```

## how it works

```
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│    FLAMO     │  ───▶ │     JSON     │  ───▶ │    FAUST     │
│    model     │  ◀─── │    config    │       │    code      │
│  (PyTorch)   │       │              │       │   (.dsp)     │
└──────────────┘       └──────────────┘       └──────────────┘
       flamo_to_json() ──▶     json_to_faust() ──▶
       json_to_flamo() ◀──
         ╰───────────── flamo_to_faust() ──────────────╯
```

the pipeline traverses a FLAMO model graph, extracts all parameters (delays, gains, matrices, filters), serialises them to a JSON intermediate representation, and generates valid FAUST DSP code. extraction is map-aware: matrix types with non-identity maps (orthogonal, hadamard, householder) serialise the effective matrix the model applies, with the raw trainable weights preserved for round-tripping. `json_to_flamo` reconstructs the original model from the config.

on top of the codegen core:

- `HotReload` republishes the model to a running FAUST plugin during training, so you hear the optimisation while it runs
- macro-controls (`rt60`, `dry_wet`, `pre_delay`) add performance knobs to the generated plugin without touching the trained parameters
- `certify` computes a stability certificate for every feedback loop, written as `.cert.json` next to the `.dsp`
- `export_juce` turns a config into an installed VST3/AU plugin in one call

## installation

```bash
pip install -e .
```

for full FLAMO model support (requires PyTorch):

```bash
pip install -e ".[full]"
```

building plugins additionally requires the [FAUST](https://faust.grame.fr) distribution and [JUCE](https://juce.com).

## quick start

```python
from rt_fdn import flamo_to_faust

#given a trained FLAMO model and sample rate
faust_code = flamo_to_faust(model, fs=48000, name="MyReverb")

#write to file
with open("reverb.dsp", "w") as f:
    f.write(faust_code)
```

or use the two-step pipeline for inspection:

```python
from rt_fdn import flamo_to_json, json_to_faust

config = flamo_to_json(model, fs=48000, name="MyReverb")
faust_code = json_to_faust(config, controls={"rt60": True, "dry_wet": True})
```

### hear it while it trains

```python
from rt_fdn import HotReload

live = HotReload(fs=48000, name="MyReverb", controls={"rt60": True})
for step in range(n_steps):
    loss = criterion(model(x), target)
    loss.backward()
    optimiser.step()
    live.update(model)
live.update(model, force=True)
```

the hot-reload CLAP plugin (FAUST interpreter plus file watcher) lives in `faust/architecture/clap/`. reloads take about 100 ms and knob positions survive them. full script: `examples/live_training.py`.

### ship it

```python
from rt_fdn import flamo_to_json, export_juce

export_juce(
    flamo_to_json(model, fs=48000, name="MyReverb"),
    "exported/", name="MyReverb",
    controls={"rt60": True, "dry_wet": True, "pre_delay": True},
    juce_modules="~/JUCE/modules",
    build=True,
)
```

one call: FAUST generation, stability certificate, JUCE project, release build, install into the user plugin folders (macOS). the export refuses to build a model whose certificate says `unstable` or `not-certified`; pass `strict=False` to override. full script: `examples/export_plugin.py`.

### certify

```python
from rt_fdn import certify

cert = certify(config)
print(cert["verdict"])
```

the criterion is small-gain: the product of per-element spectral norms around each feedback loop must stay below one at every frequency, evaluated on the parameter values as emitted (single precision). verdicts are `certified-stable`, `marginally-stable`, `indeterminate`, `not-certified`, `unstable`. a lossless prototype is marginally stable; with the `rt60` control it is certified at any knob position.

## equivalence

generated FAUST matches FLAMO sample-exactly, direct paths included. top: the two impulse responses overlaid (they coincide). bottom: their difference, about 100 dB below the response, which is float32 arithmetic noise rather than model mismatch. all four stereo paths match identically; the suite pins them.

<p align="center">
<img src="plots/ir_match.png" width="80%">
</p>

the rt60 macro-control on the compiled plugin, measured by Schroeder integration, follows the ideal decay for the slider value:

<p align="center">
<img src="plots/rt60_validation.png" width="70%">
</p>

regenerate the figures with `python examples/make_plots.py`.

## supported modules

| FLAMO module | FAUST output | description |
|---|---|---|
| `parallelDelay` | `@(n)` / `de.fdelay` | integer or fractional sample delays |
| `Gain` / `Matrix` | sum-of-products function | mixing matrices (hoisted, map-aware) |
| `HouseholderMatrix` | sum-of-products function | emitted as the effective matrix |
| `parallelGain` | `*(g)` | per-channel diagonal gains |
| `parallelSOSFilter` | `fi.tf2(...)` | cascaded biquad filters |
| `Series` | `:` | sequential composition |
| `Parallel` | `,` / `:>` | side-by-side or summing |
| `Recursion` | `~` | feedback loops (FDN core) |
| `Biquad` / `SVF` | `fi.tf2` / `fi.svf.*` | single-channel filters |
| `Shell` | *(unwrapped)* | FFT wrapper skipped |

## testing

```bash
#unit tests (no external dependencies)
pytest tests/ -q --ignore=tests/integration

#integration tests (requires flamo venv + faust compiler)
pytest tests/integration/ -v
```

195 unit tests validate the full pipeline: map-aware parameter extraction, delay quantisation, SOS normalisation, gain classification, graph traversal, code generation, macro-control wiring, multichannel arities, hot-reload publishing, certificate verdicts, and export orchestration.

integration tests compare impulse responses between FLAMO (frequency domain) and generated FAUST (time domain) sample-by-sample.

## project structure

```
src/rt_fdn/
  codegen/
    flamo_to_json.py     parameter extraction and graph traversal
    json_to_faust.py     FAUST code generation and macro-controls
    json_to_flamo.py     model reconstruction from JSON config
    flamo_to_faust.py    convenience wrapper (both steps)
  hotreload.py           training-time live publishing
  certificate.py         small-gain stability certificate
  export.py              JUCE plugin export
examples/
    live_training.py
    export_plugin.py
    make_plots.py
tests/
    test_flamo_to_json.py
    test_json_to_faust.py
    test_flamo_to_faust.py
    test_param_extraction.py
    test_hotreload.py
    test_certificate.py
    test_export.py
    integration/
        test_ir_comparison.py
        generate_flamo_ir.py
```

## related projects

- [FLAMO](https://github.com/gdalsanto/flamo) — differentiable audio processing framework
- [pyFDN](https://github.com/artificial-audio/pyFDN) — python feedback delay networks
- [FAUST](https://faust.grame.fr/) — functional audio stream

## licence

MIT — see [LICENSE](LICENSE) for details.

#certificate
#author: Facundo Franchino
"""
stability certificate for generated fdn topologies

computes a formal sufficient condition for bibo stability of the
feedback loops in a json config, and emits a certificate dict that can
be written alongside the generated .dsp.


verdicts, from best to worst:

    certified-stable   loop gain bound < 1 at every grid frequency
    marginally-stable  bound equal to one within tolerance (lossless
                       prototype): bounded but non-decaying
    indeterminate      the loop contains elements this analysis cannot
                       bound (unknown module types, nested recursions)
    not-certified      bound exceeds one somewhere: stability is NOT
                       proven; it is not proven unstable either, since
                       small gain is only sufficient
    unstable           an iir section has poles on or outside the unit
                       circle: the realisation is internally unstable
                       regardless of the loop around it
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rt_fdn.codegen.json_to_faust import _normalise_controls

#classification tolerance around a loop gain of exactly one. wide
#enough to absorb float32 rounding of a genuinely orthogonal matrix
#(observed at the 1e-7 level), narrow enough that real attenuation
#or real growth is classified as such.
_MARGINAL_TOL = 1e-4

#severity order for aggregating per-loop verdicts into one
_SEVERITY = [
    "certified-stable",
    "marginally-stable",
    "indeterminate",
    "not-certified",
    "unstable",
]


def _worse(a: str, b: str) -> str:
    return a if _SEVERITY.index(a) >= _SEVERITY.index(b) else b


def _as_emitted(values: Any) -> np.ndarray:
    """round values through the emission chain: ten significant
    figures (json_to_faust._fmt) then single precision (faust default).

    returned as float64 so downstream linear algebra runs in double,
    but carrying exactly the values the compiled plugin multiplies by.
    """
    arr = np.asarray(values, dtype=np.float64)
    rounded = np.vectorize(lambda x: float(f"{x:.10g}"))(arr) if arr.size else arr
    return np.float32(rounded).astype(np.float64)


def _sos_response(section: list[float], omega: np.ndarray) -> np.ndarray:
    """complex frequency response of one normalised sos section.

    section is [b0, b1, b2, a1, a2] with a0 = 1 implicit, matching the
    json config and faust's fi.tf2.
    """
    b0, b1, b2, a1, a2 = section
    z1 = np.exp(-1j * omega)
    z2 = z1 * z1
    return (b0 + b1 * z1 + b2 * z2) / (1.0 + a1 * z1 + a2 * z2)


def _section_pole_modulus(section: list[float]) -> float:
    """largest pole modulus of one sos section (z^2 + a1 z + a2)."""
    a1, a2 = section[3], section[4]
    return float(np.max(np.abs(np.roots([1.0, a1, a2]))))


#walking a loop: each element contributes a per-frequency upper bound
#on its spectral norm. the walk returns an array over the grid, or
#None when an element cannot be bounded.

class _LoopWalk:
    """accumulates the small-gain bound for one feedback loop."""

    def __init__(self, omega: np.ndarray, fs: float, rt60_max_s: float | None):
        self.omega = omega
        self.fs = fs
        #worst-case rt60 slider position (longest decay = weakest
        #attenuation), None when the control is not requested
        self.rt60_max_s = rt60_max_s
        self.bound: np.ndarray | None = np.ones_like(omega)
        self.delays: list[float] = []
        self.notes: list[str] = []
        self.sigma_max_matrices: list[float] = []

    def _mul(self, factor: np.ndarray | float) -> None:
        if self.bound is not None:
            self.bound = self.bound * factor

    def _indeterminate(self, why: str) -> None:
        self.bound = None
        self.notes.append(why)

    def visit(self, node: dict[str, Any] | None) -> None:
        if node is None or self.bound is None:
            return
        node_type = node.get("type", "Leaf")

        if node_type == "Series":
            for child in node.get("children", []):
                self.visit(child)
            return
        if node_type == "Shell":
            for child in node.get("children", []):
                self.visit(child)
            return
        if node_type in ("Parallel", "Recursion"):
            #parallel routing and nested loops change the algebra:
            #a series product bound no longer applies
            self._indeterminate(
                f"loop contains a {node_type} node, "
                "small-gain product bound does not apply"
            )
            return

        self._visit_leaf(node)

    def _visit_leaf(self, node: dict[str, Any]) -> None:
        module_type = node.get("module_type", "")
        params = node.get("params", {})

        if module_type in ("parallelDelay", "Delay", "variableDelay",
                           "fractionalDelay"):
            samples = params.get("samples") or params.get("samples_fractional")
            if samples:
                self.delays.extend(float(s) for s in samples)
                #delays are unit modulus; the rt60 macro control adds a
                #per-line attenuation 10^(-3 m_i / (fs rt60)). at the
                #worst-case slider position the largest of these gains
                #(shortest line) bounds the diagonal's spectral norm.
                if self.rt60_max_s is not None:
                    m_min = min(float(s) for s in samples)
                    gain = 10.0 ** (-3.0 * m_min / (self.fs * self.rt60_max_s))
                    self._mul(gain)
            return

        if module_type in ("Gain", "Matrix", "HouseholderMatrix", "parallelGain"):
            if "matrix" in params:
                m = _as_emitted(params["matrix"])
                sigma = float(np.linalg.norm(m, 2))
                self.sigma_max_matrices.append(sigma)
                self._mul(sigma)
            elif "gains" in params:
                g = _as_emitted(params["gains"])
                self._mul(float(np.max(np.abs(g))))
            #no params: passthrough, factor one
            return

        if module_type == "parallelSOSFilter":
            sos = params.get("sos", [])
            if not sos:
                return
            #per frequency, the diagonal of per-channel cascades has
            #spectral norm max over channels of the cascade magnitude
            n_channels = len(sos[0])
            mags = np.ones((n_channels, len(self.omega)))
            for section_row in sos:
                for ch in range(n_channels):
                    coeffs = _as_emitted(section_row[ch]).tolist()
                    mags[ch] *= np.abs(_sos_response(coeffs, self.omega))
            self._mul(np.max(mags, axis=0))
            return

        if module_type == "Biquad":
            if "coeffs" in params:
                coeffs = _as_emitted(params["coeffs"]).tolist()
                self._mul(np.abs(_sos_response(coeffs, self.omega)))
                return
            keys = ("b0", "b1", "b2", "a1", "a2")
            if any(k in params for k in keys):
                coeffs = _as_emitted(
                    [params.get(k, 1.0 if k == "b0" else 0.0) for k in keys]
                ).tolist()
                self._mul(np.abs(_sos_response(coeffs, self.omega)))
                return
            return

        self._indeterminate(
            f"no spectral norm bound for module type "
            f"'{module_type}' (node '{node.get('name', '?')}')"
        )


def _collect(node: dict[str, Any] | None, node_type: str, found: list) -> None:
    """collect all nodes of a given type from the config tree."""
    if node is None:
        return
    if node.get("type") == node_type:
        found.append(node)
    for child in node.get("children", []):
        _collect(child, node_type, found)
    for key in ("fF", "fB"):
        _collect(node.get(key), node_type, found)


def _collect_sos_sections(node: dict[str, Any] | None, found: list) -> None:
    """collect every normalised sos section in the whole config."""
    if node is None:
        return
    params = node.get("params", {})
    for section_row in params.get("sos", []):
        for channel_coeffs in section_row:
            found.append(channel_coeffs)
    if "coeffs" in params:
        found.append(params["coeffs"])
    for child in node.get("children", []):
        _collect_sos_sections(child, found)
    for key in ("fF", "fB"):
        _collect_sos_sections(node.get(key), found)


def _certify_loop(
    loop: dict[str, Any],
    omega: np.ndarray,
    fs: float,
    rt60_max_s: float | None,
) -> dict[str, Any]:
    """certificate entry for one recursion node."""
    walk = _LoopWalk(omega, fs, rt60_max_s)
    walk.visit(loop.get("fF"))
    walk.visit(loop.get("fB"))

    entry: dict[str, Any] = {
        "name": loop.get("name", "recursion"),
        "delays_samples": walk.delays,
        "notes": walk.notes,
    }
    if rt60_max_s is not None:
        entry["rt60_control_worst_case_s"] = rt60_max_s

    #supplementary spectral radius of the feedback matrix, reported
    #because reviewers expect it, explicitly not the criterion
    fb = loop.get("fB") or {}
    fb_matrix = (fb.get("params") or {}).get("matrix")
    if fb_matrix is not None:
        m = _as_emitted(fb_matrix)
        if m.ndim == 2 and m.shape[0] == m.shape[1]:
            eigs = np.linalg.eigvals(m)
            entry["feedback_spectral_radius"] = float(np.max(np.abs(eigs)))
            entry["feedback_sigma_max"] = float(np.linalg.norm(m, 2))

    if walk.bound is None:
        entry["verdict"] = "indeterminate"
        return entry

    g_max = float(np.max(walk.bound))
    g_min = float(np.min(walk.bound))
    entry["loop_gain_bound_max"] = g_max
    entry["loop_gain_bound_min"] = g_min

    if g_max < 1.0 - _MARGINAL_TOL:
        entry["verdict"] = "certified-stable"
    elif g_max <= 1.0 + _MARGINAL_TOL:
        entry["verdict"] = "marginally-stable"
        entry["notes"].append(
            "loop gain bound is one within tolerance: bounded but "
            "non-decaying (lossless prototype)"
        )
    else:
        entry["verdict"] = "not-certified"
        entry["notes"].append(
            "loop gain bound exceeds one: stability is not proven; "
            "small gain is sufficient, not necessary"
        )

    #rt60 estimate from the per-pass gain bound and the mean loop time:
    #a pass of m samples attenuated by g repeats every m/fs seconds,
    #so level decays 60 db after t = -3 m / (fs log10 g)
    if walk.delays and g_max < 1.0 - _MARGINAL_TOL:
        m_mean = float(np.mean(walk.delays))
        entry["rt60_estimate_s"] = {
            "max": float(-3.0 * m_mean / (fs * np.log10(g_max))),
            "min": float(-3.0 * m_mean / (fs * np.log10(max(g_min, 1e-12)))),
        }

    return entry


#public api

def certify(
    config: dict[str, Any],
    *,
    n_freq: int = 1024,
    controls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """compute a stability certificate for a json config dict.

    analyses every feedback loop with a small-gain bound over a
    frequency grid and checks every iir section's poles, using the
    parameter values exactly as emitted to faust (ten significant
    figures, single precision).

    parameters
    ----------
    config : dict
        json config dict as produced by flamo_to_json(). macro
        controls embedded under config["controls"] are taken into
        account (the rt60 control adds guaranteed loop attenuation
        at any slider position).
    n_freq : int
        number of grid points on [0, pi] for frequency-dependent
        elements.
    controls : dict, optional
        call-time macro controls, merged over any embedded in the
        config with the same precedence as json_to_faust(), so the
        certificate describes the same dsp the codegen emits.

    returns
    -------
    certificate : dict
        json-serialisable certificate with an overall verdict, one
        entry per feedback loop, and a global filter pole check.
    """
    fs = float(config.get("fs", 48000))
    omega = np.linspace(0.0, np.pi, int(n_freq))

    #worst-case rt60 slider position when the control is requested.
    #merge order mirrors json_to_faust: call-time entries win.
    requested = dict(config.get("controls") or {})
    if controls:
        requested.update(controls)
    normalised = _normalise_controls(requested)
    rt60_max_s = normalised["rt60"]["max"] if "rt60" in normalised else None

    #global filter pole check, loops or not
    sections: list = []
    _collect_sos_sections(config, sections)
    pole_moduli = [
        _section_pole_modulus(_as_emitted(s).tolist()) for s in sections
    ]
    max_pole = max(pole_moduli) if pole_moduli else 0.0
    filters_stable = all(p < 1.0 for p in pole_moduli)

    loops: list[dict] = []
    _collect(config, "Recursion", loops)
    loop_entries = [_certify_loop(lp, omega, fs, rt60_max_s) for lp in loops]

    verdict = "certified-stable"
    for entry in loop_entries:
        verdict = _worse(verdict, entry["verdict"])
    if not filters_stable:
        verdict = _worse(verdict, "unstable")

    certificate: dict[str, Any] = {
        "name": config.get("name", "untitled"),
        "fs": int(fs),
        "generated_by": "rt-fdn",
        "method": (
            "small-gain: product of per-element spectral norms over "
            f"{int(n_freq)} frequencies, sufficient condition for "
            "bibo stability"
        ),
        "precision": "values as emitted (10 significant figures, float32)",
        "verdict": verdict,
        "filters": {
            "n_sections": len(sections),
            "max_pole_modulus": float(max_pole),
            "all_sections_stable": bool(filters_stable),
        },
        "loops": loop_entries,
    }
    if not loops:
        certificate["notes"] = ["no feedback loops found"]
    return certificate


def write_certificate(
    config: dict[str, Any],
    dsp_path: str | Path,
    *,
    n_freq: int = 1024,
    controls: dict[str, Any] | None = None,
) -> Path:
    """write a .cert.json certificate next to a generated .dsp file.

    returns the certificate path. the certificate describes the same
    config the .dsp was generated from; the caller is responsible for
    keeping the two in step (including passing the same call-time
    controls that were given to json_to_faust).
    """
    cert = certify(config, n_freq=n_freq, controls=controls)
    path = Path(dsp_path).with_suffix(".cert.json")
    path.write_text(json.dumps(cert, indent=2) + "\n")
    return path

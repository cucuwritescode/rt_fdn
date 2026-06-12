#test_certificate
#author: Facundo Franchino
"""
tests for the small-gain stability certificate.

configs are built by hand to exercise each verdict class. the
certificate must reason about the values as emitted to faust, use
sigma_max (never the spectral radius) as the loop criterion, and
refuse to certify what it cannot bound.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from rt_fdn.certificate import certify, write_certificate


#config builders

HADAMARD4 = [
    [0.5, 0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5, 0.5],
]

DELAYS4 = [100, 200, 300, 400]


def _delay_leaf(samples):
    return {
        "type": "Leaf", "name": "delays", "module_type": "parallelDelay",
        "params": {"samples": list(samples)},
        "input_channels": len(samples), "output_channels": len(samples),
    }


def _matrix_leaf(matrix, name="fb"):
    return {
        "type": "Leaf", "name": name, "module_type": "Gain",
        "params": {"matrix": [list(r) for r in matrix]},
        "input_channels": len(matrix[0]), "output_channels": len(matrix),
    }


def _sos_leaf(b0, a1, n_ch):
    #one section per channel: h(z) = b0 / (1 + a1 z^-1)
    section = [[b0, 0.0, 0.0, a1, 0.0] for _ in range(n_ch)]
    return {
        "type": "Leaf", "name": "absorption", "module_type": "parallelSOSFilter",
        "params": {"sos": [section]},
        "input_channels": n_ch, "output_channels": n_ch,
    }


def _loop_config(matrix, samples=DELAYS4, sos=None, controls=None):
    ff_children = [_delay_leaf(samples)]
    if sos is not None:
        ff_children.append(sos)
    config = {
        "type": "Recursion",
        "name": "loop",
        "fF": {"type": "Series", "name": "fF", "children": ff_children},
        "fB": _matrix_leaf(matrix),
        "fs": 48000,
    }
    if controls is not None:
        config["controls"] = controls
    return config


#verdict classes

class TestVerdicts:
    def test_lossless_orthogonal_is_marginal(self):
        cert = certify(_loop_config(HADAMARD4))
        assert cert["verdict"] == "marginally-stable"
        loop = cert["loops"][0]
        assert loop["loop_gain_bound_max"] == pytest.approx(1.0, abs=1e-6)
        #spectral radius reported as supplementary information
        assert loop["feedback_spectral_radius"] == pytest.approx(1.0, abs=1e-6)

    def test_attenuated_matrix_is_certified(self):
        scaled = [[0.9 * v for v in row] for row in HADAMARD4]
        cert = certify(_loop_config(scaled))
        assert cert["verdict"] == "certified-stable"
        loop = cert["loops"][0]
        assert loop["loop_gain_bound_max"] == pytest.approx(0.9, abs=1e-6)

    def test_rt60_estimate_matches_jot_formula(self):
        scaled = [[0.9 * v for v in row] for row in HADAMARD4]
        cert = certify(_loop_config(scaled))
        est = cert["loops"][0]["rt60_estimate_s"]
        m_mean = float(np.mean(DELAYS4))
        expected = -3.0 * m_mean / (48000.0 * math.log10(0.9))
        assert est["max"] == pytest.approx(expected, rel=1e-6)

    def test_expanding_matrix_is_not_certified(self):
        scaled = [[1.1 * v for v in row] for row in HADAMARD4]
        cert = certify(_loop_config(scaled))
        assert cert["verdict"] == "not-certified"
        #not-certified is not a proof of instability, the note says so
        notes = " ".join(cert["loops"][0]["notes"])
        assert "not proven" in notes

    def test_passive_absorption_certifies_lossless_matrix(self):
        #h(z) = 0.5/(1 - 0.35 z^-1): dc gain 0.5/0.65 < 1, passive
        sos = _sos_leaf(0.5, -0.35, 4)
        cert = certify(_loop_config(HADAMARD4, sos=sos))
        assert cert["verdict"] == "certified-stable"
        loop = cert["loops"][0]
        assert loop["loop_gain_bound_max"] == pytest.approx(0.5 / 0.65, rel=1e-4)

    def test_unity_dc_absorption_stays_marginal(self):
        #h(z) = 0.65/(1 - 0.35 z^-1): dc gain exactly one, the loop
        #is lossless at dc and the certificate must say marginal
        sos = _sos_leaf(0.65, -0.35, 4)
        cert = certify(_loop_config(HADAMARD4, sos=sos))
        assert cert["verdict"] == "marginally-stable"

    def test_unstable_filter_section_is_unstable(self):
        #poles at |z| = sqrt(1.2) > 1: internally unstable realisation
        sos = {
            "type": "Leaf", "name": "bad", "module_type": "parallelSOSFilter",
            "params": {"sos": [[[1.0, 0.0, 0.0, 0.0, 1.2]] * 4]},
            "input_channels": 4, "output_channels": 4,
        }
        cert = certify(_loop_config(HADAMARD4, sos=sos))
        assert cert["verdict"] == "unstable"
        assert cert["filters"]["all_sections_stable"] is False
        assert cert["filters"]["max_pole_modulus"] == pytest.approx(
            math.sqrt(1.2), rel=1e-6
        )

    def test_unknown_module_in_loop_is_indeterminate(self):
        config = _loop_config(HADAMARD4)
        config["fF"]["children"].append({
            "type": "Leaf", "name": "mystery", "module_type": "NeuralBlock",
            "params": {}, "input_channels": 4, "output_channels": 4,
        })
        cert = certify(config)
        assert cert["verdict"] == "indeterminate"
        assert any("NeuralBlock" in n for n in cert["loops"][0]["notes"])

    def test_rt60_control_certifies_lossless_prototype(self):
        #the rt60 macro control attenuates every line at any slider
        #position, so a lossless prototype becomes certified
        cert = certify(_loop_config(HADAMARD4, controls={"rt60": True}))
        assert cert["verdict"] == "certified-stable"
        loop = cert["loops"][0]
        #worst case is the slider maximum (10 s default): the shortest
        #line keeps the largest gain 10^(-3*100/(48000*10))
        expected = 10.0 ** (-3.0 * 100 / (48000.0 * 10.0))
        assert loop["loop_gain_bound_max"] == pytest.approx(expected, rel=1e-6)
        assert loop["rt60_control_worst_case_s"] == 10.0

    def test_no_feedback_is_trivially_stable(self):
        config = dict(_matrix_leaf([[0.5, 0.5]], name="mix"))
        config["fs"] = 48000
        cert = certify(config)
        assert cert["verdict"] == "certified-stable"
        assert cert["loops"] == []
        assert "no feedback loops found" in cert["notes"]


#certificate artefact

class TestCertificateArtefact:
    def test_json_serialisable(self):
        cert = certify(_loop_config(HADAMARD4))
        parsed = json.loads(json.dumps(cert))
        assert parsed["verdict"] == "marginally-stable"

    def test_states_method_and_precision(self):
        cert = certify(_loop_config(HADAMARD4))
        assert "small-gain" in cert["method"]
        assert "float32" in cert["precision"]

    def test_write_certificate_next_to_dsp(self, tmp_path):
        dsp_path = tmp_path / "reverb.dsp"
        cert_path = write_certificate(_loop_config(HADAMARD4), dsp_path)
        assert cert_path == tmp_path / "reverb.cert.json"
        on_disk = json.loads(cert_path.read_text())
        assert on_disk["verdict"] == "marginally-stable"

    def test_delays_recorded(self):
        cert = certify(_loop_config(HADAMARD4))
        assert cert["loops"][0]["delays_samples"] == [100.0, 200.0, 300.0, 400.0]

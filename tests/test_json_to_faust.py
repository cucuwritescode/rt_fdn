#test_json_to_faust
#author: Facundo Franchino
"""
faust code generation from json config dicts

each test builds a minimal json config (matching flamo_to_json output)
and verifies that the generated faust code is correct and compilable.
"""

from __future__ import annotations

import pytest

from rt_fdn.codegen.json_to_faust import json_to_faust


#helpers

def _leaf(name: str, module_type: str, params: dict, n_ch: int = 4) -> dict:
    """shorthand for building a leaf node dict."""
    return {
        "type": "Leaf",
        "name": name,
        "module_type": module_type,
        "params": params,
        "input_channels": n_ch,
        "output_channels": n_ch,
    }


def _wrap_config(node: dict, name: str = "test", fs: int = 48000) -> dict:
    """wrap a node in a minimal root config with fs and name."""
    config = dict(node)
    config["fs"] = fs
    config["name"] = name
    return config


#header and structure tests

class TestHeader:
    def test_includes_import(self):
        config = _wrap_config({"type": "Shell", "children": []})
        code = json_to_faust(config)
        assert 'import("stdfaust.lib");' in code

    def test_includes_name(self):
        config = _wrap_config({"type": "Shell", "children": []}, name="MyReverb")
        code = json_to_faust(config)
        assert "//MyReverb" in code

    def test_includes_sample_rate(self):
        config = _wrap_config({"type": "Shell", "children": []}, fs=44100)
        code = json_to_faust(config)
        assert "44100" in code

    def test_has_process(self):
        config = _wrap_config({"type": "Shell", "children": []})
        code = json_to_faust(config)
        assert "process = " in code

    def test_empty_shell_is_wire(self):
        config = _wrap_config({"type": "Shell", "children": []})
        code = json_to_faust(config)
        assert "process = _;" in code


#delay tests

class TestDelayCodegen:
    def test_single_delay(self):
        node = _leaf("d", "parallelDelay", {"samples": [1000]}, n_ch=1)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "@(1000)" in code

    def test_multiple_delays(self):
        node = _leaf("d", "parallelDelay", {"samples": [1103, 1447, 1811, 2137]})
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "@(1103)" in code
        assert "@(1447)" in code
        assert "@(1811)" in code
        assert "@(2137)" in code

    def test_delays_joined_parallel(self):
        #four delays should be composed with ,
        node = _leaf("d", "parallelDelay", {"samples": [100, 200, 300, 400]})
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "@(100) , @(200) , @(300) , @(400)" in code


#diagonal gain tests

class TestDiagonalGainCodegen:
    def test_single_gain(self):
        node = _leaf("g", "parallelGain", {"gains": [0.5]}, n_ch=1)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "*(0.5)" in code

    def test_multiple_gains(self):
        node = _leaf("g", "parallelGain", {"gains": [0.1, 0.2, 0.3]}, n_ch=3)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "*(0.1)" in code
        assert "*(0.2)" in code
        assert "*(0.3)" in code

    def test_gain_module_diagonal(self):
        #Gain with "gains" key (classified as diagonal by flamo_to_json)
        node = _leaf("g", "Gain", {"gains": [1.0, 0.5]}, n_ch=2)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "*(1)" in code
        assert "*(0.5)" in code


#matrix gain tests

class TestMatrixGainCodegen:
    def test_identity_matrix(self):
        matrix = [[1.0, 0.0], [0.0, 1.0]]
        node = _leaf("m", "Gain", {"matrix": matrix}, n_ch=2)
        config = _wrap_config(node, name="m")
        code = json_to_faust(config)
        #should produce a function definition
        assert "m(x0, x1)" in code
        #identity: row0 = x0, row1 = x1
        assert "x0" in code
        assert "x1" in code

    def test_hadamard_matrix(self):
        matrix = [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5],
        ]
        node = _leaf("feedback", "Gain", {"matrix": matrix})
        config = _wrap_config(node, name="feedback")
        code = json_to_faust(config)
        #function should be hoisted as a definition
        assert "feedback(x0, x1, x2, x3)" in code
        #check that it appears as a definition line ending with ;
        assert "feedback(x0, x1, x2, x3) =" in code

    def test_zero_row_emits_zero(self):
        matrix = [[0.0, 0.0], [1.0, 0.0]]
        node = _leaf("m", "Gain", {"matrix": matrix}, n_ch=2)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "0.0" in code

    def test_matrix_module_type(self):
        #Matrix type should behave identically to Gain with matrix params
        matrix = [[0.7, 0.3], [0.3, 0.7]]
        node = _leaf("mix", "Matrix", {"matrix": matrix}, n_ch=2)
        config = _wrap_config(node, name="mix")
        code = json_to_faust(config)
        assert "mix(x0, x1)" in code

    def test_householder_emits_full_matrix_function(self):
        #flamo_to_json serialises the effective I - 2uu^T for
        #HouseholderMatrix, so the emitter must produce an N-input
        #mixing function, never a column vector
        matrix = [
            [0.5, -0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5, -0.5],
            [-0.5, -0.5, -0.5, 0.5],
        ]
        node = _leaf("fb", "HouseholderMatrix", {"matrix": matrix})
        config = _wrap_config(node, name="fb")
        code = json_to_faust(config)
        assert "fb(x0, x1, x2, x3) =" in code


#macro control tests

def _recursion_config(controls: dict | None = None) -> dict:
    """2-channel fdn loop: parallel delays fed back through a matrix."""
    node = {
        "type": "Recursion",
        "name": "loop",
        "fF": _leaf("delays", "parallelDelay", {"samples": [100, 200]}, n_ch=2),
        "fB": _leaf("fb", "Gain", {"matrix": [[0.0, 0.7], [-0.7, 0.0]]}, n_ch=2),
    }
    config = _wrap_config(node, name="macro_test")
    if controls is not None:
        config["controls"] = controls
    return config


class TestMacroControls:
    def test_no_controls_output_unchanged(self):
        code = json_to_faust(_recursion_config())
        assert "ctl_" not in code
        assert "//macro controls" not in code

    def test_rt60_emits_per_line_jot_gains(self):
        code = json_to_faust(_recursion_config({"rt60": True}))
        #slider calibrated in seconds, smoothed
        assert 'ctl_rt60 = hslider("rt60 [unit:s]", 2, 0.1, 10, 0.01) : si.smoo;' in code
        #attenuation uses the nominal delay (100), the line itself the
        #compensated delay (99), so decay rate is homogeneous per line
        assert "@(99)*(ba.db2linear(-60.0*100/(ma.SR*ctl_rt60)))" in code
        assert "@(199)*(ba.db2linear(-60.0*200/(ma.SR*ctl_rt60)))" in code

    def test_rt60_without_recursion_is_omitted(self):
        #a bare delay outside any feedback loop has no loop time, the
        #control must be dropped with a warning, not silently attached
        node = _leaf("d", "parallelDelay", {"samples": [100]}, n_ch=1)
        config = _wrap_config(node)
        config["controls"] = {"rt60": True}
        code = json_to_faust(config)
        assert "warning: rt60" in code
        assert "hslider" not in code
        assert "ba.db2linear" not in code

    def test_dry_wet_wraps_process(self):
        code = json_to_faust(_recursion_config({"dry_wet": True}))
        assert 'ctl_drywet = hslider("dry/wet", 1, 0, 1, 0.01) : si.smoo;' in code
        #equal-sum crossfade: wet scaled by the slider, dry by its complement
        assert "si.bus(2) <:" in code
        assert "par(i, 2, *(ctl_drywet))" in code
        assert "par(i, 2, *(1.0 - ctl_drywet))" in code
        assert ":> si.bus(2)" in code

    def test_dry_wet_channel_mismatch_warns(self):
        #4 inputs mixed to 1 output: no dry path mapping exists
        node = _leaf("out", "Gain", {"matrix": [[0.25, 0.25, 0.25, 0.25]]})
        node["input_channels"] = 4
        node["output_channels"] = 1
        config = _wrap_config(node)
        config["controls"] = {"dry_wet": True}
        code = json_to_faust(config)
        assert "warning: dry_wet" in code
        assert "ctl_drywet =" not in code

    def test_pre_delay_on_each_input(self):
        code = json_to_faust(_recursion_config({"pre_delay": True}))
        assert 'ctl_predelay = hslider("pre-delay [unit:ms]", 0, 0, 250, 1);' in code
        assert "par(i, 2, de.delay(65536, int(ctl_predelay*0.001*ma.SR)))" in code

    def test_combined_controls_pre_delay_inside_wet_path(self):
        code = json_to_faust(
            _recursion_config({"rt60": True, "dry_wet": True, "pre_delay": True})
        )
        #all three sliders present
        assert "ctl_rt60 =" in code
        assert "ctl_drywet =" in code
        assert "ctl_predelay =" in code
        #pre-delay wraps the core first, so it sits inside the wet
        #branch of the dry/wet split: the dry signal is not pre-delayed
        wet_start = code.index("<:")
        predelay_pos = code.index("de.delay(65536")
        dry_pos = code.index("par(i, 2, *(1.0 - ctl_drywet))")
        assert wet_start < predelay_pos < dry_pos

    def test_custom_ranges(self):
        code = json_to_faust(
            _recursion_config({"rt60": {"init": 3.5, "max": 20.0}})
        )
        assert 'hslider("rt60 [unit:s]", 3.5, 0.1, 20, 0.01)' in code

    def test_call_time_controls_override_config(self):
        config = _recursion_config({"rt60": {"init": 2.0}})
        code = json_to_faust(config, controls={"rt60": {"init": 5.0}})
        assert 'hslider("rt60 [unit:s]", 5, 0.1, 10, 0.01)' in code

    def test_unknown_control_raises(self):
        with pytest.raises(ValueError, match="unknown macro control"):
            json_to_faust(_recursion_config({"reverb_amount": True}))

    def test_disabled_control_is_skipped(self):
        code = json_to_faust(_recursion_config({"rt60": False}))
        assert "ctl_rt60" not in code


#multichannel io tests

def _stereo_fdn_config(controls: dict | None = None) -> dict:
    """stereo-in stereo-out fdn: B (4x2) -> 4-line loop -> C (2x4)."""
    config = {
        "type": "Series",
        "name": "stereo",
        "children": [
            _leaf("b_in", "Gain",
                  {"matrix": [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]}),
            {
                "type": "Recursion",
                "name": "loop",
                "fF": _leaf("delays", "parallelDelay",
                            {"samples": [100, 200, 300, 400]}, n_ch=4),
                "fB": _leaf("fb", "Gain", {"matrix": [
                    [0.5, 0.5, 0.5, 0.5],
                    [0.5, -0.5, 0.5, -0.5],
                    [0.5, 0.5, -0.5, -0.5],
                    [0.5, -0.5, -0.5, 0.5],
                ]}),
            },
            _leaf("c_out", "Gain",
                  {"matrix": [[0.25, 0.25, 0.0, 0.0], [0.0, 0.0, 0.25, 0.25]]}),
        ],
    }
    config["children"][0]["input_channels"] = 2
    config["children"][0]["output_channels"] = 4
    config["children"][2]["input_channels"] = 4
    config["children"][2]["output_channels"] = 2
    config = _wrap_config(config, name="stereo")
    if controls is not None:
        config["controls"] = controls
    return config


class TestMultichannelIO:
    """B in R^{N x k_in} and C in R^{k_out x N} must emit through the
    generic matrix path with the correct arities: multichannel io is
    a property of the matrices, not special-cased plumbing."""

    def test_stereo_input_matrix_has_two_args(self):
        code = json_to_faust(_stereo_fdn_config())
        #B is 4x2: function of two inputs producing four rows
        assert "b_in(x0, x1) =" in code

    def test_stereo_output_matrix_mixes_four_lines(self):
        code = json_to_faust(_stereo_fdn_config())
        #C is 2x4: function of four inputs producing two rows
        assert "c_out(x0, x1, x2, x3) =" in code

    def test_loop_adders_follow_feedback_not_io(self):
        #the recursion interleaves 4 feedback channels regardless of
        #the external io width
        code = json_to_faust(_stereo_fdn_config())
        assert "ro.interleave(4, 2) : par(i, 4, +)" in code

    def test_dry_wet_uses_stereo_bus(self):
        code = json_to_faust(_stereo_fdn_config({"dry_wet": True}))
        assert "si.bus(2) <:" in code
        assert "par(i, 2, *(ctl_drywet))" in code

    def test_pre_delay_on_both_inputs(self):
        code = json_to_faust(_stereo_fdn_config({"pre_delay": True}))
        assert "par(i, 2, de.delay(65536, int(ctl_predelay*0.001*ma.SR)))" in code

    def test_asymmetric_io_emits_correct_arities(self):
        #mono in, stereo out: dry/wet must refuse, matrices still correct
        config = _stereo_fdn_config({"dry_wet": True})
        config["children"][0]["params"]["matrix"] = [[1.0], [1.0], [1.0], [1.0]]
        config["children"][0]["input_channels"] = 1
        code = json_to_faust(config)
        assert "b_in(x0) =" in code
        assert "c_out(x0, x1, x2, x3) =" in code
        assert "warning: dry_wet" in code


#sos filter tests

class TestSOSFilterCodegen:
    def test_single_section_single_channel(self):
        #one section, one channel, 5 coefficients (already normalised)
        sos = [[[1.0, 0.0, 0.0, -0.9, 0.0]]]
        node = _leaf("f", "parallelSOSFilter", {"sos": sos}, n_ch=1)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "fi.tf2(" in code
        assert "-0.9" in code

    def test_multiple_sections_cascaded(self):
        #two sections should be composed in series with :
        sos = [
            [[0.998, 0.0, 0.0, -0.002, 0.0]],
            [[0.5, 0.1, 0.0, -0.3, 0.05]],
        ]
        node = _leaf("f", "parallelSOSFilter", {"sos": sos}, n_ch=1)
        config = _wrap_config(node)
        code = json_to_faust(config)
        #two fi.tf2 calls joined by :
        assert code.count("fi.tf2(") == 2

    def test_multiple_channels_parallel(self):
        #one section, two channels should be composed with ,
        sos = [[[0.9, 0.0, 0.0, -0.1, 0.0], [0.8, 0.0, 0.0, -0.2, 0.0]]]
        node = _leaf("f", "parallelSOSFilter", {"sos": sos}, n_ch=2)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert code.count("fi.tf2(") == 2
        assert "," in code


#composition tests

class TestComposition:
    def test_series(self):
        delay = _leaf("d", "parallelDelay", {"samples": [500]}, n_ch=1)
        gain = _leaf("g", "parallelGain", {"gains": [0.5]}, n_ch=1)
        series = {
            "type": "Series",
            "name": "chain",
            "children": [delay, gain],
        }
        config = _wrap_config(series)
        code = json_to_faust(config)
        #series uses : operator
        assert ":" in code
        assert "@(500)" in code
        assert "*(0.5)" in code

    def test_parallel_no_sum(self):
        #flamo Parallel is a y-split: both branches see the same input
        g1 = _leaf("a", "parallelGain", {"gains": [0.5]}, n_ch=1)
        g2 = _leaf("b", "parallelGain", {"gains": [0.3]}, n_ch=1)
        par = {
            "type": "Parallel",
            "name": "split",
            "children": [g1, g2],
            "sum_output": False,
        }
        config = _wrap_config(par)
        code = json_to_faust(config)
        assert "_ <: (*(0.5) , *(0.3))" in code
        #outputs concatenate, no merge
        assert ":>" not in code

    def test_parallel_with_sum(self):
        g1 = _leaf("a", "parallelGain", {"gains": [0.5]}, n_ch=1)
        g2 = _leaf("b", "parallelGain", {"gains": [0.3]}, n_ch=1)
        par = {
            "type": "Parallel",
            "name": "merge",
            "children": [g1, g2],
            "sum_output": True,
        }
        config = _wrap_config(par)
        code = json_to_faust(config)
        #shared input split, branch outputs summed: y = a(x) + b(x)
        assert "_ <: (*(0.5) , *(0.3)) :> _" in code

    def test_recursion_output_compensation(self):
        #delays inside the loop are shortened by one for the ~ sample;
        #a one-sample delay on the recursion output restores absolute
        #arrival times so wet and direct paths stay aligned
        code = json_to_faust(_recursion_config())
        assert "(@(99) , @(199))" in code  #loop delays shortened
        assert ": par(i, 2, @(1)))" in code  #output re-delayed outside loop

    def test_parallel_direct_path_structure(self):
        #the standard dss fdn: Parallel(fdn_branch, D) with summing.
        #both the wet branch and the direct gain see the one input
        fdn_branch = {
            "type": "Series",
            "name": "wet",
            "children": [
                _leaf("b_in", "Gain", {"matrix": [[1.0], [1.0]]}, n_ch=2),
                _leaf("c_out", "Gain", {"matrix": [[0.5, 0.5]]}, n_ch=2),
            ],
        }
        fdn_branch["children"][0]["input_channels"] = 1
        fdn_branch["children"][1]["output_channels"] = 1
        direct = _leaf("D", "Gain", {"matrix": [[0.2]]}, n_ch=1)
        par = {
            "type": "Parallel",
            "name": "core",
            "children": [fdn_branch, direct],
            "sum_output": True,
        }
        code = json_to_faust(_wrap_config(par))
        #one shared input split across wet and direct, summed to one out
        assert "_ <: " in code
        assert ":> _" in code

    def test_parallel_multichannel_sum_keeps_width(self):
        #two 2-channel branches summed: output stays 2 wide, not 1
        g1 = _leaf("a", "parallelGain", {"gains": [0.5, 0.4]}, n_ch=2)
        g2 = _leaf("b", "parallelGain", {"gains": [0.3, 0.2]}, n_ch=2)
        par = {
            "type": "Parallel",
            "name": "merge2",
            "children": [g1, g2],
            "sum_output": True,
        }
        code = json_to_faust(_wrap_config(par))
        assert "si.bus(2) <: " in code
        assert ":> si.bus(2)" in code

    def test_recursion(self):
        delay = _leaf("d", "parallelDelay", {"samples": [1000]}, n_ch=1)
        gain = _leaf("g", "parallelGain", {"gains": [0.5]}, n_ch=1)
        rec = {
            "type": "Recursion",
            "name": "loop",
            "fF": delay,
            "fB": gain,
        }
        config = _wrap_config(rec)
        code = json_to_faust(config)
        #recursion uses ~ operator
        assert "~" in code
        #delays inside recursion are decremented by 1 to compensate for
        #the implicit one-sample delay from the ~ operator
        assert "@(999)" in code
        assert "*(0.5)" in code

    def test_recursion_multichannel_interleave(self):
        #regression test for the N>1 routing bug.
        #without ro.interleave, par(i,N,+) receives feedback signals
        #in the wrong slots: the ~ operator delivers feedback to the
        #first N inputs contiguously, but par(i,N,+) expects pairs
        #(fb0,ext0, fb1,ext1, ...). interleaving is required so each
        #adder sums one feedback signal with one external signal.
        delay = _leaf("d", "parallelDelay",
                       {"samples": [1103, 1447, 1811, 2137]})
        matrix = [
            [0.5,  0.5,  0.5,  0.5],
            [0.5, -0.5,  0.5, -0.5],
            [0.5,  0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5,  0.5],
        ]
        fb = _leaf("fB", "Gain", {"matrix": matrix})
        rec = {
            "type": "Recursion",
            "name": "loop",
            "fF": delay,
            "fB": fb,
        }
        config = _wrap_config(rec)
        code = json_to_faust(config)
        #must interleave before the adders for correct routing
        assert "ro.interleave(4, 2)" in code
        assert "par(i, 4, +)" in code
        #interleave must come before the adders in the chain
        assert "ro.interleave(4, 2) : par(i, 4, +)" in code

    def test_recursion_single_channel_no_interleave(self):
        #N=1 uses a bare + (two inputs: one feedback, one external).
        #no interleave is needed because the routing is already correct.
        delay = _leaf("d", "parallelDelay", {"samples": [500]}, n_ch=1)
        gain = _leaf("g", "parallelGain", {"gains": [0.7]}, n_ch=1)
        rec = {
            "type": "Recursion",
            "name": "loop",
            "fF": delay,
            "fB": gain,
        }
        config = _wrap_config(rec)
        code = json_to_faust(config)
        assert "ro.interleave" not in code
        assert "+ :" in code

    def test_shell_unwraps(self):
        gain = _leaf("g", "parallelGain", {"gains": [0.8]}, n_ch=1)
        shell = {
            "type": "Shell",
            "name": "wrapper",
            "children": [gain],
        }
        config = _wrap_config(shell)
        code = json_to_faust(config)
        #shell should not appear in the output, just its core
        assert "Shell" not in code.split("//")[-1]
        assert "*(0.8)" in code


#full fdn integration test

class TestFullFDN:
    """test code generation for a complete 4-channel fdn structure.

    this mirrors the mock_fdn_model from test_flamo_to_json.py but
    works from the json config dict directly.
    """

    @pytest.fixture
    def fdn_config(self) -> dict:
        """a 4-channel fdn config matching FLAMO_RT_SPEC.md section 6."""
        return {
            "type": "Shell",
            "name": "TestFDN",
            "fs": 48000,
            "children": [{
                "type": "Parallel",
                "name": "core",
                "sum_output": True,
                "children": [
                    {
                        "type": "Series",
                        "name": "brA",
                        "children": [
                            _leaf("input_gain", "Gain",
                                  {"gains": [1.0, 1.0, 1.0, 1.0]}),
                            {
                                "type": "Recursion",
                                "name": "feedback_loop",
                                "fF": {
                                    "type": "Series",
                                    "name": "fF",
                                    "children": [
                                        _leaf("delay", "parallelDelay",
                                              {"samples": [1103, 1447, 1811, 2137]}),
                                        _leaf("filter", "parallelSOSFilter", {
                                            "sos": [
                                                [
                                                    [0.998, 0.0, 0.0, -0.002, 0.0],
                                                    [0.998, 0.0, 0.0, -0.002, 0.0],
                                                    [0.998, 0.0, 0.0, -0.002, 0.0],
                                                    [0.998, 0.0, 0.0, -0.002, 0.0],
                                                ]
                                            ]
                                        }),
                                    ],
                                },
                                "fB": _leaf("feedback_matrix", "Gain", {
                                    "matrix": [
                                        [0.5, 0.5, 0.5, 0.5],
                                        [0.5, -0.5, 0.5, -0.5],
                                        [0.5, 0.5, -0.5, -0.5],
                                        [0.5, -0.5, -0.5, 0.5],
                                    ]
                                }),
                            },
                            _leaf("output_gain", "Gain",
                                  {"gains": [0.25, 0.25, 0.25, 0.25]}),
                        ],
                    },
                    _leaf("direct", "Gain", {"gains": [0.0]}, n_ch=1),
                ],
            }],
        }

    def test_generates_valid_structure(self, fdn_config):
        code = json_to_faust(fdn_config)
        #must have import, process, and the key dsp elements
        assert 'import("stdfaust.lib");' in code
        assert "process = " in code
        assert "~" in code  #recursion
        assert ":>" in code  #parallel sum

    def test_all_delays_present(self, fdn_config):
        code = json_to_faust(fdn_config)
        #delays are decremented by 1 inside recursion to compensate for
        #the implicit one-sample delay from the ~ operator
        for d in [1102, 1446, 1810, 2136]:
            assert f"@({d})" in code

    def test_feedback_matrix_hoisted(self, fdn_config):
        code = json_to_faust(fdn_config)
        #the feedback matrix should appear as a top-level definition
        assert "feedback_matrix(x0, x1, x2, x3) =" in code

    def test_filters_present(self, fdn_config):
        code = json_to_faust(fdn_config)
        assert "fi.tf2(" in code
        #4 channels, 1 section each = 4 fi.tf2 calls
        assert code.count("fi.tf2(") == 4

    def test_no_trailing_whitespace(self, fdn_config):
        code = json_to_faust(fdn_config)
        for line in code.split("\n"):
            assert line == line.rstrip(), f"trailing whitespace: {line!r}"


#edge cases

class TestEdgeCases:
    def test_unknown_module_emits_wire(self):
        node = _leaf("mystery", "FancyProcessor", {"raw": [1, 2, 3]})
        config = _wrap_config(node)
        code = json_to_faust(config)
        #unknown modules become passthrough wires
        assert "process = _;" in code

    def test_gain_with_no_params(self):
        node = _leaf("g", "Gain", {})
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "process = _;" in code

    def test_recursion_with_missing_fb(self):
        delay = _leaf("d", "parallelDelay", {"samples": [100]}, n_ch=1)
        rec = {
            "type": "Recursion",
            "name": "loop",
            "fF": delay,
            "fB": None,
        }
        config = _wrap_config(rec)
        code = json_to_faust(config)
        #should still produce valid code with _ as feedback
        assert "~" in code
        assert "_" in code

    def test_deeply_nested_series(self):
        leaf = _leaf("g", "parallelGain", {"gains": [0.5, 0.5]}, n_ch=2)
        inner = {"type": "Series", "name": "inner", "children": [leaf]}
        outer = {"type": "Series", "name": "outer", "children": [inner]}
        config = _wrap_config(outer)
        code = json_to_faust(config)
        assert "*(0.5)" in code

    def test_integer_gain_no_decimal(self):
        #integer values should not have decimal points in faust output
        node = _leaf("g", "parallelGain", {"gains": [1.0]}, n_ch=1)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "*(1)" in code

    def test_negative_coefficient(self):
        node = _leaf("g", "parallelGain", {"gains": [-0.707]}, n_ch=1)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "*(-0.707)" in code

    def test_float_representation_noise(self):
        #0.3 in ieee754 is 0.30000000000000004. _fmt must clean this up.
        node = _leaf("g", "parallelGain", {"gains": [0.30000000000000004]}, n_ch=1)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "*(0.3)" in code
        assert "0.30000000000000004" not in code


#fractional and variable delay tests

class TestFractionalDelayCodegen:
    def test_single_fractional_delay(self):
        node = _leaf("d", "fractionalDelay",
                      {"samples": [1000.5]}, n_ch=1)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "de.fdelay(" in code
        assert "1000.5" in code

    def test_multiple_fractional_delays(self):
        node = _leaf("d", "fractionalDelay",
                      {"samples": [500.3, 750.7, 1000.1]}, n_ch=3)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert code.count("de.fdelay(") == 3
        assert "500.3" in code
        assert "750.7" in code
        assert "1000.1" in code

    def test_fractional_delay_buffer_size(self):
        #buffer size should be next power of two above the max delay
        node = _leaf("d", "fractionalDelay",
                      {"samples": [1000.5]}, n_ch=1)
        config = _wrap_config(node)
        code = json_to_faust(config)
        #max delay 1000.5, int + 2 = 1002, next power of two = 1024
        assert "de.fdelay(1024," in code

    def test_fractional_delay_in_recursion(self):
        #fractional delays inside recursion are decremented by 1.0
        delay = _leaf("d", "fractionalDelay",
                       {"samples": [500.5]}, n_ch=1)
        gain = _leaf("g", "parallelGain", {"gains": [0.5]}, n_ch=1)
        rec = {
            "type": "Recursion",
            "name": "loop",
            "fF": delay,
            "fB": gain,
        }
        config = _wrap_config(rec)
        code = json_to_faust(config)
        #500.5 - 1.0 = 499.5
        assert "499.5" in code

    def test_fractional_delay_samples_fractional_key(self):
        #the emitter also accepts "samples_fractional" as the param key
        node = _leaf("d", "parallelDelay",
                      {"samples_fractional": [200.25, 300.75]}, n_ch=2)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "de.fdelay(" in code
        assert "200.25" in code
        assert "300.75" in code

    def test_parallel_delay_isint_false_uses_fractional(self):
        #when flamo meta says isint=False, emit de.fdelay even when
        #both samples and samples_fractional are present
        node = _leaf("d", "parallelDelay",
                      {"samples": [200, 301],
                       "samples_fractional": [200.25, 300.75]}, n_ch=2)
        node["flamo"] = {"isint": False, "max_len": 1000, "unit": 1}
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "de.fdelay(" in code
        #the fractional values are what appear in the output
        assert "200.25" in code
        assert "300.75" in code

    def test_parallel_delay_isint_true_keeps_integer(self):
        #even when samples_fractional is present, isint=True keeps @(n)
        node = _leaf("d", "parallelDelay",
                      {"samples": [200, 301],
                       "samples_fractional": [200.25, 300.75]}, n_ch=2)
        node["flamo"] = {"isint": True, "max_len": 1000, "unit": 1}
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "de.fdelay(" not in code
        assert "@(200)" in code
        assert "@(301)" in code

    def test_parallel_delay_default_isint_true(self):
        #absent isint metadata, default is integer (back-compat)
        node = _leaf("d", "parallelDelay",
                      {"samples": [200, 301],
                       "samples_fractional": [200.25, 300.75]}, n_ch=2)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "de.fdelay(" not in code
        assert "@(200)" in code

    def test_variable_delay(self):
        node = _leaf("d", "variableDelay",
                      {"samples": [1000, 2000]}, n_ch=2)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert code.count("de.delay(") == 2
        #buffer size: max 2000, next power of two = 2048
        assert "de.delay(2048," in code

    def test_variable_delay_in_recursion(self):
        delay = _leaf("d", "variableDelay",
                       {"samples": [1000]}, n_ch=1)
        gain = _leaf("g", "parallelGain", {"gains": [0.7]}, n_ch=1)
        rec = {
            "type": "Recursion",
            "name": "loop",
            "fF": delay,
            "fB": gain,
        }
        config = _wrap_config(rec)
        code = json_to_faust(config)
        #1000 - 1 = 999
        assert "de.delay(1024, 999)" in code


#topology comment tests

class TestTopologyComments:
    def test_fdn_has_topology_comments(self):
        delay = _leaf("d", "parallelDelay",
                       {"samples": [1103, 1447, 1811, 2137]})
        matrix = [[0.5, 0.5, 0.5, 0.5],
                  [0.5, -0.5, 0.5, -0.5],
                  [0.5, 0.5, -0.5, -0.5],
                  [0.5, -0.5, -0.5, 0.5]]
        fb = _leaf("fB", "Gain", {"matrix": matrix})
        rec = {"type": "Recursion", "name": "loop", "fF": delay, "fB": fb}
        config = _wrap_config(rec)
        code = json_to_faust(config)
        assert "//fdn topology:" in code
        assert "//4 delay lines:" in code
        assert "23.0" in code  #1103/48000*1000 = 22.98 ~ 23.0
        assert "//feedback matrix: 4x4" in code

    def test_fdn_with_absorption_comments(self):
        delay = _leaf("d", "parallelDelay", {"samples": [500]}, n_ch=1)
        sos_node = _leaf("f", "parallelSOSFilter", {
            "sos": [[[0.998, 0.0, 0.0, -0.002, 0.0]]]
        }, n_ch=1)
        fF = {"type": "Series", "name": "fF", "children": [delay, sos_node]}
        fb = _leaf("g", "parallelGain", {"gains": [0.5]}, n_ch=1)
        rec = {"type": "Recursion", "name": "loop", "fF": fF, "fB": fb}
        config = _wrap_config(rec)
        code = json_to_faust(config)
        assert "//absorption: 1 biquad section per channel" in code

    def test_no_topology_without_recursion(self):
        #simple gain has no fdn topology to describe
        node = _leaf("g", "parallelGain", {"gains": [0.5]}, n_ch=1)
        config = _wrap_config(node)
        code = json_to_faust(config)
        assert "//fdn topology:" not in code

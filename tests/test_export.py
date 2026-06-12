#test_export
#author: Facundo Franchino
"""
tests for the juce export helper.

faust2juce is faked via monkeypatching so the tests run without the
faust toolchain; the real tool is exercised by the verification
scripts. what is under test here is the orchestration: dsp emission,
certificate gating, argument construction, and the returned paths.
"""

from __future__ import annotations

import json

import pytest

import rt_fdn.export as export_module
from rt_fdn.export import export_juce


HADAMARD4 = [
    [0.5, 0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5, 0.5],
]


def _loop_config(scale=1.0):
    matrix = [[scale * v for v in row] for row in HADAMARD4]
    return {
        "type": "Recursion",
        "name": "loop",
        "fF": {
            "type": "Leaf", "name": "delays", "module_type": "parallelDelay",
            "params": {"samples": [100, 200, 300, 400]},
            "input_channels": 4, "output_channels": 4,
        },
        "fB": {
            "type": "Leaf", "name": "fb", "module_type": "Gain",
            "params": {"matrix": matrix},
            "input_channels": 4, "output_channels": 4,
        },
        "fs": 48000,
        "name": "Test",
    }


@pytest.fixture
def fake_faust2juce(monkeypatch):
    """substitute the external tools with a recorder mimicking their
    on-disk layout: faust2juce creates the project dir, projucer
    creates the xcode project, xcodebuild creates the artifacts."""
    calls = []

    def fake_which(tool):
        return f"/usr/local/bin/{tool}"

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(args, cwd=None, capture_output=None, text=None, env=None):
        calls.append({"args": args, "cwd": cwd, "env": env})
        from pathlib import Path
        prog = Path(args[0]).name
        if prog == "xcode-select":
            return _Result()
        if prog == "Projucer":
            jucer = Path(args[-1])
            name = jucer.stem
            (jucer.parent / "Builds" / "MacOSX" / f"{name}.xcodeproj"
             ).mkdir(parents=True, exist_ok=True)
            return _Result()
        if prog == "xcodebuild":
            proj = Path(args[args.index("-project") + 1])
            name = proj.stem
            release = proj.parent / "build" / "Release"
            (release / f"{name}.vst3").mkdir(parents=True, exist_ok=True)
            (release / f"{name}.component").mkdir(parents=True, exist_ok=True)
            return _Result()
        #faust2juce creates <basename>/ next to the dsp, containing
        #a jucer project with the hard-coded GRAME manufacturer
        dsp_name = args[-1]
        stem = Path(dsp_name).stem
        proj = Path(cwd) / stem
        proj.mkdir(parents=True, exist_ok=True)
        (proj / f"{stem}.jucer").write_text(
            f'<JUCERPROJECT name="{stem}" pluginManufacturer="GRAME" '
            f'pluginManufacturerCode="Manu" pluginCode="0000"/>\n'
        )
        return _Result()

    monkeypatch.setattr(export_module.shutil, "which", fake_which)
    monkeypatch.setattr(export_module.subprocess, "run", fake_run)
    return calls


class TestExportJuce:
    def test_exports_dsp_project_and_certificate(self, tmp_path, fake_faust2juce):
        #attenuated loop: certified-stable, passes strict
        result = export_juce(_loop_config(0.9), tmp_path, name="MyReverb")

        assert result["dsp"] == tmp_path / "MyReverb.dsp"
        assert "process = " in result["dsp"].read_text()
        assert result["project"] == tmp_path / "MyReverb"
        assert result["project"].is_dir()
        assert result["verdict"] == "certified-stable"
        cert = json.loads(result["certificate"].read_text())
        assert cert["verdict"] == "certified-stable"

    def test_strict_blocks_not_certified(self, tmp_path, fake_faust2juce):
        with pytest.raises(ValueError, match="not-certified"):
            export_juce(_loop_config(1.1), tmp_path, name="Hot")
        #nothing was handed to faust2juce
        assert fake_faust2juce == []

    def test_strict_false_exports_anyway(self, tmp_path, fake_faust2juce):
        result = export_juce(_loop_config(1.1), tmp_path, name="Hot", strict=False)
        assert result["verdict"] == "not-certified"
        assert result["project"].is_dir()

    def test_marginal_lossless_passes_strict(self, tmp_path, fake_faust2juce):
        result = export_juce(_loop_config(1.0), tmp_path, name="Lossless")
        assert result["verdict"] == "marginally-stable"

    def test_controls_reach_the_dsp(self, tmp_path, fake_faust2juce):
        result = export_juce(
            _loop_config(1.0), tmp_path, name="Knobs",
            controls={"rt60": True, "dry_wet": True},
        )
        code = result["dsp"].read_text()
        assert "ctl_rt60" in code
        assert "ctl_drywet" in code
        #with the rt60 control the lossless prototype is certified
        assert result["verdict"] == "certified-stable"

    def test_faust_code_string_skips_certificate(self, tmp_path, fake_faust2juce):
        code = 'import("stdfaust.lib");\nprocess = _;\n'
        result = export_juce(code, tmp_path, name="Raw")
        assert result["certificate"] is None
        assert result["verdict"] is None
        assert result["dsp"].read_text() == code

    def test_standalone_and_extra_args(self, tmp_path, fake_faust2juce):
        export_juce(
            _loop_config(0.9), tmp_path, name="App",
            standalone=True, extra_args=("-midi",),
        )
        args = fake_faust2juce[0]["args"]
        assert "-standalone" in args
        assert "-midi" in args
        assert args[-1] == "App.dsp"

    def test_missing_tool_raises_with_hint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(export_module.shutil, "which", lambda t: None)
        with pytest.raises(RuntimeError, match="not found on PATH"):
            export_juce(_loop_config(0.9), tmp_path, name="X")

    def test_name_is_sanitised(self, tmp_path, fake_faust2juce):
        result = export_juce(_loop_config(0.9), tmp_path, name="My Reverb 2!")
        assert result["dsp"].name == "My_Reverb_2_.dsp"

    def test_manufacturer_defaults_to_rt_fdn(self, tmp_path, fake_faust2juce):
        result = export_juce(_loop_config(0.9), tmp_path, name="Verb")
        jucer = (result["project"] / "Verb.jucer").read_text()
        assert 'pluginManufacturer="rt-fdn"' in jucer
        assert "GRAME" not in jucer
        #the four-char code is identity: untouched unless requested
        assert 'pluginManufacturerCode="Manu"' in jucer

    def test_manufacturer_and_code_override(self, tmp_path, fake_faust2juce):
        result = export_juce(
            _loop_config(0.9), tmp_path, name="Verb",
            manufacturer="GauchoDSP", manufacturer_code="Gdsp",
        )
        jucer = (result["project"] / "Verb.jucer").read_text()
        assert 'pluginManufacturer="GauchoDSP"' in jucer
        assert 'pluginManufacturerCode="Gdsp"' in jucer

    def test_juce_modules_flag_passed_through(self, tmp_path, fake_faust2juce):
        export_juce(
            _loop_config(0.9), tmp_path, name="Mod",
            juce_modules="~/JUCE/modules",
        )
        args = fake_faust2juce[0]["args"]
        i = args.index("-jucemodulesdir")
        assert args[i + 1].endswith("JUCE/modules")
        assert "~" not in args[i + 1]  #expanded

    @pytest.mark.skipif(__import__("sys").platform != "darwin",
                        reason="build path is macos-only")
    def test_build_chains_projucer_and_xcodebuild(self, tmp_path, fake_faust2juce):
        result = export_juce(
            _loop_config(0.9), tmp_path, name="Built", build=True,
        )
        progs = [c["args"][0].split("/")[-1] for c in fake_faust2juce]
        #faust2juce, then projucer resave, then xcodebuild
        assert "Projucer" in progs
        assert "xcodebuild" in progs
        assert progs.index("Projucer") < progs.index("xcodebuild")
        xc = next(c for c in fake_faust2juce if c["args"][0] == "xcodebuild")
        assert "Built - VST3" in xc["args"]
        assert "Built - AU" in xc["args"]
        assert result["vst3"] is not None and result["vst3"].exists()
        assert result["au"] is not None and result["au"].exists()
        assert "installed" in result

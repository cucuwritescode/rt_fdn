#export
#author: Facundo Franchino
"""
export generated faust code as a juce plugin project (vst3/au)

the .dsp is the product; plugin formats are downstream targets. this
module wraps the faust2juce tool (shipped with the faust distribution)
which generates a juce project building to vst3, au and standalone
with llvm-compiled dsp, no interpreter overhead in the shipped
artifact. macro-control sliders become automatable daw parameters in
every format without per-target work.

the exporter is certificate-aware. by default it refuses to ship a
model whose stability certificate says "unstable" (internally unstable
filter sections) or "not-certified" (loop gain bound above one). a
marginally stable lossless prototype passes, as does an indeterminate
one (absence of proof is not proof of absence); the certificate is
written next to the .dsp either way so the verdict travels with the
artifact.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from rt_fdn.codegen.json_to_faust import json_to_faust, _safe_name
from rt_fdn.certificate import certify, write_certificate

#verdicts that block a strict export
_BLOCKING_VERDICTS = ("unstable", "not-certified")

#default projucer location on macos when not on PATH
_PROJUCER_APP = "/Applications/Projucer.app/Contents/MacOS/Projucer"


def _find_projucer(explicit: str | None) -> str:
    """locate the projucer binary."""
    if explicit:
        return explicit
    found = shutil.which("Projucer")
    if found:
        return found
    if Path(_PROJUCER_APP).exists():
        return _PROJUCER_APP
    raise RuntimeError(
        "Projucer not found. install JUCE (https://juce.com) or pass "
        "projucer=<path to Projucer binary>."
    )


def _patch_jucer(
    jucer: Path,
    manufacturer: str,
    manufacturer_code: str | None,
) -> None:
    """set the plugin manufacturer in the generated jucer project.

    faust2juce hard-codes pluginManufacturer="GRAME"; daws group
    plugins under this name, so it should identify the actual vendor.
    the four-character manufacturer code is only touched when given
    explicitly: changing it changes the plugin's registered identity
    and daws treat the result as a brand-new plugin.
    """
    if not jucer.exists():
        return
    text = jucer.read_text()
    text = re.sub(
        r'pluginManufacturer="[^"]*"',
        f'pluginManufacturer="{manufacturer}"',
        text,
    )
    if manufacturer_code:
        text = re.sub(
            r'pluginManufacturerCode="[^"]*"',
            f'pluginManufacturerCode="{manufacturer_code}"',
            text,
        )
    jucer.write_text(text)


def _xcode_env(developer_dir: str | None) -> dict[str, str]:
    """build an environment where xcodebuild can find full xcode.

    a machine whose xcode-select points at the command line tools
    cannot run xcodebuild; if a full xcode lives in /Applications the
    DEVELOPER_DIR override selects it without needing sudo.
    """
    env = dict(os.environ)
    if developer_dir:
        env["DEVELOPER_DIR"] = str(developer_dir)
        return env
    try:
        result = subprocess.run(
            ["xcode-select", "-p"], capture_output=True, text=True,
        )
        active = (result.stdout or "").strip()
    except OSError:
        active = ""
    if "Xcode.app" not in active and Path("/Applications/Xcode.app").exists():
        env["DEVELOPER_DIR"] = "/Applications/Xcode.app/Contents/Developer"
    return env


def export_juce(
    source: dict[str, Any] | str,
    out_dir: str | Path,
    *,
    name: str = "FDN",
    controls: dict[str, Any] | None = None,
    certificate: bool = True,
    strict: bool = True,
    standalone: bool = False,
    faust2juce: str = "faust2juce",
    juce_modules: str | Path | None = None,
    extra_args: tuple[str, ...] = (),
    manufacturer: str = "rt-fdn",
    manufacturer_code: str | None = None,
    build: bool = False,
    projucer: str | None = None,
    developer_dir: str | None = None,
) -> dict[str, Any]:
    """generate a juce plugin project from a config dict or faust code.

    parameters
    ----------
    source : dict | str
        a json config dict (output of flamo_to_json), or a complete
        faust source string. certification requires the config form:
        faust code cannot be analysed, only configs can.
    out_dir : str | Path
        directory receiving the .dsp, the certificate, and the juce
        project folder. created if missing.
    name : str
        plugin name; sanitised for the .dsp filename and project dir.
    controls : dict, optional
        macro controls forwarded to json_to_faust (config form only).
    certificate : bool
        compute the stability certificate and write it next to the
        .dsp (config form only).
    strict : bool
        refuse to export when the certificate verdict is "unstable"
        or "not-certified". marginal and indeterminate verdicts pass.
    standalone : bool
        generate a standalone application project instead of a plugin.
    faust2juce : str
        tool name or path, for non-standard installations.
    juce_modules : str | Path, optional
        path to the JUCE modules folder (e.g. ~/JUCE/modules). the
        generated project references it; required for building unless
        the project sits inside the juce tree.
    extra_args : tuple
        additional arguments passed through to faust2juce.
    manufacturer : str
        vendor name daws display and group plugins under. patched
        into the generated project (faust2juce hard-codes "GRAME").
    manufacturer_code : str, optional
        four-character manufacturer code. leave unset to keep the
        generated default: changing it gives the plugin a new
        identity and daws will list it as a separate plugin.
    build : bool
        also build the project to installed plugins (macos only:
        projucer resave + xcodebuild, vst3 and au targets, release).
        juce's plugin copy step installs the results into
        ~/Library/Audio/Plug-Ins automatically; rescan in the daw.
    projucer : str, optional
        path to the projucer binary when not on PATH or in
        /Applications.
    developer_dir : str, optional
        xcode developer directory for machines whose xcode-select
        points at the command line tools. auto-detected when omitted.

    returns
    -------
    result : dict
        {"dsp": Path, "project": Path, "certificate": Path | None,
         "verdict": str | None}, plus with build=True:
        {"vst3": Path | None, "au": Path | None, "installed": [Path]}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_name(name)
    dsp_path = out_dir / f"{safe}.dsp"

    cert_path: Path | None = None
    verdict: str | None = None

    if isinstance(source, dict):
        code = json_to_faust(source, controls=controls)
        if certificate:
            #same call-time controls as the codegen, so the verdict
            #describes the dsp actually being shipped
            cert = certify(source, controls=controls)
            verdict = cert["verdict"]
            if strict and verdict in _BLOCKING_VERDICTS:
                reasons = [
                    note
                    for loop in cert["loops"]
                    for note in loop.get("notes", [])
                ]
                if not cert["filters"]["all_sections_stable"]:
                    reasons.append(
                        "filter section poles reach modulus "
                        f"{cert['filters']['max_pole_modulus']:.6f}"
                    )
                raise ValueError(
                    f"refusing to export '{name}': certificate verdict "
                    f"is '{verdict}'. " + "; ".join(reasons) +
                    ". pass strict=False to export anyway."
                )
    else:
        code = source
        if certificate:
            #faust source cannot be certified, only configs can
            certificate = False

    dsp_path.write_text(code)
    if certificate and isinstance(source, dict):
        cert_path = write_certificate(source, dsp_path, controls=controls)

    tool = shutil.which(faust2juce)
    if tool is None:
        raise RuntimeError(
            f"'{faust2juce}' not found on PATH. it ships with the faust "
            "distribution (https://faust.grame.fr); install faust or "
            "pass faust2juce=<path>."
        )

    args = [tool]
    if standalone:
        args.append("-standalone")
    if juce_modules is not None:
        args.extend(["-jucemodulesdir", str(Path(juce_modules).expanduser())])
    args.extend(extra_args)
    args.append(dsp_path.name)

    #faust2juce creates <basename>/ next to the .dsp it is given
    result = subprocess.run(
        args, cwd=str(out_dir), capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"faust2juce failed (exit {result.returncode}):\n"
            f"{result.stderr.strip() or result.stdout.strip()}"
        )

    project_dir = out_dir / safe
    if not project_dir.is_dir():
        raise RuntimeError(
            f"faust2juce reported success but no project directory "
            f"appeared at {project_dir}"
        )

    #identify the vendor before any build, so manual builders get it too
    _patch_jucer(
        project_dir / f"{safe}.jucer", manufacturer, manufacturer_code,
    )

    out: dict[str, Any] = {
        "dsp": dsp_path,
        "project": project_dir,
        "certificate": cert_path,
        "verdict": verdict,
    }
    if not build:
        return out

    if sys.platform != "darwin":
        raise RuntimeError(
            "build=True is currently macos-only (projucer + xcodebuild). "
            "the generated project also contains visual studio and "
            "linux makefile exporters; build those with your platform "
            "toolchain."
        )

    #regenerate the native build files, then compile vst3 + au
    projucer_bin = _find_projucer(projucer)
    result = subprocess.run(
        [projucer_bin, "--resave", str(project_dir / f"{safe}.jucer")],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"projucer --resave failed:\n"
            f"{result.stderr.strip() or result.stdout.strip()}"
        )

    xcodeproj = project_dir / "Builds" / "MacOSX" / f"{safe}.xcodeproj"
    if not xcodeproj.exists():
        raise RuntimeError(
            f"projucer did not generate an xcode project at {xcodeproj}"
        )

    result = subprocess.run(
        [
            "xcodebuild", "-project", str(xcodeproj),
            "-configuration", "Release",
            "-target", f"{safe} - VST3",
            "-target", f"{safe} - AU",
            "build",
        ],
        capture_output=True, text=True, env=_xcode_env(developer_dir),
    )
    if result.returncode != 0:
        tail = (result.stderr.strip() or result.stdout.strip())[-2000:]
        raise RuntimeError(f"xcodebuild failed:\n{tail}")

    release = project_dir / "Builds" / "MacOSX" / "build" / "Release"
    vst3 = release / f"{safe}.vst3"
    au = release / f"{safe}.component"
    out["vst3"] = vst3 if vst3.exists() else None
    out["au"] = au if au.exists() else None

    #juce's plugin copy step installs into the user plugin folders
    plug_ins = Path.home() / "Library" / "Audio" / "Plug-Ins"
    out["installed"] = [
        p for p in (
            plug_ins / "VST3" / f"{safe}.vst3",
            plug_ins / "Components" / f"{safe}.component",
        ) if p.exists()
    ]
    return out

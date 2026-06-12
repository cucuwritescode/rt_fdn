#flamo_to_faust
#author: Facundo Franchino
"""
convenience function combining flamo_to_json and json_to_faust.

single call to go from a flamo model to valid faust dsp source code.
this is the primary public interface for most users.
"""

from __future__ import annotations

from typing import Any

from rt_fdn.codegen.flamo_to_json import flamo_to_json
from rt_fdn.codegen.json_to_faust import json_to_faust


def flamo_to_faust(
    model: Any,
    fs: float,
    *,
    name: str = "FDN",
    controls: dict[str, Any] | None = None,
) -> str:
    """convert a flamo model directly to faust dsp source code.

    combines flamo_to_json() and json_to_faust() in a single call.
    equivalent to json_to_faust(flamo_to_json(model, fs, name=name)).

    parameters
    ----------
    model : flamo model
        a Shell, Recursion, Series, Parallel, or leaf dsp module.
    fs : float
        sampling rate in hz. used to convert delay lengths from seconds
        to integer samples.
    name : str
        name for the dsp, appears in the generated faust header comment.
    controls : dict, optional
        macro controls to expose as sliders ("rt60", "dry_wet",
        "pre_delay"), see json_to_faust(). these are post-hoc
        performance controls layered onto the trained model, not the
        trained parameters themselves.

    returns
    -------
    faust_code : str
        complete faust dsp source code ready for compilation or interpretation.
    """
    config = flamo_to_json(model, fs, name=name)
    return json_to_faust(config, controls=controls)

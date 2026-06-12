#created by Facundo Franchino March 2026
"""rt-fdn: Real-time deployment of FLAMO audio graphs via FAUST."""

__version__ = "0.1.0"

from rt_fdn.codegen.flamo_to_json import flamo_to_json
from rt_fdn.codegen.json_to_faust import json_to_faust
from rt_fdn.codegen.flamo_to_faust import flamo_to_faust
from rt_fdn.hotreload import HotReload
from rt_fdn.certificate import certify, write_certificate
from rt_fdn.export import export_juce

__all__ = [
    "flamo_to_json", "json_to_faust", "json_to_flamo", "flamo_to_faust",
    "HotReload", "certify", "write_certificate", "export_juce",
]


def __getattr__(name):
    #json_to_flamo needs flamo and torch; everything else runs on
    #numpy alone. loading it lazily keeps `import rt_fdn` instant and
    #dependency-light while still exposing the full api at the root.
    if name == "json_to_flamo":
        from rt_fdn.codegen.json_to_flamo import json_to_flamo
        return json_to_flamo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

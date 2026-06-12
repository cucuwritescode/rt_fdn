#codegen, flamo model to json config export, faust code generation,
#and flamo model reconstruction from json config
#author: Facundo Franchino

from rt_fdn.codegen.flamo_to_json import flamo_to_json
from rt_fdn.codegen.json_to_faust import json_to_faust
from rt_fdn.codegen.flamo_to_faust import flamo_to_faust

__all__ = ["flamo_to_json", "json_to_faust", "flamo_to_faust", "json_to_flamo"]


def __getattr__(name):
    #json_to_flamo needs flamo and torch. loading it lazily keeps
    #`import rt_fdn` light and working on machines without them;
    #the heavy import happens only when reconstruction is requested.
    if name == "json_to_flamo":
        from rt_fdn.codegen.json_to_flamo import json_to_flamo
        return json_to_flamo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

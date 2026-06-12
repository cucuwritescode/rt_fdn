#test_package
#author: Facundo Franchino
"""
the package must import instantly and dependency-light.

`import rt_fdn` exposes the whole api at the root, but must never
pull torch or flamo in by itself: those load lazily, only when model
reconstruction (json_to_flamo) is actually requested.
"""

from __future__ import annotations

import subprocess
import sys


def test_root_import_pulls_no_heavy_dependencies():
    #subprocess so the check is immune to whatever this test session
    #has already imported
    code = (
        "import rt_fdn, sys; "
        "print('torch' in sys.modules, 'flamo' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-2:] == ["False", "False"]


def test_full_api_at_the_root():
    import rt_fdn

    for name in ("flamo_to_json", "json_to_faust", "flamo_to_faust",
                 "HotReload", "certify", "write_certificate", "export_juce"):
        assert callable(getattr(rt_fdn, name))
    #json_to_flamo is reachable at the root too, resolved lazily
    assert "json_to_flamo" in rt_fdn.__all__

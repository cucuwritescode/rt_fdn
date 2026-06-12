#test_hotreload
#author: Facundo Franchino
"""
tests for the training-time hot-reload publisher.

uses the same lightweight mock modules as the other test files so the
tests run without torch or flamo installed, and tmp_path fixtures so
no real plugin paths are touched.
"""

from __future__ import annotations

import numpy as np
import pytest

import rt_fdn.hotreload as hotreload_module
from rt_fdn.hotreload import HotReload


#mock flamo module (matches conventions in test_flamo_to_json.py)

class _MockParam:
    """stand-in for torch nn.Parameter, supports detach().cpu().numpy()."""

    def __init__(self, values: np.ndarray):
        self._values = np.asarray(values, dtype=np.float64)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values.copy()


class parallelGain:
    """mock flamo parallelGain module whose values can be mutated."""

    def __init__(self, values: np.ndarray):
        self.param = _MockParam(np.asarray(values, dtype=np.float64))
        self.input_channels = len(values)
        self.output_channels = len(values)

    def set_values(self, values: np.ndarray):
        self.param = _MockParam(np.asarray(values, dtype=np.float64))


@pytest.fixture
def paths(tmp_path):
    """dsp and config paths inside the pytest temp directory."""
    return tmp_path / "live.dsp", tmp_path / "faust-current-dsp.txt"


def _make(paths, **kwargs):
    dsp_path, config_path = paths
    defaults = dict(every=1, min_interval=0.0)
    defaults.update(kwargs)
    return HotReload(
        48000.0, name="Live", dsp_path=dsp_path, config_path=config_path,
        **defaults,
    )


class TestPublishing:
    def test_first_update_writes_dsp_and_config(self, paths):
        dsp_path, config_path = paths
        live = _make(paths)
        model = parallelGain(np.array([0.5, 0.25]))

        assert live.update(model) is True
        code = dsp_path.read_text()
        assert "process = " in code
        assert "*(0.5)" in code
        #config first line names the dsp path, trimmed
        assert config_path.read_text().splitlines()[0] == str(dsp_path)

    def test_unchanged_model_publishes_once(self, paths):
        live = _make(paths)
        model = parallelGain(np.array([0.5]))

        assert live.update(model) is True
        #identical code must not rewrite the file or trigger a reload
        assert live.update(model) is False
        assert live.update(model) is False

    def test_changed_model_republishes(self, paths):
        dsp_path, _ = paths
        live = _make(paths)
        model = parallelGain(np.array([0.5]))

        assert live.update(model) is True
        model.set_values(np.array([0.75]))
        assert live.update(model) is True
        assert "*(0.75)" in dsp_path.read_text()

    def test_no_temp_files_left_behind(self, paths):
        dsp_path, _ = paths
        live = _make(paths)
        model = parallelGain(np.array([0.5]))
        live.update(model)
        model.set_values(np.array([0.6]))
        live.update(model)

        leftovers = list(dsp_path.parent.glob("*.tmp"))
        assert leftovers == []

    def test_wrong_config_is_corrected(self, paths):
        dsp_path, config_path = paths
        config_path.write_text("/somewhere/else.dsp\n")
        live = _make(paths)
        live.update(parallelGain(np.array([0.5])))

        assert config_path.read_text().splitlines()[0] == str(dsp_path)


class TestThrottling:
    def test_every_n_steps(self, paths):
        live = _make(paths, every=3)
        model = parallelGain(np.array([0.5]))

        published = []
        for step in range(7):
            model.set_values(np.array([0.5 + 0.01 * step]))
            published.append(live.update(model))

        #calls 1, 4, 7 qualify under every=3
        assert published == [True, False, False, True, False, False, True]

    def test_min_interval_suppresses_fast_publishes(self, paths, monkeypatch):
        clock = [0.0]
        monkeypatch.setattr(hotreload_module.time, "monotonic", lambda: clock[0])

        live = _make(paths, min_interval=1.0)
        model = parallelGain(np.array([0.5]))

        assert live.update(model) is True
        #half a second later: change is suppressed
        clock[0] = 0.5
        model.set_values(np.array([0.6]))
        assert live.update(model) is False
        #past the interval: published
        clock[0] = 1.5
        assert live.update(model) is True

    def test_force_bypasses_throttles(self, paths, monkeypatch):
        clock = [0.0]
        monkeypatch.setattr(hotreload_module.time, "monotonic", lambda: clock[0])

        live = _make(paths, every=10, min_interval=100.0)
        model = parallelGain(np.array([0.5]))

        assert live.update(model) is True
        model.set_values(np.array([0.9]))
        #throttled by both every and min_interval
        assert live.update(model) is False
        #force publishes the final state regardless
        assert live.update(model, force=True) is True

    def test_force_still_skips_identical_code(self, paths):
        live = _make(paths)
        model = parallelGain(np.array([0.5]))
        assert live.update(model) is True
        #nothing changed: even a forced update writes nothing
        assert live.update(model, force=True) is False

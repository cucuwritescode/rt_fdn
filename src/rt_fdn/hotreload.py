#hotreload
#author: Facundo Franchino
"""
publish flamo models to the faust hot-reload plugin during training

the clap hot-reload plugin (faust/architecture/clap/simple-faust.cpp)
watches a config file (/tmp/faust-current-dsp.txt by default) whose
first line names the .dsp file to load, and additionally watches that
.dsp file itself. any add/modify/move event on either file triggers a
reload through the faust interpreter, which takes in the order of
50-200 ms. parameters are preserved across reloads by address.

this module closes the loop from the training side: construct a
HotReload once before the training loop, then call update(model) after
each optimiser step. the model is re-emitted to faust and published
only when the generated code actually changed, subject to step and
time throttles, so a paused or converged optimisation causes no reloads.

    from rt_fdn import HotReload

    live = HotReload(fs=48000, name="MyReverb")
    for step in range(n_steps):
        loss = criterion(model(x), target)
        loss.backward()
        optimiser.step()
        live.update(model)
    live.update(model, force=True)

writes are atomic (temp file + rename in the same directory) so the
plugin never compiles a half-written file.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

from rt_fdn.codegen.flamo_to_faust import flamo_to_faust

#paths matching the defaults baked into the clap plugin
DEFAULT_CONFIG_PATH = "/tmp/faust-current-dsp.txt"
DEFAULT_DSP_PATH = "/tmp/rt-fdn-live.dsp"


def _atomic_write(path: Path, text: str) -> None:
    """write text to a file atomically via a sibling temp file and rename.

    the plugin compiles the .dsp the moment its watcher fires, so a
    plain open/write/close risks a compile of a half-written file.
    os.replace is atomic when source and target share a filesystem
    (guaranteed here, the temp file lives in the target directory)
    and the rename raises a moved event that the watcher treats as a
    reload trigger, exactly like an in-place modification.
    """
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, str(path))
    except BaseException:
        #never leave temp files behind on failure
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class HotReload:
    """re-emit a flamo model to faust and publish it to the running plugin.

    parameters
    ----------
    fs : float
        sampling rate in hz, forwarded to flamo_to_faust().
    name : str
        dsp name, appears in the generated faust header comment.
    dsp_path : str | Path
        where the generated .dsp is written. the basename must stay
        stable across the run: the plugin matches reload events
        against the basename of the file it has loaded.
    config_path : str | Path
        the plugin's watched config file. its first line is kept
        pointing at dsp_path.
    every : int
        publish at most every n-th call to update(). the first call
        always qualifies. default 1 (every call).
    min_interval : float
        minimum seconds between published reloads. protects the
        plugin from reload storms when the training loop runs faster
        than the interpreter can swap factories. default 0.5.
    controls : dict, optional
        macro controls ("rt60", "dry_wet", "pre_delay") forwarded to
        flamo_to_faust(). slider addresses stay stable across
        publishes, and the plugin preserves parameter values by
        address, so knob positions survive every reload.
    """

    def __init__(
        self,
        fs: float,
        *,
        name: str = "FDN",
        dsp_path: str | Path = DEFAULT_DSP_PATH,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        every: int = 1,
        min_interval: float = 0.5,
        controls: dict[str, Any] | None = None,
    ):
        self.fs = fs
        self.name = name
        self.controls = controls
        self.dsp_path = Path(dsp_path)
        self.config_path = Path(config_path)
        self.every = max(1, int(every))
        self.min_interval = float(min_interval)
        self._calls = 0
        self._last_publish_time = float("-inf")
        self._last_code: str | None = None

    def update(self, model: Any, *, force: bool = False) -> bool:
        """regenerate faust code from the model and publish it if due.

        call this after each optimiser step. returns True when a new
        .dsp was actually written (and the plugin will reload), False
        when the call was throttled or the generated code is unchanged.

        force=True bypasses the step and time throttles, for a final
        guaranteed publish after the training loop ends.
        """
        self._calls += 1
        if not force:
            if (self._calls - 1) % self.every != 0:
                return False
            if time.monotonic() - self._last_publish_time < self.min_interval:
                return False
        code = flamo_to_faust(
            model, self.fs, name=self.name, controls=self.controls
        )
        return self.publish(code)

    def publish(self, code: str) -> bool:
        """publish a faust source string to the plugin.

        lower-level entry point for callers who generate code
        themselves (for instance via json_to_faust after editing the
        config). skips the write when the code is identical to the
        last published version, so no spurious reload fires.
        """
        if code == self._last_code:
            return False
        _atomic_write(self.dsp_path, code)
        self._point_plugin_at_dsp()
        self._last_code = code
        self._last_publish_time = time.monotonic()
        return True

    def _point_plugin_at_dsp(self) -> None:
        """keep the config file's first line naming our dsp path.

        rewritten only when wrong: the plugin also reloads on config
        file changes, and rewriting an already-correct config would
        trigger a second, redundant reload per publish.
        """
        target = str(self.dsp_path)
        try:
            current = self.config_path.read_text().splitlines()
            if current and current[0].strip() == target:
                return
        except OSError:
            pass
        _atomic_write(self.config_path, target + "\n")

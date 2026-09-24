#!/usr/bin/env python3
"""Small stdlib-only regression test for the Track1 token timestep contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_layout_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "diffsynth"
        / "pipelines"
        / "dual_stream_timestep_contract.py"
    )
    return _load_module("dual_stream_timestep_contract", path)


def test_track1_conditional_rgb_token_timesteps() -> None:
    module = _load_layout_module()
    layout = module.dual_stream_timestep_layout(
        rgb_temporal=3,
        rgb_spatial=2,
        flow_temporal=3,
        flow_spatial=1,
    )
    timestep = 700
    values = [0] * layout.total_tokens
    values[layout.rgb_future] = [timestep] * (
        layout.rgb_future.stop - layout.rgb_future.start
    )
    assert values == [0, 0, 700, 700, 700, 700, 0, 0, 0]
    assert values[layout.rgb_first] == [0, 0]
    assert values[layout.rgb_future] == [700, 700, 700, 700]
    assert values[layout.flow_all] == [0, 0, 0]


def test_conditional_objective_freezes_flow_head() -> None:
    path = Path(__file__).resolve().parents[1] / "training" / "objective_contract.py"
    module = _load_module("objective_contract", path)

    class Parameter:
        def __init__(self):
            self.requires_grad = True

        def numel(self):
            return 3

    class Head:
        def __init__(self):
            self.values = [Parameter(), Parameter()]

        def requires_grad_(self, enabled):
            for value in self.values:
                value.requires_grad = enabled

        def named_parameters(self):
            return [(f"p{index}", value) for index, value in enumerate(self.values)]

        def parameters(self):
            return iter(self.values)

    flow_stream = type("FlowStream", (), {"flow_head": Head()})()
    frozen = module.freeze_conditional_flow_head(
        flow_stream, module.TRACK1_CONDITIONAL_RGB
    )
    assert frozen == 6
    assert not any(value.requires_grad for value in flow_stream.flow_head.values)


if __name__ == "__main__":
    test_track1_conditional_rgb_token_timesteps()
    test_conditional_objective_freezes_flow_head()
    print("Track1 conditional RGB timestep contract: PASS")

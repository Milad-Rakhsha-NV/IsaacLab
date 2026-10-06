# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Release environment-path regexes select native Newton sensor rows."""

import newton
import pytest
import warp as wp
from isaaclab_newton.physics import NewtonManager


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("kind", ["body", "shape"])
def test_contact_sensor_resolves_replicated_path_regex(device, kind, monkeypatch):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    builder = newton.ModelBuilder()
    for world in range(3):
        builder.begin_world()
        path = f"/World/envs/env_{world}/Robot"
        for name in ("base", "left_FOOT", "right_FOOT"):
            body = builder.add_link(label=f"{path}/{name}")
            builder.add_articulation([builder.add_joint_free(body)])
            builder.add_shape_sphere(body, radius=0.1, label=f"{path}/{name}/collision")
        builder.end_world()
    model = builder.finalize(device=device)
    monkeypatch.setattr(NewtonManager, "_model", model)
    monkeypatch.setattr(NewtonManager, "_solver", None)
    monkeypatch.setattr(NewtonManager, "_newton_contact_sensors", {})
    monkeypatch.setattr(NewtonManager, "_report_contacts", False)
    expression = r"/World/envs/env_[^/]+/Robot/(left|right)_FOOT"
    kwargs = {"body_names_expr": expression} if kind == "body" else {"shape_names_expr": expression + "/collision"}
    key = NewtonManager.add_contact_sensor(**kwargs)
    sensor = NewtonManager._newton_contact_sensors[key]
    assert sensor.sensing_indices == [1, 2, 4, 5, 7, 8]
    assert sensor.total_force.shape == (6,)
    assert sensor.total_force_friction.shape == (6,)

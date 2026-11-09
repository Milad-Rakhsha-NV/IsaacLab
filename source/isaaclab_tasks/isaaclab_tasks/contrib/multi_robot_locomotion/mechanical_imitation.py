# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action fitting for the mechanical Ant source-competence diagnostic."""

import torch


def descriptor_roundoff(actual: dict, reference: dict) -> dict:
    """Accept only absolute roundoff up to 1e-12; integer topology and all tensor shapes must match.

    USD transform composition can produce tiny residual translations instead of exact zeros
    at different replication counts. Return every nonzero maximum difference for the manifest.
    This deliberately uses zero relative tolerance, including on large sentinel values.
    """
    if actual.keys() != reference.keys():
        raise ValueError("Descriptor prototypes differ from the teacher.")
    differences = {}
    for name, fields in actual.items():
        if fields.keys() != reference[name].keys():
            raise ValueError("Descriptor fields differ from the teacher.")
        for key, value in fields.items():
            expected = reference[name][key]
            try:
                torch.testing.assert_close(value, expected, rtol=0, atol=1e-12 if value.is_floating_point() else 0)
            except AssertionError as error:
                raise ValueError(f"Physical descriptor changed: {name}/{key}.") from error
            maximum = float((value.double() - expected.double()).abs().max())
            if maximum:
                differences[f"{name}/{key}"] = maximum
    return differences


def balanced_action_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, embodiment: torch.Tensor
) -> torch.Tensor:
    """Average squared raw-action error over valid actuators, then samples, then bodies.

    Actions use the task's dimensionless, pre-clipping command units. Each body has equal
    weight even with unequal sample counts or actuator counts. Padded actions have zero
    contribution and gradient. Clipping before this loss would discard teacher information
    used by the task's previous-action observation and action penalty.
    """
    error = (prediction - target).masked_fill(~mask, 0).square()
    per_sample = error.sum(-1) / mask.sum(-1).clamp_min(1)
    return torch.stack([per_sample[embodiment == body].mean() for body in embodiment.unique()]).mean()

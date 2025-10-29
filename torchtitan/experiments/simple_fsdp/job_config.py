# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Compile:
    model_backend_override: str | None = None
    """Override backend to compile in simplefsdp. Additional backend includes aot_eager_autobucketing"""


@dataclass
class Parallelism:
    simple_fsdp_reshard_after_forward: Literal["default", "always", "never"] = "default"
    """
    `simple_fsdp_reshard_after_forward` specifies the policy for applying `reshard_after_forward`
    within an SimpleFSDP setup. `reshard_after_forward` controls parameter behavior after forward,
    trading off memory and communication.

    The supported policies include "default", "always" and "never":

    - "default" applies default resharding behavior, implementing "smart defaults" for known optimal
      scenarios.
    - "always" will enable `reshard_after_forward` for all forward passes.
    - "never" will disable `reshard_after_forward` for all forward passes.
    """


@dataclass
class JobConfig:
    compile: Compile = field(default_factory=Compile)
    parallelism: Parallelism = field(default_factory=Parallelism)

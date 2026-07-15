# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import os
import sys

try:
    from .version import __version__  # noqa
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()

__all__ = ["pdb"]

# backwards compatibility to support `from eccore.X import Y`
from eccore.distributed import utils as distributed_utils
from eccore.logging import meters, metrics, progress_bar  # noqa

sys.modules["eccore.distributed_utils"] = distributed_utils
sys.modules["eccore.meters"] = meters
sys.modules["eccore.metrics"] = metrics
sys.modules["eccore.progress_bar"] = progress_bar

import eccore.losses  # noqa
import eccore.distributed  # noqa
import eccore.models  # noqa
import eccore.modules  # noqa
import eccore.optim  # noqa
import eccore.optim.lr_scheduler  # noqa
import eccore.tasks  # noqa


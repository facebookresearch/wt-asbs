# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="facebook/wt-asbs",
    local_dir=Path(__file__).parent.parent,
    allow_patterns=["md_data/"],
)

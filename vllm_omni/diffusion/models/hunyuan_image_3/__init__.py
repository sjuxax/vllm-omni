# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hunyuan Image 3 diffusion model components.

This package deliberately avoids eager imports so submodules like
``system_prompt`` can be imported without recursively loading the pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_image_3_transformer import (
        HunyuanImage3Model,
        HunyuanImage3Text2ImagePipeline,
    )
    from vllm_omni.diffusion.models.hunyuan_image_3.pipeline_hunyuan_image_3 import (
        HunyuanImage3Pipeline,
    )

__all__ = [
    "HunyuanImage3Pipeline",
    "HunyuanImage3Model",
    "HunyuanImage3Text2ImagePipeline",
]


def __getattr__(name: str) -> Any:
    if name in {"HunyuanImage3Model", "HunyuanImage3Text2ImagePipeline"}:
        from vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_image_3_transformer import (
            HunyuanImage3Model,
            HunyuanImage3Text2ImagePipeline,
        )

        if name == "HunyuanImage3Model":
            return HunyuanImage3Model
        return HunyuanImage3Text2ImagePipeline

    if name == "HunyuanImage3Pipeline":
        from vllm_omni.diffusion.models.hunyuan_image_3.pipeline_hunyuan_image_3 import (
            HunyuanImage3Pipeline,
        )

        return HunyuanImage3Pipeline

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

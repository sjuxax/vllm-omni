# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for routing HunyuanImage3 CoT through a staged AR -> diffusion flow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.diffusion.models.hunyuan_image_3.system_prompt import get_system_prompt
from vllm_omni.inputs.data import OmniTextPrompt

logger = init_logger(__name__)

HUNYUAN_ORIGINAL_PROMPT_KEY = "hunyuan_original_prompt"
HUNYUAN_BOT_TASK_KEY = "hunyuan_bot_task"
HUNYUAN_COT_SYSTEM_PROMPT_KEY = "hunyuan_cot_system_prompt"
HUNYUAN_COT_TEXT_KEY = "hunyuan_cot_text"

HUNYUAN_COT_TASKS = {"think", "recaption", "think_recaption"}


def is_hunyuan_cot_task(bot_task: str | None) -> bool:
    return bot_task in HUNYUAN_COT_TASKS


def get_hunyuan_first_bot_task(bot_task: str | None) -> str:
    if bot_task is None:
        return "image"
    if bot_task.startswith("think"):
        return "think"
    if "recaption" in bot_task:
        return "recaption"
    return "image"


def wrap_hunyuan_stage0_prompt(
    prompt: Mapping[str, Any] | str,
    bot_task: str,
) -> OmniTextPrompt:
    """Build the AR-stage prompt for Hunyuan CoT requests.

    The stage-0 vLLM model expects a plain text prompt plus optional raw images.
    We keep the original prompt payload in ``additional_information`` so the
    downstream diffusion stage can reconstruct the request without depending on
    the stage-0 prompt text format.
    """
    if isinstance(prompt, Mapping):
        original_prompt: dict[str, Any] = dict(prompt)
    else:
        original_prompt = {"prompt": prompt}

    prompt_text = str(original_prompt.get("prompt") or "")
    first_bot_task = get_hunyuan_first_bot_task(bot_task)
    system_prompt = get_system_prompt(sys_type="dynamic", bot_task=first_bot_task)
    multimodal = original_prompt.get("multi_modal_data") or {}
    image_count = _count_input_images(multimodal)
    assistant_prefix = "<think>" if first_bot_task == "think" else "<recaption>"

    # Hunyuan's pretrain template concatenates the system prompt, image markers,
    # user text, and assistant prefix directly.
    stage0_prompt_text = "".join(
        part
        for part in (
            system_prompt or "",
            "<img>" * image_count,
            prompt_text,
            assistant_prefix,
        )
        if part
    )

    wrapped_prompt: OmniTextPrompt = {"prompt": stage0_prompt_text}
    if original_prompt.get("multi_modal_data") is not None:
        wrapped_prompt["multi_modal_data"] = original_prompt["multi_modal_data"]
    if original_prompt.get("negative_prompt") is not None:
        wrapped_prompt["negative_prompt"] = original_prompt["negative_prompt"]

    additional_information = _copy_additional_information(original_prompt)
    original_prompt_payload = _copy_prompt_payload(original_prompt)
    original_prompt_payload.pop("multi_modal_data", None)
    additional_information[HUNYUAN_ORIGINAL_PROMPT_KEY] = original_prompt_payload
    additional_information[HUNYUAN_BOT_TASK_KEY] = bot_task
    additional_information[HUNYUAN_COT_SYSTEM_PROMPT_KEY] = system_prompt
    wrapped_prompt["additional_information"] = additional_information
    return wrapped_prompt


def ar2diffusion(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTextPrompt | TextPrompt | list[OmniTextPrompt | TextPrompt] | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTextPrompt]:
    """Convert stage-0 Hunyuan CoT outputs into stage-1 diffusion prompts."""
    del requires_multimodal_data
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    source_outputs = stage_list[source_stage_id].engine_outputs
    if source_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    prompt_list = prompt if isinstance(prompt, list) else [prompt]
    diffusion_inputs: list[OmniTextPrompt] = []

    for idx, source_output in enumerate(source_outputs):
        source_prompt = prompt_list[idx] if idx < len(prompt_list) else None
        source_prompt_dict = _prompt_to_dict(source_prompt)
        source_metadata = _copy_additional_information(source_prompt_dict)
        original_prompt_dict = _prompt_to_dict(
            source_metadata.get(HUNYUAN_ORIGINAL_PROMPT_KEY, source_prompt_dict)
        )

        diffusion_prompt: OmniTextPrompt = {
            "prompt": str(original_prompt_dict.get("prompt") or ""),
        }
        if original_prompt_dict.get("negative_prompt") is not None:
            diffusion_prompt["negative_prompt"] = original_prompt_dict["negative_prompt"]
        if original_prompt_dict.get("multi_modal_data") is not None:
            diffusion_prompt["multi_modal_data"] = original_prompt_dict["multi_modal_data"]
        elif source_prompt_dict.get("multi_modal_data") is not None:
            diffusion_prompt["multi_modal_data"] = source_prompt_dict["multi_modal_data"]

        diffusion_metadata = _copy_additional_information(original_prompt_dict)
        bot_task = source_metadata.get(HUNYUAN_BOT_TASK_KEY)
        if is_hunyuan_cot_task(bot_task):
            output = source_output.outputs[0]
            generated_text = getattr(output, "text", "") or ""
            diffusion_metadata[HUNYUAN_BOT_TASK_KEY] = bot_task
            diffusion_metadata[HUNYUAN_COT_SYSTEM_PROMPT_KEY] = source_metadata.get(
                HUNYUAN_COT_SYSTEM_PROMPT_KEY
            )
            diffusion_metadata[HUNYUAN_COT_TEXT_KEY] = normalize_hunyuan_cot_text(generated_text, str(bot_task))
            logger.info("Prepared Hunyuan CoT text for diffusion request %s", source_output.request_id)

        if diffusion_metadata:
            diffusion_prompt["additional_information"] = diffusion_metadata
        diffusion_inputs.append(diffusion_prompt)

    return diffusion_inputs


def normalize_hunyuan_cot_text(generated_text: str, bot_task: str) -> str:
    normalized = generated_text.lstrip()
    if normalized.startswith("<think>") or normalized.startswith("<recaption>"):
        return generated_text
    if get_hunyuan_first_bot_task(bot_task) == "think":
        return f"<think>{generated_text}"
    return f"<recaption>{generated_text}"


def _count_input_images(multimodal: Mapping[str, Any]) -> int:
    images = multimodal.get("image")
    if images is None:
        images = multimodal.get("images")
    if images is None:
        return 0
    if isinstance(images, list):
        return len(images)
    return 1


def _copy_additional_information(prompt_dict: Mapping[str, Any]) -> dict[str, Any]:
    additional_information = prompt_dict.get("additional_information")
    if isinstance(additional_information, Mapping):
        return dict(additional_information)
    return {}


def _copy_prompt_payload(prompt: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(prompt)
    if isinstance(copied.get("additional_information"), Mapping):
        copied["additional_information"] = dict(copied["additional_information"])
    return copied


def _prompt_to_dict(prompt: Mapping[str, Any] | str | None) -> dict[str, Any]:
    if prompt is None:
        return {}
    if isinstance(prompt, Mapping):
        return dict(prompt)
    return {"prompt": prompt}


__all__ = [
    "HUNYUAN_BOT_TASK_KEY",
    "HUNYUAN_COT_SYSTEM_PROMPT_KEY",
    "HUNYUAN_COT_TASKS",
    "HUNYUAN_COT_TEXT_KEY",
    "HUNYUAN_ORIGINAL_PROMPT_KEY",
    "ar2diffusion",
    "get_hunyuan_first_bot_task",
    "is_hunyuan_cot_task",
    "normalize_hunyuan_cot_text",
    "wrap_hunyuan_stage0_prompt",
]

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
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_RECAPTION_OPEN = "<recaption>"
_RECAPTION_CLOSE = "</recaption>"


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
    assistant_prefix = _THINK_OPEN if first_bot_task == "think" else _RECAPTION_OPEN
    stage0_prompt_text = _build_stage0_prompt_text(
        prompt_text=prompt_text,
        image_count=image_count,
        system_prompt=system_prompt,
        assistant_prefix=assistant_prefix,
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


def build_hunyuan_stage0_followup_prompt(
    prompt: Mapping[str, Any] | str,
    bot_task: str,
    cot_prefix_text: str,
) -> OmniTextPrompt:
    """Build the second-stage AR prompt for staged think_recaption requests."""
    if isinstance(prompt, Mapping):
        original_prompt: dict[str, Any] = dict(prompt)
    else:
        original_prompt = {"prompt": prompt}

    prompt_text = str(original_prompt.get("prompt") or "")
    first_bot_task = get_hunyuan_first_bot_task(bot_task)
    system_prompt = get_system_prompt(sys_type="dynamic", bot_task=first_bot_task)
    multimodal = original_prompt.get("multi_modal_data") or {}
    image_count = _count_input_images(multimodal)

    stage0_prompt_text = _build_stage0_prompt_text(
        prompt_text=prompt_text,
        image_count=image_count,
        system_prompt=system_prompt,
        cot_prefix_text=cot_prefix_text,
        assistant_prefix=_RECAPTION_OPEN,
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


def build_hunyuan_diffusion_prompt(
    prompt: Mapping[str, Any] | str,
    bot_task: str,
    cot_text: str,
    *,
    system_prompt: str | None = None,
) -> OmniTextPrompt:
    """Build a direct stage-1 diffusion prompt from finalized CoT text."""
    original_prompt = _prompt_to_dict(prompt)
    diffusion_prompt: OmniTextPrompt = {
        "prompt": str(original_prompt.get("prompt") or ""),
    }
    if original_prompt.get("negative_prompt") is not None:
        diffusion_prompt["negative_prompt"] = original_prompt["negative_prompt"]
    if original_prompt.get("multi_modal_data") is not None:
        diffusion_prompt["multi_modal_data"] = original_prompt["multi_modal_data"]

    additional_information = _copy_additional_information(original_prompt)
    additional_information[HUNYUAN_BOT_TASK_KEY] = bot_task
    additional_information[HUNYUAN_COT_TEXT_KEY] = cot_text
    if system_prompt is not None:
        additional_information[HUNYUAN_COT_SYSTEM_PROMPT_KEY] = system_prompt
    if additional_information:
        diffusion_prompt["additional_information"] = additional_information
    return diffusion_prompt


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
    sanitized = sanitize_hunyuan_cot_text(generated_text, bot_task)
    if sanitized:
        return sanitized

    if get_hunyuan_first_bot_task(bot_task) == "think":
        return f"{_THINK_OPEN}{generated_text}"
    return f"{_RECAPTION_OPEN}{generated_text}"


def extract_hunyuan_revised_prompt(cot_text: str) -> str:
    recaption_block = _extract_tag_block(cot_text, _RECAPTION_OPEN, _RECAPTION_CLOSE)
    if recaption_block is not None:
        return recaption_block[len(_RECAPTION_OPEN) : -len(_RECAPTION_CLOSE)].strip()

    think_block = _extract_tag_block(cot_text, _THINK_OPEN, _THINK_CLOSE)
    if think_block is not None:
        return think_block[len(_THINK_OPEN) : -len(_THINK_CLOSE)].strip()

    return cot_text.strip()


def sanitize_hunyuan_cot_text(generated_text: str, bot_task: str) -> str:
    text = generated_text.strip()
    if not text:
        return text

    first_bot_task = get_hunyuan_first_bot_task(bot_task)
    wants_recaption = "recaption" in bot_task

    think_block = _extract_tag_block(
        text if _THINK_OPEN in text else f"{_THINK_OPEN}{text}",
        _THINK_OPEN,
        _THINK_CLOSE,
        fallback_end_tags=("</answer>", "<|endoftext|>", _RECAPTION_OPEN),
    ) if first_bot_task == "think" else None

    recaption_source = text
    if think_block is not None and _THINK_CLOSE in think_block:
        think_close_idx = text.find(_THINK_CLOSE)
        if think_close_idx != -1:
            recaption_source = text[think_close_idx + len(_THINK_CLOSE) :]

    recaption_block = _extract_tag_block(
        recaption_source if _RECAPTION_OPEN in recaption_source else f"{_RECAPTION_OPEN}{recaption_source}",
        _RECAPTION_OPEN,
        _RECAPTION_CLOSE,
        fallback_end_tags=("</answer>", "<|endoftext|>"),
    ) if wants_recaption or first_bot_task == "recaption" else None

    if think_block is not None and recaption_block is not None:
        return f"{think_block}{recaption_block}"
    if think_block is not None:
        return think_block
    if recaption_block is not None:
        return recaption_block

    return text


def _extract_tag_block(
    text: str,
    start_tag: str,
    end_tag: str,
    *,
    fallback_end_tags: tuple[str, ...] = (),
) -> str | None:
    start_idx = text.find(start_tag)
    if start_idx == -1:
        return None

    end_idx = text.find(end_tag, start_idx + len(start_tag))
    closing_tag = end_tag
    if end_idx == -1:
        fallback_candidates = [
            text.find(tag, start_idx + len(start_tag))
            for tag in fallback_end_tags
            if text.find(tag, start_idx + len(start_tag)) != -1
        ]
        if fallback_candidates:
            end_idx = min(fallback_candidates)
        else:
            return text[start_idx:]

    return text[start_idx:end_idx] + closing_tag


def _count_input_images(multimodal: Mapping[str, Any]) -> int:
    images = multimodal.get("image")
    if images is None:
        images = multimodal.get("images")
    if images is None:
        return 0
    if isinstance(images, list):
        return len(images)
    return 1


def _build_stage0_prompt_text(
    *,
    prompt_text: str,
    image_count: int,
    system_prompt: str | None,
    assistant_prefix: str,
    cot_prefix_text: str = "",
) -> str:
    # Hunyuan's pretrain template concatenates system prompt, image markers,
    # user text, any prior CoT text, and the next assistant prefix directly.
    return "".join(
        part
        for part in (
            system_prompt or "",
            "<img>" * image_count,
            prompt_text,
            cot_prefix_text,
            assistant_prefix,
        )
        if part
    )


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
    "build_hunyuan_diffusion_prompt",
    "build_hunyuan_stage0_followup_prompt",
    "extract_hunyuan_revised_prompt",
    "get_hunyuan_first_bot_task",
    "is_hunyuan_cot_task",
    "normalize_hunyuan_cot_text",
    "sanitize_hunyuan_cot_text",
    "wrap_hunyuan_stage0_prompt",
]

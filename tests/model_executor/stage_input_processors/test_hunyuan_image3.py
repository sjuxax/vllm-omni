# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm_omni.model_executor.stage_input_processors.hunyuan_image3 import (
    HUNYUAN_BOT_TASK_KEY,
    HUNYUAN_COT_SYSTEM_PROMPT_KEY,
    HUNYUAN_COT_TEXT_KEY,
    ar2diffusion,
    extract_hunyuan_revised_prompt,
    normalize_hunyuan_cot_text,
    wrap_hunyuan_stage0_prompt,
)


def test_wrap_hunyuan_stage0_prompt_preserves_original_payload():
    wrapped = wrap_hunyuan_stage0_prompt(
        {
            "prompt": "replace the sky with a storm",
            "negative_prompt": "blurry",
            "multi_modal_data": {"image": ["image-a.png", "image-b.png"]},
        },
        "think_recaption",
    )

    assert wrapped["prompt"].endswith("<think>")
    assert wrapped["prompt"].count("<img>") == 2
    metadata = wrapped["additional_information"]
    assert metadata[HUNYUAN_BOT_TASK_KEY] == "think_recaption"
    assert metadata["hunyuan_original_prompt"]["prompt"] == "replace the sky with a storm"


def test_hunyuan_ar2diffusion_restores_original_prompt_and_attaches_cot_text():
    source_output = SimpleNamespace(
        request_id="req-0",
        outputs=[SimpleNamespace(text="step by step</think><recaption>storm clouds</recaption>")],
    )
    stage_list = [SimpleNamespace(engine_outputs=[source_output])]
    wrapped_prompt = wrap_hunyuan_stage0_prompt(
        {
            "prompt": "replace the sky with a storm",
            "negative_prompt": "blurry",
            "multi_modal_data": {"image": ["image-a.png"]},
        },
        "think_recaption",
    )

    diffusion_inputs = ar2diffusion(stage_list, [0], wrapped_prompt)

    assert len(diffusion_inputs) == 1
    diffusion_prompt = diffusion_inputs[0]
    assert diffusion_prompt["prompt"] == "replace the sky with a storm"
    assert diffusion_prompt["negative_prompt"] == "blurry"
    assert diffusion_prompt["multi_modal_data"]["image"] == ["image-a.png"]
    assert diffusion_prompt["additional_information"][HUNYUAN_COT_TEXT_KEY].startswith("<think>")
    assert diffusion_prompt["additional_information"][HUNYUAN_BOT_TASK_KEY] == "think_recaption"
    assert diffusion_prompt["additional_information"][HUNYUAN_COT_SYSTEM_PROMPT_KEY] is not None


def test_normalize_hunyuan_cot_text_truncates_generation_spill():
    noisy_output = (
        "first pass</think><recaption>storm clouds over the city</recaption>"
        "<|endoftext|><recaption>garbage tail</recaption>"
    )

    normalized = normalize_hunyuan_cot_text(noisy_output, "think_recaption")

    assert normalized == "<think>first pass</think><recaption>storm clouds over the city</recaption>"
    assert extract_hunyuan_revised_prompt(normalized) == "storm clouds over the city"


def test_normalize_hunyuan_cot_text_accepts_answer_close_as_recaption_end():
    noisy_output = "drafting</think><recaption>storm clouds over the city</answer><|endoftext|>"

    normalized = normalize_hunyuan_cot_text(noisy_output, "think_recaption")

    assert normalized == "<think>drafting</think><recaption>storm clouds over the city</recaption>"
    assert extract_hunyuan_revised_prompt(normalized) == "storm clouds over the city"

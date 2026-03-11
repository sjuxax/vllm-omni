# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from unittest.mock import MagicMock

from vllm_omni.model_executor.stage_input_processors.hunyuan_image3 import (
    HUNYUAN_BOT_TASK_KEY,
    HUNYUAN_COT_SYSTEM_PROMPT_KEY,
    HUNYUAN_COT_TEXT_KEY,
    ar2diffusion,
    build_hunyuan_stage0_followup_prompt,
    extract_hunyuan_revised_prompt,
    normalize_hunyuan_cot_text,
    resolve_hunyuan_stop_token_ids,
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


def test_build_hunyuan_stage0_followup_prompt_appends_recaption_stage():
    wrapped = build_hunyuan_stage0_followup_prompt(
        {
            "prompt": "replace the sky with a storm",
            "multi_modal_data": {"image": ["image-a.png"]},
        },
        "think_recaption",
        "<think>reason about the composition</think>",
    )

    assert wrapped["prompt"].endswith("<recaption>")
    assert "</think><recaption>" in wrapped["prompt"]
    assert wrapped["additional_information"][HUNYUAN_BOT_TASK_KEY] == "think_recaption"


def _make_fake_tokenizer(token_map: dict[str, int]):
    tok = MagicMock()
    tok.convert_tokens_to_ids = lambda t: token_map.get(t)
    tok.eos_token_id = token_map.get("<|endoftext|>", 0)
    return tok


def test_resolve_hunyuan_stop_token_ids_think():
    tok = _make_fake_tokenizer({
        "</think>": 100,
        "</answer>": 101,
        "<boi>": 102,
        "<|endoftext|>": 103,
        "</recaption>": 104,
    })
    ids = resolve_hunyuan_stop_token_ids(tok, "think")
    assert 100 in ids  # </think>
    assert 101 in ids  # </answer>
    assert 102 in ids  # <boi>
    assert 103 in ids  # eos
    assert 104 not in ids  # </recaption> NOT included for think


def test_resolve_hunyuan_stop_token_ids_recaption():
    tok = _make_fake_tokenizer({
        "</think>": 100,
        "</answer>": 101,
        "<boi>": 102,
        "<|endoftext|>": 103,
        "</recaption>": 104,
    })
    ids = resolve_hunyuan_stop_token_ids(tok, "recaption")
    assert 104 in ids  # </recaption>
    assert 101 in ids  # </answer>
    assert 102 in ids  # <boi>
    assert 103 in ids  # eos
    assert 100 not in ids  # </think> NOT included for recaption

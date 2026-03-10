# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
HunyuanImage-3.0-Instruct: Text-to-Image with Chain-of-Thought Reasoning

This example demonstrates the `think_recaption` bot_task which performs
two-stage autoregressive text generation before image diffusion:

  1. **Think** – the model generates a <think>...</think> block with structured
     reasoning about subject, composition, environment, lighting, etc.
  2. **Recaption** – the model generates a <recaption>...</recaption> block
     containing a refined, professional-grade prompt.
  3. **Diffuse** – the full CoT trace is fed into the diffusion stage as
     context, producing a higher-fidelity image.

Usage
-----
    python text_to_image_cot.py --prompt "A cat sitting on a windowsill watching the rain"

Supported bot_task values:
    think_recaption  –  Full think → recaption → image pipeline (default)
    recaption        –  Recaption-only (skip the think stage)
    think            –  Think-only (skip the recaption stage)
    image            –  Direct image generation (no CoT, same as default pipeline)
"""

import argparse
import time
from pathlib import Path

import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an image with HunyuanImage-3.0-Instruct using chain-of-thought reasoning."
    )
    parser.add_argument(
        "--model",
        default="tencent/HunyuanImage-3.0-Instruct",
        help="Model name or local path.",
    )
    parser.add_argument(
        "--prompt",
        default="A cat sitting on a windowsill watching the rain",
        help="Text prompt for image generation.",
    )
    parser.add_argument(
        "--bot-task",
        default="think_recaption",
        choices=["think_recaption", "recaption", "think", "image"],
        help="Bot task controlling CoT behavior (default: think_recaption).",
    )
    parser.add_argument(
        "--use-system-prompt",
        default="dynamic",
        choices=["dynamic", "en_unified", "en_vanilla", "en_recaption", "en_think_recaption", "None"],
        help="System prompt variant (default: dynamic – auto-select based on bot_task).",
    )
    parser.add_argument(
        "--drop-think",
        action="store_true",
        help="If set, strip the <think> block from CoT before feeding into diffusion "
             "(keeps only <recaption>). Reduces context length at the cost of some quality.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--height", type=int, default=1024, help="Image height.")
    parser.add_argument("--width", type=int, default=1024, help="Image width.")
    parser.add_argument(
        "--num-inference-steps", type=int, default=50, help="Number of denoising steps."
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=5.0, help="Classifier-free guidance scale."
    )
    parser.add_argument(
        "--output", type=str, default="hunyuan_cot_output.png", help="Output image path."
    )
    parser.add_argument(
        "--enforce-eager", action="store_true", help="Disable torch.compile."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    omni = Omni(
        model=args.model,
        enforce_eager=args.enforce_eager,
    )

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    extra_args = {
        "bot_task": args.bot_task,
    }

    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        generator=generator,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        extra_args=extra_args,
    )

    print(f"\n{'=' * 60}")
    print(f"  Model:       {args.model}")
    print(f"  Prompt:      {args.prompt}")
    print(f"  Bot task:    {args.bot_task}")
    print(f"  Image size:  {args.width}x{args.height}")
    print(f"  Steps:       {args.num_inference_steps}")
    print(f"  Guidance:    {args.guidance_scale}")
    print(f"  Seed:        {args.seed}")
    print(f"  Drop think:  {args.drop_think}")
    print(f"{'=' * 60}\n")

    start = time.perf_counter()
    outputs = omni.generate(
        {"prompt": args.prompt},
        sampling_params,
    )
    elapsed = time.perf_counter() - start

    if not outputs or len(outputs) == 0:
        raise ValueError("No output generated from omni.generate()")

    first_output = outputs[0]
    req_out = first_output.request_output[0]

    # Print CoT text if available
    custom_output = getattr(req_out, "custom_output", None) or {}
    cot_text = custom_output.get("cot_text")
    if cot_text:
        print(f"\n{'─' * 60}")
        print("Chain-of-Thought Output:")
        print(f"{'─' * 60}")
        print(cot_text[0])
        print(f"{'─' * 60}\n")

    # Save image
    images = req_out.images
    if not images:
        raise ValueError("No images found in output")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path)
    print(f"Saved image to {output_path}")
    print(f"Total generation time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()

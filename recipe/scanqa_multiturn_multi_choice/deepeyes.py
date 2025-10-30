# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations
from collections import Counter
from typing import Iterable, List
import string

import io
import logging
import os
import random
import re
import json
import numpy as np

import requests
from openai import OpenAI
from PIL import Image

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.vision_utils import process_image, process_video


logger = logging.getLogger(__name__)

from io import BytesIO
from PIL import Image

def resize_processed_image(
    image: dict | Image.Image,
    width: int,
    height: int,
    *,
    keep_aspect: bool = False,
    resample: Image.Resampling = Image.Resampling.LANCZOS
) -> Image.Image:
    """
    Resize an image produced by `process_image` without cropping or padding.

    Args:
        image: Either a PIL Image or the same dict consumed by `process_image`.
        width, height: Target dimensions in pixels.
        keep_aspect: If False (default), resize to exactly (width, height) which
            can change aspect ratio (no crop, no pad). If True, preserve aspect
            ratio and 'fit' inside the box; the returned image may be smaller in
            one dimension than requested.
        resample: PIL resampling filter (default LANCZOS).

    Returns:
        PIL.Image.Image resized accordingly.
    """
    im = process_image(image)  # ensures RGB PIL image

    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("`width` and `height` must be positive integers.")

    if keep_aspect:
        src_w, src_h = im.size
        scale = min(width / src_w, height / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        return im.resize((new_w, new_h), resample=resample)

    # Exact resize to requested resolution (may change aspect ratio).
    return im.resize((width, height), resample=resample)


class CustomRLHFDataset(RLHFDataset):
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        row_dict[self.prompt_key] = [
            {
                "role": "system",
                # We don't need tool description, because custom_chat_template will add it.
                "content": (
                    "You are a helpful assistant. You can call functions to assist with the user query. "
                    "Important: You must call only one function at a time. After each function call, "
                    "wait for the execution result before making the next function call if needed."
                ),
            },
            {
                "role": "user",
                "content": row_dict[self.prompt_key][0]["content"],
            },
        ]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                # high_resolution_images = [process_image(image) for image in row_dict_images]
                # images = [resize_processed_image(high_resolution_image.copy(), 336, 224) for high_resolution_image in high_resolution_images]

                images = [process_image(image) for image in row_dict_images]

                high_resolution_image_paths = row_dict.get("image_paths")
                high_resolution_images = [process_image(image) for image in high_resolution_image_paths]

                # high_resolution_images = images

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205  # noqa: E501
                multi_modal_data["image"] = images

            model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = {
            "image_resize_tool": {
                "create_kwargs": {"images": high_resolution_images},
                # "execute_kwargs": {},
                # "calc_reward_kwargs": {},
                # "release_kwargs": {},
            }
        }
        # tools_kwargs = {
        #     "image_zoom_in_tool": {
        #         "create_kwargs": {"image": images[0]},
        #         # "execute_kwargs": {},
        #         # "calc_reward_kwargs": {},
        #         # "release_kwargs": {},
        #     }
        # }
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["agent_name"] = "tool_agent"
        return row_dict


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Compute reward score for model solutions with robust handling of various formats.

    Returns a weighted combination of:
    - Accuracy reward (0.8 weight): Whether the answer is semantically correct
    - Format reward (0.2 weight): Whether the output follows expected format
    - Tool reward (1.2 weight): Whether tools were used when answer is correct
    """
    # Initialize tracking variables
    is_format_error = False

    solution_str = solution_str.split("assistant")[-1]

    # 1. Check <think> tag format
    count_think_1 = solution_str.count("<think>")
    count_think_2 = solution_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    # 2. Check vision tokens (skip this since tokenizer removes special tokens)
    # We'll use <tool_call> and <tool_response> instead to detect tool usage

    # 3. Extract answer text with multiple fallback strategies
    answer_text = ""

    # Strategy 1: Try to extract from <answer> tags first
    predict_no_think = (
        solution_str.split("</think>")[1].strip() if "</think>" in solution_str else solution_str.strip()
    )

    # Check <answer> tag format
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    # Try to extract from <answer> tags
    answer_match = re.search(r"<answer>(.*?)</answer>", predict_no_think, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        answer_text = predict_no_think.strip()
    # else:
    #     # No proper <answer> tags found - this is a format error
    #     is_format_error = True

    #     # Strategy 2: If no <answer> tags, extract content after tool responses
    #     # Look for pattern: <tool_response>...</tool_response>assistant\n[actual_answer]
    #     tool_response_match = re.search(
    #         r"</tool_response>\s*assistant\s*\n(.*?)$", predict_no_think, re.DOTALL | re.MULTILINE
    #     )
    #     if tool_response_match:
    #         answer_text = tool_response_match.group(1).strip()
    #     else:
    #         # Strategy 3: If no tool responses, look for content after </think>
    #         if "</think>" in solution_str:
    #             # Remove any remaining tool-related tags and extract meaningful content
    #             remaining_content = predict_no_think
    #             # Remove tool calls and responses
    #             remaining_content = re.sub(r"<tool_call>.*?</tool_call>", "", remaining_content, flags=re.DOTALL)
    #             remaining_content = re.sub(
    #                 r"<tool_response>.*?</tool_response>", "", remaining_content, flags=re.DOTALL
    #             )
    #             # Remove user/assistant markers
    #             remaining_content = re.sub(r"\b(user|assistant)\b", "", remaining_content)
    #             answer_text = remaining_content.strip()
    #         else:
    #             # Strategy 4: Use the entire solution_str as fallback
    #             answer_text = solution_str.strip()

    # Clean up answer text
    answer_text = answer_text.strip()

    golds = json.loads(ground_truth)
    em_reward = exact_match(answer_text, golds)
    f1_reward = f1_max_over_refs(answer_text, golds)
    acc_reward = 0.5 * em_reward + 1.5 * f1_reward

    # Penalize excessively long answers (potential judge hacking)
    if len(answer_text) >= 200:
        acc_reward = 0.0
        is_format_error = True
    else:
        # acc_reward = 1.0
        pass

    # # 5. Check tool usage - look for tool_call/tool_response patterns instead of vision tokens
    tool_rewards = extra_info.get("tool_rewards", [0.0])
    nums = [float(x) for x in tool_rewards if isinstance(x, (int, float))]
    tool_reward = float(sum(nums) / len(nums)) if nums else 0.0

    # Format reward: penalty for format errors
    format_reward = -1.0 if is_format_error else 0.0

    # Final weighted score
    final_score = 0.8 * acc_reward + 0.2 * format_reward + 1.2 * (tool_reward * int(acc_reward > 0.5))

    return final_score

    
#!/usr/bin/env python3
"""
ScanQA-style EM (Exact Match) and token-level F1 with shared normalization.

Normalization (SQuAD-style + light number mapping):
- lowercase
- strip whitespace, collapse multiple spaces
- remove punctuation
- remove English articles: a, an, the
- map simple number-words -> digits (e.g., "two" -> "2")

Both metrics take the max over references (gold answers + aliases).
"""

# ---------- Normalization helpers ----------

_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

def _normalize(s: str | None) -> str:
    """SQuAD-style normalization plus simple number-word mapping."""
    if s is None:
        return ""
    s = s.lower().strip()
    # number words -> digits (token-wise; avoids 'stone' -> 'st1e')
    toks = [ _NUM_WORDS.get(t, t) for t in s.split() ]
    s = " ".join(toks)
    # strip punctuation
    s = s.translate(_PUNCT_TABLE)
    # remove articles
    toks = [t for t in s.split() if t not in _ARTICLES]
    # collapse spaces
    return " ".join(toks)

def _tokenize(s: str | None) -> List[str]:
    s = _normalize(s)
    return s.split() if s else []


# ---------- Metrics ----------

def exact_match(pred: str, refs: Iterable[str]) -> int:
    """Return 1 if normalized pred equals any normalized ref, else 0."""
    p = _normalize(pred)
    for r in refs:
        if p == _normalize(r):
            return 1
    return 0

def _f1_single(pred: str, ref: str) -> float:
    """Token-level F1 for a single reference."""
    p_toks = _tokenize(pred)
    r_toks = _tokenize(ref)

    if not p_toks and not r_toks:
        return 1.0
    if not p_toks or not r_toks:
        return 0.0

    p_cnt = Counter(p_toks)
    r_cnt = Counter(r_toks)
    overlap = sum((p_cnt & r_cnt).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(p_toks)
    recall = overlap / len(r_toks)
    return 2 * precision * recall / (precision + recall)

def f1_max_over_refs(pred: str, refs: Iterable[str]) -> float:
    """Max token-level F1 over references."""
    refs = list(refs)
    if not refs:
        return 0.0
    return max(_f1_single(pred, r) for r in refs)


if __name__ == "__main__":
    # Test case 1: No tool, well-formatted answer (JSON ground truth)
    predict_str = "<think>Quick reasoning...</think><answer>left</answer>"
    ground_truth = '["left", "to the left"]'
    extra_info = {
        "question": "Is the woman to the left or to the right of the man who is holding the camera?",
    }
    print("=== Test Case 1: No tool, well-formatted ===")
    import time
    time_start = time.time()
    score = compute_score("common_reasoning", predict_str, ground_truth, extra_info)
    print(f"Score: {score}")
    time_end = time.time()
    print(f"Time: {time_end - time_start}")

    # Test case 2: Tool used, missing <answer> tags (format error, but tool usage detected)
    problematic_solution = """<tool_call>
{"name": "image_resize_tool", "arguments": {"timestamp": -1}}
</tool_call>user
<tool_response>
Selected image at timestamp -1.
</tool_response>
assistant
Yes, the white van is indeed situated in the bottom part of the picture."""
    problematic_ground_truth = '["Yes, the white van is indeed situated in the bottom part of the picture."]'
    problematic_extra_info = {
        "question": "Is the white van in the bottom part of the picture?",
    }

    print("\n=== Test Case 2: Tool used, missing <answer> tags ===")
    print(f"Solution: {problematic_solution}")
    print(f"Ground truth: {problematic_ground_truth}")

    time_start = time.time()
    score2 = compute_score("common_reasoning", problematic_solution, problematic_ground_truth, problematic_extra_info)
    print(f"Score: {score2}")
    time_end = time.time()
    print(f"Time: {time_end - time_start}")

    # Test case 3: Well-formatted case with tool
    well_formatted_solution = """<think>
I need to review a different frame to answer confidently.
</think>
<tool_call>
{"name": "image_resize_tool", "arguments": {"timestamp": -1}}
</tool_call>
<tool_response>
Selected image at timestamp -1.
</tool_response>
<answer>Yes, the white van is indeed situated in the bottom part of the picture.</answer>"""

    print("\n=== Test Case 3: Well-formatted case with tool ===")
    time_start = time.time()
    score3 = compute_score(
        "common_reasoning", well_formatted_solution, problematic_ground_truth, problematic_extra_info
    )
    print(f"Score: {score3}")
    time_end = time.time()
    print(f"Time: {time_end - time_start}")
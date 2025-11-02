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
        prev_tool_name = "image_resize_tool"
        current_tool_name = "temporal_zoom_tool"

        prev_format_requirement = "Format strictly as <think>...</think><tool_call>...</tool_call>(if tools needed)<answer>...</answer>."
        current_format_requirement = "For the final answer, format strictly as <think>...</think><answer>...</answer>."

        row_dict: dict = self.dataframe[item]
        row_dict[self.prompt_key] = [
            {
                "role": "user",
                "content": row_dict[self.prompt_key][0]["content"].replace(prev_tool_name, current_tool_name).replace(prev_format_requirement, current_format_requirement),
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
                images = [process_image(image) for image in row_dict_images]

                all_images = list()
                for k, imgs in row_dict["non_key_frame_image_paths"].items():
                    all_images.extend([process_image({"image": image}) for image in imgs])

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

        assert all_images is not None, f'There is trouble with {row_dict}, image None'

        tools_kwargs = {
            "temporal_zoom_tool": {
                "create_kwargs": {"images": all_images},
                # "execute_kwargs": {},
                # "calc_reward_kwargs": {},
                # "release_kwargs": {},
            }
        }
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

    raw_solution_str = solution_str

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
        solution_str.split("</think>")[-1].strip() if "</think>" in solution_str else solution_str.strip()
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
        is_format_error = True

    # Clean up answer text
    answer_text = answer_text.strip()

    golds = json.loads(ground_truth)
    acc_reward_candidate = []
    for gold in golds:
        em_reward = (em_refined(answer_text, gold))
        bleu4_reward = (bleu4_score(answer_text, gold))
        rouge_l_reward = (rouge_l_score(answer_text, gold))
        cider_n_reward = (cider_n(answer_text, gold)) * 2       # NOTE: score for exact match is 0.5 somehow
        current_acc_reward = 0.3 * em_reward + 0.01 * bleu4_reward + 0.01 * rouge_l_reward + 0.3 * cider_n_reward
        acc_reward_candidate.append(current_acc_reward)
    acc_reward = max(acc_reward_candidate)

    # Penalize excessively long answers (potential judge hacking)
    if len(answer_text) >= 200:
        acc_reward = 0.0
        is_format_error = True
    else:
        # acc_reward = 1.0
        pass

    # # 5. Check tool usage - look for tool_call/tool_response patterns instead of vision tokens
    has_tool_usage = bool(
        re.search(r"<tool_call>.*?</tool_call>", raw_solution_str, re.DOTALL)
        and re.search(r"<tool_response>.*?</tool_response>", raw_solution_str, re.DOTALL)
    )

    tool_rewards = extra_info.get("tool_rewards", [0.0])
    nums = [float(x) for x in tool_rewards if isinstance(x, (int, float))]
    tool_reward = float(sum(nums) / len(nums)) if nums else 0.0
    tool_reward = tool_reward * int(acc_reward > 0.5)

    # Format reward: penalty for format errors
    format_reward = -1.0 if is_format_error else 0.0

    # Final weighted score
    final_score = 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward

    return final_score

    
import math
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Iterable, Optional

# ----------------------------
# Helpers
# ----------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")

def _maybe_extract_answer(s: str) -> str:
    """
    If your outputs are wrapped like <answer>...</answer>, extract that span.
    Otherwise return the original string unchanged.
    """
    m = re.search(r"<answer>(.*?)</answer>", s, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else s

def normalize_text(s: str) -> str:
    """
    'Refined' normalization commonly used for EM in QA:
    - extract <answer>...</answer> if present
    - lowercase
    - strip
    - collapse whitespace
    - remove punctuation (tweak if your official eval keeps punctuation)
    """
    s = _maybe_extract_answer(s)
    s = s.lower().strip()
    s = _WHITESPACE_RE.sub(" ", s)
    s = _PUNCT_RE.sub("", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s

def tokenize_words(s: str) -> List[str]:
    s = normalize_text(s)
    return s.split() if s else []

def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)] if n > 0 else []

# ----------------------------
# 1) Exact Match (refined)
# ----------------------------

def em_refined(output_str: str, target_str: str) -> int:
    """
    Returns 1 if normalized output == normalized target, else 0.
    """
    return int(normalize_text(output_str) == normalize_text(target_str))

# ----------------------------
# 2) BLEU-4 (sentence-level with smoothing)
#   - Returns 0..100 (percentage scale)
# ----------------------------

def bleu4_score(output_str: str, target_str: str) -> float:
    """
    Sentence-level BLEU-4 with smoothing (method-1: add-one).
    Single reference version.
    Returns BLEU-4 on a 0..100 scale.
    """
    cand = tokenize_words(output_str)
    ref = tokenize_words(target_str)
    if len(cand) == 0:
        return 0.0
    # Modified n-gram precisions with clipping
    precisions = []
    for n in range(1, 5):
        cand_ngrams = Counter(ngrams(cand, n))
        ref_ngrams = Counter(ngrams(ref, n))
        overlap = 0
        total = 0
        for g, c in cand_ngrams.items():
            overlap += min(c, ref_ngrams.get(g, 0))
            total += c
        # smoothing: +1 / +1
        p_n = (overlap + 1.0) / (total + 1.0) if total > 0 else 1.0
        precisions.append(p_n)

    # Brevity penalty (single ref)
    c = len(cand)
    r = len(ref)
    if c == 0:
        return 0.0
    bp = 1.0 if c > r else math.exp(1 - float(r) / max(1, c))

    # geometric mean of precisions
    log_prec = sum(math.log(p) for p in precisions) / 4.0
    bleu = bp * math.exp(log_prec)
    return 100.0 * bleu

# ----------------------------
# 3) ROUGE-L (F1 over LCS)
#   - Returns 0..100
# ----------------------------

def _lcs_length(a: List[str], b: List[str]) -> int:
    """
    Longest Common Subsequence length (O(len(a)*len(b)) DP).
    """
    la, lb = len(a), len(b)
    dp = [0] * (lb + 1)
    for i in range(1, la + 1):
        prev = 0
        for j in range(1, lb + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[lb]

def rouge_l_score(output_str: str, target_str: str, beta: float = 1.0) -> float:
    """
    ROUGE-L F-measure between candidate and single reference (0..100).
    """
    cand = tokenize_words(output_str)
    ref = tokenize_words(target_str)
    if len(cand) == 0 or len(ref) == 0:
        return 0.0
    lcs = _lcs_length(cand, ref)
    prec = lcs / len(cand)
    rec = lcs / len(ref)
    if prec == 0 and rec == 0:
        return 0.0
    beta2 = beta * beta
    f1 = (1 + beta2) * prec * rec / (rec + beta2 * prec) if (rec + beta2 * prec) > 0 else 0.0
    return 100.0 * f1

# ----------------------------
# 4) CIDEr (lightweight proxy of CIDEr-D)
#   - Returns cider on 0..100 scale
#   - Accepts optional 'idf_corpus' to compute IDF; if None, uses TF-only (idf=1)
# ----------------------------

def _build_df(corpus_refs: Iterable[str], n_max: int = 4) -> Dict[int, Dict[Tuple[str, ...], int]]:
    """
    Build document frequency (DF) counts for n-grams 1..n_max over a reference corpus.
    DF is number of refs in which the n-gram appears at least once.
    """
    df: Dict[int, Dict[Tuple[str, ...], int]] = {n: defaultdict(int) for n in range(1, n_max + 1)}
    for ref_text in corpus_refs:
        tokens = tokenize_words(ref_text)
        for n in range(1, n_max + 1):
            seen = set(ngrams(tokens, n))
            for g in seen:
                df[n][g] += 1
    return df

def _tf_vector(tokens: List[str], n: int) -> Dict[Tuple[str, ...], float]:
    g = ngrams(tokens, n)
    counts = Counter(g)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}  # normalized TF

def _cosine_sim_weighted(tf_c: Dict, tf_r: Dict, idf: Dict[Tuple[str, ...], float]) -> float:
    # dot
    dot = 0.0
    for g, wc in tf_c.items():
        if g in tf_r:
            w = idf.get(g, 1.0)
            dot += wc * tf_r[g] * (w * w)
    # norms
    def _norm(tf):
        return math.sqrt(sum((v * idf.get(g, 1.0)) ** 2 for g, v in tf.items()))
    nc = _norm(tf_c)
    nr = _norm(tf_r)
    if nc == 0 or nr == 0:
        return 0.0
    return dot / (nc * nr)

def cider_score_0_100(
    output_str: str,
    target_str: str,
    idf_corpus: Optional[Iterable[str]] = None,
    n_max: int = 4,
    sigma: float = 6.0
) -> float:
    """
    A lightweight CIDEr-D style proxy:
    - TF-IDF cosine similarity averaged over n=1..n_max
    - Gaussian length penalty (sigma as in CIDEr-D)
    - Scaled to 0..100 for convenience
    If idf_corpus is None, uses idf=1 for all n-grams (still useful; just lacks DF weighting).
    """
    cand = tokenize_words(output_str)
    ref = tokenize_words(target_str)

    # Build IDF if corpus provided
    idf_by_n: Dict[int, Dict[Tuple[str, ...], float]] = {n: defaultdict(lambda: 1.0) for n in range(1, n_max + 1)}
    if idf_corpus is not None:
        df = _build_df(idf_corpus, n_max=n_max)
        N = max(1, len(list(idf_corpus)))
        for n in range(1, n_max + 1):
            for g, d in df[n].items():
                # IDF per CIDEr: log((N + 1) / (df + 1))
                idf_by_n[n][g] = math.log((N + 1) / (d + 1))

    # Gaussian length penalty
    len_pen = math.exp(-((len(cand) - len(ref)) ** 2) / (2 * (sigma ** 2))) if sigma > 0 else 1.0

    sims = []
    for n in range(1, n_max + 1):
        tf_c = _tf_vector(cand, n)
        tf_r = _tf_vector(ref, n)
        sim = _cosine_sim_weighted(tf_c, tf_r, idf_by_n[n])
        sims.append(sim)

    cider = len_pen * (sum(sims) / max(1, len(sims)))  # 0..1-ish
    return 100.0 * cider  # scale to 0..100

# ----------------------------
# 5) Normalized CIDEr in [0,1]
# ----------------------------

def cider_n(output_str: str, target_str: str, idf_corpus: Optional[Iterable[str]] = None) -> float:
    """
    Normalized CIDEr proxy in [0,1], by dividing the 0..100 variant by 100 and clipping.
    """
    c = cider_score_0_100(output_str, target_str, idf_corpus=idf_corpus)
    return max(0.0, min(c / 100.0, 1.0))

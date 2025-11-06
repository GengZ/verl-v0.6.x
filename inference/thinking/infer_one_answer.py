import json

from PIL import Image
from typing import Iterable, Any, List, Dict, Union, Sequence, Optional

import torch
from datasets import load_dataset, Dataset
from transformers import AutoProcessor, AutoModelForVision2Seq

from load_data import load_hf_parquet
from extract_think import extract_last_summary


def _frames_block_initial(initial_frames: List[Any]) -> List[Dict]:
    """Convert [img0, img1, ...] to alternating text–image blocks labeled as Frame-{k}."""
    block = []
    for k, img in enumerate(initial_frames):
        block.append({"type": "text", "text": f"Frame-{k}:"})
        block.append({"type": "image", "image": img})
    return block


def _frames_block_interval(interval_frames: List[Any], i: int) -> List[Dict]:
    """Convert [img_b0, img_b1, ...] to alternating text–image blocks labeled as Between-i-i+1-{t:02d}."""
    block = []
    for t, img in enumerate(interval_frames):
        block.append({"type": "text", "text": f"Between-{i}-{i+1}-{t:02d}:"})
        block.append({"type": "image", "image": img})
    return block


def build_user_prompt(
    initial_frames: List[Any],
    interval_frames: List[Any],
    i: int,
    question: str,
    answer: str,
    selection_thinking: str,
) -> Dict:
    """
    Build a {'role': 'user', 'content': [...]} message for VQA reasoning verification.

    Args:
        initial_frames: list of initial downsampled frame images.
        interval_frames: list of retrieved in-between frame images.
        i: index of the starting keyframe for the interval (i → i+1).
        question: the VQA question text.
        answer: the correct answer text.
        selection_thinking: reasoning or justification for selecting this interval (from previous step).
    """
    content: List[Dict] = []

    # ---- Context & Rules ----
    content.append({
        "type": "text",
        "text": (
            "You are a vision question answering verifier. "
            "You are given downsampled frames from a video and additional frames retrieved "
            "between two adjacent downsampled frames i and i+1. "
            "Your task is to verify why the selected interval and frames support the correct answer.\n\n"
            "Rules:\n"
            "1) Provide a short evidence summary, not step-by-step internal reasoning.\n"
            "2) Explicitly cite frame IDs (e.g., Frame-2, Between-2-3-05).\n"
            "3) Focus on visible evidence that supports the ground-truth answer.\n"
            "4) Keep the evidence summary 3–6 concise bullet points.\n"
            "5) Then output the final answer using the strict schema below."
        )
    })

    # ---- Question ----
    content.append({"type": "text", "text": f"Question:\n{question}"})

    # ---- Why these frames were selected ----
    content.append({
        "type": "text",
        "text": f"Why these frames were selected (model thinking):\n{selection_thinking}"
    })

    # ---- Ground-truth Answer ----
    content.append({
        "type": "text",
        "text": (
            f"Ground-truth Answer:\n{answer}\n\n"
            "Explain concisely why this answer is supported by the provided frames."
        )
    })

    # ---- Initial Downsampled Frames ----
    content.append({"type": "text", "text": "Initial Downsampled Frames (each labeled as Frame-{k}):"})
    content.extend(_frames_block_initial(initial_frames))

    # ---- Retrieved Interval Frames ----
    content.append({
        "type": "text",
        "text": (
            f"Retrieved Frames Between Frame-{i} and Frame-{i+1} "
            f"(each labeled as Between-{i}-{i+1}-{{t:02d}} in chronological order):"
        )
    })
    content.extend(_frames_block_interval(interval_frames, i))

    # ---- Output Schema ----
    content.append({
        "type": "text",
        "text": (
            "Output Schema (strict):\n"
            "<evidence_summary>\n"
            "- [Frame-ID(s)] Visual fact supporting the answer.\n"
            "- ... (3–6 bullets total)\n"
            "</evidence_summary>\n"
            f"<final_answer>{answer}</final_answer>"
        )
    })

    return {"role": "user", "content": content}

def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file into a list of dicts."""
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def pair_jsonl_with_parquet(jsonl_data: List[Dict], parquet_data: Dataset) -> List[Dict]:
    """Pair JSONL items with matching rows in Parquet data by question_id."""
    # Build a lookup dict: question_id -> row
    parquet_lookup = {
        row["extra_info"]["question_id"]: row
        for row in parquet_data
    }

    # Assign row if exists in lookup
    for item in jsonl_data:
        qid = item["question_id"]
        if qid in parquet_lookup:
            item["row"] = parquet_lookup[qid]

    return jsonl_data

def _to_pil_rgb(img_like: Any) -> Image.Image:
    if isinstance(img_like, Image.Image):
        return img_like.convert("RGB")
    if isinstance(img_like, str):
        return Image.open(img_like).convert("RGB")
    # Allow dicts like {"image": "/path/to.jpg"} from your original example
    if isinstance(img_like, dict) and "image" in img_like:
        return _to_pil_rgb(img_like["image"])
    raise TypeError(f"Unsupported image input type: {type(img_like)}")


def _format_intervals_text(intervals: Sequence[int]) -> str:
    """
    Render intervals as human text for the prompt.
    For a single i -> 'Frames between i and i+1'
    For multiple -> 'Frames between 3 and 4; 7 and 8; 12 and 13'
    """
    pairs = [f"Frames between {i} and {i + 1}" for i in intervals]
    return "; ".join(pairs)

@torch.inference_mode()
def run_one_example(
    initial_frames: Union[Sequence[Any], Sequence[Dict[str, Any]]],
    interval_frames: Union[Sequence[Any], Sequence[Dict[str, Any]]],
    i: int,
    question: str,
    answer: str,
    selection_thinking: str,
    model,
    processor: Optional[AutoProcessor] = None,
    *,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    do_sample: bool = True
) -> str:
    """
    Run one example using a loaded Qwen2.5-VL model, using the same prompt format as `build_user_prompt`.

    Args:
        initial_frames: List of PIL.Image, file paths, or dicts like {"image": "/path"} for downsampled frames.
        interval_frames: List of PIL.Image, file paths, or dicts like {"image": "/path"} for retrieved in-between frames.
        i: Interval start index (verifying evidence between Frame-i and Frame-(i+1)).
        question: The VQA question.
        answer: The ground-truth answer to be verified.
        selection_thinking: Prior reasoning/justification for selecting this interval.
        model: A loaded AutoModelForVision2Seq (e.g., Qwen2.5-VL-* Instruct).
        processor: Optional AutoProcessor. If None, inferred from model.config.
        temperature, max_new_tokens, do_sample: Generation params.

    Returns:
        The decoded model output text.
    """
    device = next(model.parameters()).device
    model.eval()

    # Resolve processor if not provided
    if processor is None:
        model_id_like = getattr(getattr(model, "config", None), "_name_or_path", None)
        if model_id_like is None:
            raise ValueError("processor is None and model.config._name_or_path is unavailable; "
                             "please pass `processor` explicitly.")
        processor = AutoProcessor.from_pretrained(model_id_like, trust_remote_code=True)

    # Normalize to PIL RGB and preserve order: initial frames, then interval frames
    pil_initial: List[Image.Image] = [_to_pil_rgb(im) for im in initial_frames]
    pil_interval: List[Image.Image] = [_to_pil_rgb(im) for im in interval_frames]
    all_pil_images: List[Image.Image] = pil_initial + pil_interval

    # Build messages using the same schema as `main()`
    user_msg = build_user_prompt(
        initial_frames=pil_initial,
        interval_frames=pil_interval,
        i=i,
        question=question,
        answer=answer,
        selection_thinking=selection_thinking,
    )
    messages = [user_msg]

    # Tokenize chat + prepare pixel values
    input_ids = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(device)

    inputs = {"input_ids": input_ids}
    image_inputs = processor.image_processor(all_pil_images, return_tensors="pt").to(device)
    inputs.update(image_inputs)

    # Generate
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )

    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_text

def main():
    jsonl_data = load_jsonl("/home/geng/git/verl_0.6/outputs/reasoning.jsonl")
    parquet_data = load_hf_parquet("/workspace/data/verl/scanqa_images_16_keyframes_120_non_keyframes_504x504_with_label/train.parquet")
    jsonl_data = pair_jsonl_with_parquet(jsonl_data, parquet_data)

    # 1) Load model and processor once
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # 2) Iterate over paired items and generate outputs
    for item in jsonl_data:
        row = item.get("row")
        if row is None:
            continue

        # initial frames
        initial_frames = row["images"]
        initial_frames = [_to_pil_rgb(im) for im in initial_frames]

        # interval frames
        i = item["interval"]
        interval_frames = row["non_key_frame_image_paths"][str(i)]
        interval_frames = [_to_pil_rgb(im) for im in interval_frames]

        # question, answer, thinking
        question = row["extra_info"]["question"]
        answer = row["extra_info"]["answer"]
        thinking = item["think"]

        # run model with unified prompt
        output_text = run_one_example(
            initial_frames=initial_frames,
            interval_frames=interval_frames,
            i=i,
            question=question,
            answer=answer,
            selection_thinking=thinking,
            model=model,
            processor=processor,
            temperature=0.7,
            max_new_tokens=512,
            do_sample=True,
        )

        record = {
            "question_id": item['row']["extra_info"]["question_id"],
            "row_index": item['row_index'],
            "interval": item['interval'],
            "question": item['question'],
            "answer": item['row']["extra_info"]["answer"],
            "thinking": item['think'],
            "output": output_text,
            "summary": extract_last_summary(output_text),
        }

        with open("/home/geng/git/verl_0.6/outputs/reasoning_answer.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
    exit()
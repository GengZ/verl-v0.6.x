from PIL import Image
from typing import Iterable, Any, List, Dict, Union, Sequence, Optional

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

def make_text_image_contents(
    images: Iterable[Any],
    start: int = 0,
    prefix: str = "Frame",
    sep: str = "-"
) -> List[Dict]:
    """
    Build a flat list like:
    [{'type': 'text', 'text': 'Frame-0: '}, {'type': 'image', 'image': <img0>}, ...]
    """
    out: List[Dict] = []
    for i, img in enumerate(images, start=start):
        out.append({"type": "text", "text": f"{prefix}{sep}{i}: "})
        out.append({"type": "image", "image": img})
    return out


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
    images: Union[Sequence[Any], Sequence[Dict[str, Any]]],
    groundtruth_intervals: Union[int, Sequence[int]],
    question: str,
    model,
    processor: Optional[AutoProcessor] = None,
    *,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    do_sample: bool = True
) -> str:
    """
    Run one example using a loaded Qwen2.5-VL model.

    Args:
        images: List of PIL.Image, file paths, or dicts like {"image": "/path"}.
        groundtruth_intervals: A single interval index i or a list of i's.
        question: The user question.
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
        # Best-effort: derive processor from model config (works for HF models)
        model_id_like = getattr(getattr(model, "config", None), "_name_or_path", None)
        if model_id_like is None:
            raise ValueError("processor is None and model.config._name_or_path is unavailable; "
                             "please pass `processor` explicitly.")
        processor = AutoProcessor.from_pretrained(model_id_like, trust_remote_code=True)

    # Normalize intervals to a list[int]
    if isinstance(groundtruth_intervals, int):
        intervals_list = [groundtruth_intervals]
    else:
        intervals_list = list(map(int, groundtruth_intervals))

    intervals_text = _format_intervals_text(intervals_list)

    # Load/normalize images to PIL RGB
    pil_images: List[Image.Image] = [_to_pil_rgb(im) for im in images]

    # Compose chat messages
    image_contents = make_text_image_contents(pil_images, start=0, prefix="Frame", sep="-")

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are an expert multimodal reasoning assistant. Your goal is to explain "
                        "WHY the PROVIDED ground-truth frame interval(s) — the segment between frame i and frame i + 1 — "
                        "are the correct regions that contain key evidence needed to answer the question.\n\n"
                        "Context:\n"
                        "• The given frames are DOWNSAMPLED snapshots from a longer video.\n"
                        "• Each frame is labeled by its index (frame 0, frame 1, frame 2, …).\n"
                        "• The ground-truth interval i corresponds to the temporal gap between frame i and frame i + 1 "
                        "that should be examined in finer detail.\n\n"
                        "Guidelines:\n"
                        "• DO NOT propose new intervals; only justify the provided one(s).\n"
                        "• DO NOT answer the question directly.\n"
                        "• Base reasoning strictly on the visible frames and question text.\n"
                        "• Discuss what each frame shows, what changes between consecutive frames, and "
                        "why the transition between frame i and i + 1 captures the key event or clue.\n\n"
                        "• Concise and clear reasoning.\n"
                        "• End with a line like 'I should focus on the transition between frame i and frame i + 1 for a closer look.'\n"
                        "Output format:\n"
                        "<think>…step-by-step reasoning that ends with the phrase "
                        "…</think>"
                        "<answer>intervals=<GROUNDTRUTH_INTERVALS>; summary=…</answer>"
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": image_contents + [
                {
                    "type": "text",
                    "text": (
                        f"Question: {question}\n\n"
                        "Explain why the following ground-truth frame interval(s) contain the key action "
                        "that needs closer inspection to answer the question. "
                        "Do not propose other intervals.\n\n"
                        f"Ground-truth key intervals (between frame i and frame i + 1): {intervals_text}"
                    ),
                },
            ],
        },
    ]

    # Tokenize chat + prepare pixel values
    input_ids = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(device)

    # `apply_chat_template` returns a tensor; wrap into dict expected by generate
    inputs = {"input_ids": input_ids}

    image_inputs = processor.image_processor(pil_images, return_tensors="pt").to(device)
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


if __name__ == "__main__":
    # -------------------------------------------------------------
    # 1. Load model and processor
    # -------------------------------------------------------------
    model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    images = \
    [
    {
        "image": "/workspace/data/scannet_sampled/min_532_long_edge_mp/scene0000_00/images/0.jpg"
    },
    {
        "image": "/workspace/data/scannet_sampled/min_532_long_edge_mp/scene0000_00/images/369.jpg"
    },
    {
        "image": "/workspace/data/scannet_sampled/min_532_long_edge_mp/scene0000_00/images/749.jpg"
    },
    {
        "image": "/workspace/data/scannet_sampled/min_532_long_edge_mp/scene0000_00/images/1118.jpg"
    },
    {
        "image": "/workspace/data/scannet_sampled/min_532_long_edge_mp/scene0000_00/images/1486.jpg"
    },
    {
        "image": "/workspace/data/scannet_sampled/min_532_long_edge_mp/scene0000_00/images/1855.jpg"
    },
    {
        "image": "/workspace/data/scannet_sampled/min_532_long_edge_mp/scene0000_00/images/2235.jpg"
    },
    {
        "image": "/workspace/data/scannet_sampled/min_532_long_edge_mp/scene0000_00/images/2604.jpg"
    },
    {
        "image": "/workspace/data/scannet_sampled/min_532_long_edge_mp/scene0000_00/images/2973.jpg"
    },
    {
        "image": "/workspace/data/scannet_sampled/min_532_long_edge_mp/scene0000_00/images/3342.jpg"
    },
    {
        "image": "/workspace/data/scannet_sampled/min_532_long_edge_mp/scene0000_00/images/3722.jpg"
    },
    {
        "image": "/workspace/data/scannet_sampled/min_532_long_edge_mp/scene0000_00/images/4091.jpg"
    },
    {
        "image": "/workspace/data/scannet_sampled/min_532_long_edge_mp/scene0000_00/images/4459.jpg"
    },
    {
        "image": "/workspace/data/scannet_sampled/min_532_long_edge_mp/scene0000_00/images/4828.jpg"
    },
    {
        "image": "/workspace/data/scannet_sampled/min_532_long_edge_mp/scene0000_00/images/5208.jpg"
    },
    {
        "image": "/workspace/data/scannet_sampled/min_532_long_edge_mp/scene0000_00/images/5577.jpg"
    }
    ]

    groundtruth_intervals = 13  # or [13]
    question = "The beige wooden bookshelf is placed next to what else?"

    text = run_one_example(images, groundtruth_intervals, question, model, processor)
    print("\n=== Generated Reasoning ===\n")
    print(text)

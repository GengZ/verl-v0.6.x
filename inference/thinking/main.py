# qwen_think_interval.py
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

from typing import Iterable, Any, List, Dict

def make_text_image_contents(
    images: Iterable[Any],
    start: int = 0,
    prefix: str = "Frame",
    sep: str = "-"
) -> List[Dict]:
    """
    Build a flat list like:
    [{'type': 'text', 'text': 'Frame-0'}, {'type': 'image', 'image': <img0>}, ...]

    Args:
        images: Iterable of image objects.
        start: Starting index for numbering.
        prefix: Text prefix before the index.
        sep: Separator between prefix and index.

    Returns:
        Flat list alternating text then image for each input image.
    """
    out: List[Dict] = []
    for i, img in enumerate(images, start=start):
        out.append({"type": "text", "text": f"{prefix}{sep}{i}: "})
        out.append({"type": "image", "image": img})
    return out

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

# -------------------------------------------------------------
# 2. Prepare input images and question
# -------------------------------------------------------------
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

image_files = [f["image"] for f in images]

groundtruth_intervals = [13]

print(images[int(groundtruth_intervals[0])])
print(images[int(groundtruth_intervals[0]) + 1])

groundtruth_intervals = f'Frames between {groundtruth_intervals[0]} and {groundtruth_intervals[0] + 1}'
print(groundtruth_intervals)

images = [Image.open(p).convert("RGB") for p in image_files]

question = "The beige wooden bookshelf is placed next to what else?"

# -------------------------------------------------------------
# 3. Compose Qwen-style chat messages
# -------------------------------------------------------------

image_contents = make_text_image_contents(images, start=0, prefix="Frame", sep="-")

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
                    f"Ground-truth key intervals (between frame i and frame i + 1): {groundtruth_intervals}"
                ),
            },
        ],
    },
]


# -------------------------------------------------------------
# 4. Process input (important: use processor.chat to get a dict)
# -------------------------------------------------------------
# The processor automatically combines chat and image inputs.
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
)
# `apply_chat_template` returns a *tensor*, so wrap it manually.
inputs = {"input_ids": inputs.to(model.device)}

# Optionally include pixel values if not already embedded in messages
image_inputs = processor.image_processor(images, return_tensors="pt").to(model.device)
inputs.update(image_inputs)

# -------------------------------------------------------------
# 5. Generate reasoning
# -------------------------------------------------------------
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )

output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("\n=== Generated Reasoning ===\n")
print(output_text)
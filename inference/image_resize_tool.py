from typing import Any, Dict, List, Tuple, Optional
from PIL import Image

def _extract_indices(args: Dict[str, Any]) -> List[int]:
    # Prefer schema key; fall back to common alias
    indices = args.get("frame_indices", args.get("frames", []))
    if isinstance(indices, str):
        # Accept comma-separated or space-separated string
        parts = [p.strip() for p in indices.replace(",", " ").split() if p.strip()]
        try:
            indices = [int(p) for p in parts]
        except Exception:
            indices = []
    if not isinstance(indices, list):
        return []
    out = []
    for x in indices:
        try:
            out.append(int(x))
        except Exception:
            continue
    # Keep order, dedupe while preserving first occurrence
    seen = set()
    ordered = []
    for i in out:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered

def _coerce_images(images: List[Any]) -> List[Image.Image]:
    # Accept list of PIL images, file paths, or dicts like {"image": path}
    coerced: List[Image.Image] = []
    for it in images:
        if isinstance(it, Image.Image):
            coerced.append(it)
        elif isinstance(it, str):
            coerced.append(Image.open(it).convert("RGB"))
        elif isinstance(it, dict) and "image" in it and isinstance(it["image"], str):
            coerced.append(Image.open(it["image"]).convert("RGB"))
        else:
            raise ValueError(f"Unsupported image item type: {type(it)}")
    return coerced

def _resize_keep_aspect(img: Image.Image, *, scale: Optional[float] = None, target_long_side: Optional[int] = None) -> Image.Image:
    w, h = img.size
    if scale is not None and scale > 0:
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    elif target_long_side is not None and target_long_side > 0:
        long = max(w, h)
        if long == 0:
            return img
        s = target_long_side / long
        new_w, new_h = max(1, int(round(w * s))), max(1, int(round(h * s)))
    else:
        # return original image
        print("Returning original image of size", img.size)
        return img
    return img.resize((new_w, new_h), resample=Image.BICUBIC)

def run_image_resize_tool_from_parsed_call(
    tool_call: Dict[str, Any],
    source_images: List[Any],
    *,
    scale: Optional[float] = None,
    target_long_side: Optional[int] = None,
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    """
    Inputs:
      - tool_call: one item from parse_agent_output()['tool_calls'].
                   Expects {'name': 'image_resize_tool', 'arguments': {'frame_indices': [..]}}
      - source_images: list of PIL.Image, file paths, or {'image': path} dicts.
      - scale or target_long_side: choose one (target_long_side wins if both None -> defaults to 672)

    Returns:
      - resized_images: list[Image.Image] in the order of requested indices
      - tool_message: a 'tool' role message you can append to messages and re-tokenize:
            {"role": "tool", "content": [{"type":"image"}, ..., {"type":"text","text":"..."}]}
    """
    if not isinstance(tool_call, dict):
        raise ValueError("tool_call must be a dict from parse_agent_output().")

    if tool_call.get("name") not in {"image_resize_tool", "resize_frames"}:
        raise ValueError(f"Unexpected tool name: {tool_call.get('name')}")

    args = tool_call.get("arguments", {}) or {}
    indices = _extract_indices(args)

    imgs = _coerce_images(source_images)
    total = len(imgs)
    valid_pairs = [(i, imgs[i]) for i in indices if 0 <= i < total]

    resized: List[Image.Image] = []
    for _, im in valid_pairs:
        resized.append(_resize_keep_aspect(im, scale=scale, target_long_side=target_long_side))

    # Build a simple tool message compatible with your chat template's tool_response branch
    # Add one {"type": "image"} per returned image so the processor aligns them.
    content = [{"type": "image"} for _ in resized]
    content.append({"type": "text", "text": f"Selected images at frame indices {indices}."})

    tool_message = {"role": "tool", "content": content}
    return resized, tool_message
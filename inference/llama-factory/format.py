import json, base64
from pathlib import Path

def to_jsonable(obj, *, embed_images=False, max_bytes=200_000):
    try:
        json.dumps(obj)  # fast path
        return obj
    except Exception:
        pass

    # PIL image
    try:
        from PIL import Image  # noqa
        import PIL
        if isinstance(obj, PIL.Image.Image):
            if not embed_images:
                return {"type": "PIL.Image", "mode": obj.mode, "size": obj.size}
            # embed as PNG (capped)
            from io import BytesIO
            buf = BytesIO()
            obj.save(buf, format="PNG")
            data = buf.getvalue()
            if len(data) > max_bytes:
                return {"type": "PIL.Image", "mode": obj.mode, "size": obj.size, "note": "omitted_image_too_large"}
            return {"type": "PIL.Image", "mode": obj.mode, "size": obj.size,
                    "data_base64_png": base64.b64encode(data).decode("utf-8")}
    except Exception:
        pass

    # torch tensor
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return {"type": "torch.Tensor", "dtype": str(obj.dtype),
                    "device": str(obj.device), "shape": list(obj.shape)}
    except Exception:
        pass

    # numpy array
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return {"type": "np.ndarray", "dtype": str(obj.dtype), "shape": list(obj.shape)}
    except Exception:
        pass

    # bytes -> base64 (small only)
    if isinstance(obj, (bytes, bytearray)):
        b = bytes(obj)
        if len(b) <= max_bytes:
            return {"type": "bytes", "len": len(b), "base64": base64.b64encode(b).decode("utf-8")}
        return {"type": "bytes", "len": len(b), "note": "omitted_bytes_too_large"}

    # pathlib.Path
    if isinstance(obj, Path):
        return str(obj)

    # sets/tuples
    if isinstance(obj, (set, tuple)):
        return [to_jsonable(x, embed_images=embed_images, max_bytes=max_bytes) for x in obj]

    # dict/list
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, embed_images=embed_images, max_bytes=max_bytes) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(x, embed_images=embed_images, max_bytes=max_bytes) for x in obj]

    # fallback: string repr
    return str(obj)

# # ---- usage ----
# # Choose whether to embed image bytes or just metadata
# EMBED_IMAGES = False  # set True if you really need the pixels

# payload = {
#     "final_assistant_text": final_assistant_text,
#     "rendered_text": rendered_text,
#     "messages": messages,        # can contain complex stuff
#     "images": images,            # list of PIL.Image or paths
# }

# with open(f"results/{idx}.json", "w", encoding="utf-8") as f:
#     json.dump(to_jsonable(payload, embed_images=EMBED_IMAGES), f, ensure_ascii=False, indent=2)

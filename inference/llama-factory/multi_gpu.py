import pprint
import os
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM
import torch
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import re
from typing import List, Dict, Any
from datasets import load_dataset
import yaml
# Add near the top imports if missing
import json
import re
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList
from format import to_jsonable
import torch.multiprocessing as mp

class StopOnEos(StoppingCriteria):
    def __init__(self, eos_ids: list[int]):
        self.eos_ids = set(i for i in eos_ids if i is not None)
    def __call__(self, input_ids, scores, **kwargs):
        # stop as soon as the last generated token matches any EOS id
        return input_ids[0, -1].item() in self.eos_ids

def _qwen_ids(processor):
    tok = processor.tokenizer
    # Try both canonical and chat-end tokens
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    eos_id = getattr(tok, "eos_token_id", None)

    ids = [i for i in {im_end_id, eos_id} if i is not None]
    if not ids:
        raise ValueError("No EOS token id found (neither <|im_end|> nor eos_token_id).")

    # Prefer to pad with EOS for decoder-only models
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        try:
            tok.pad_token = tok.eos_token if tok.eos_token is not None else "<|im_end|>"
            pad_id = tok.pad_token_id
        except Exception:
            pad_id = ids[0]

    return ids, pad_id

def render_chat_with_template(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
    add_vision_id: bool = False,
    template_str: str | None = None,
    normalize: bool = False,
) -> str:
    """
    Apply the Qwen-VL Jinja chat template to messages (and optional tools).
    - messages: list of {'role': str, 'content': str|list[segments], ...}
    - tools: OpenAI-style function tool schemas (list of dicts) or None
    - add_generation_prompt: whether to append assistant header at the end
    - add_vision_id: whether to prefix images/videos with 'Picture N:'/'Video N:'
    - template_str: override template string; defaults to CUSTOM_TEMPLATE
    - normalize: if True, converts '<image>' markers in strings to segment lists
    """
    if normalize:
        messages = update_prompt_json_like_build_messages(messages)

    # Avoid NameError if CUSTOM_TEMPLATE is not defined
    tpl = template_str or globals().get("CUSTOM_TEMPLATE", "")

    # Local import to avoid hard dependency at module import time
    from jinja2 import Environment, StrictUndefined

    env = Environment(
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    template = env.from_string(tpl)
    return template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        add_vision_id=add_vision_id,
    )

def init_qwen_vl_runtime(model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str | None = None, template_str: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto"
    ).to(device, torch.bfloat16).eval()
    processor = AutoProcessor.from_pretrained(model_name)

    # after model, processor are created
    eos_ids, pad_id = _qwen_ids(processor)
    model.generation_config.eos_token_id = eos_ids if len(eos_ids) > 1 else eos_ids[0]
    model.generation_config.pad_token_id = pad_id

    # Install the custom chat template (from arg if given; fallback to module-global if present)
    tpl = template_str
    if tpl is None:
        tpl = globals().get("CUSTOM_TEMPLATE", None)

    if tpl is not None:
        if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "chat_template"):
            processor.tokenizer.chat_template = tpl
        if hasattr(processor, "chat_template"):
            processor.chat_template = tpl

    return model, processor

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

def _parse_tool_calls_from_ids(tokenizer, response_ids: list[int]) -> tuple[str, list[dict]]:
    # Keep special tokens to not miss tags
    text = tokenizer.decode(response_ids, skip_special_tokens=False)
    calls = []
    for payload in _TOOL_CALL_RE.findall(text):
        try:
            obj = json.loads(payload.strip())
            if isinstance(obj, dict) and "name" in obj:
                calls.append(obj)
        except Exception:
            pass
    content = _TOOL_CALL_RE.sub("", text).strip()
    return content, calls

def load_openai_tools_from_yaml(yaml_path: str) -> list[dict]:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    # Each entry already matches OpenAI function tool schema: {"type":"function","function":{...}}
    return [t["tool_schema"] for t in data.get("tools", [])]

def load_iterable_parquet(
    parquet_path="/workspace/data/verl/scanqa_images_64_336x224_672x448_multiturn_format_update_1/train.parquet",
):
    return load_dataset("parquet", data_files=parquet_path)["train"]

def update_prompt_json_like_build_messages(
    prompt_json: List[Dict[str, Any]],
    image_token: str = "<image>",
    video_token: str = "<video>",
    convert_plain_text: bool = False,
) -> List[Dict[str, Any]]:
    """
    Transform string `content` fields in prompt_json into a list of segments:
    - {'type': 'image'} for each <image>
    - {'type': 'video'} for each <video>
    - {'type': 'text', 'text': '<text>'} for other text

    If `convert_plain_text` is False, only messages containing image/video markers
    are converted; otherwise, any string content becomes a single 'text' segment.

    Returns a new list (does not mutate the input list).
    """
    out = []
    pattern = re.compile(rf"({re.escape(image_token)}|{re.escape(video_token)})")

    for msg in prompt_json:
        new_msg = dict(msg)
        content = new_msg.get("content")

        if isinstance(content, str):
            if (image_token in content) or (video_token in content) or convert_plain_text:
                parts = [p for p in pattern.split(content) if p != ""]
                structured = []
                for p in parts:
                    if p == image_token:
                        structured.append({"type": "image"})
                    elif p == video_token:
                        structured.append({"type": "video"})
                    else:
                        structured.append({"type": "text", "text": p})
                new_msg["content"] = structured
        # If content is already a list or other type, leave it as-is
        out.append(new_msg)

    return out

def run_qwen_vl(
    prompt_json,
    image_items,
    *,
    model,
    processor,
    high_resolution_images: list | None = None,
    tool_config_path: str | None = None,
    max_turns: int = 3,
):
    """
    Multi-turn Qwen2.5-VL run with function/tool calls.
    - model, processor: must be provided (initialized once via init_qwen_vl_runtime)
    - tool_config_path: YAML with OpenAI-style function tool schemas (for prompting)
    - max_turns: maximum assistant turns (stops early if no tool calls)
    Returns (final_assistant_text, messages, images)
    """
    # fetch EOS/PAD once per call (or pass them in)
    eos_token_ids, pad_token_id = _qwen_ids(processor)
    stoppers = StoppingCriteriaList([StopOnEos(eos_token_ids)])

    # Prepare messages and initial images from user input
    messages = update_prompt_json_like_build_messages(prompt_json)
    source_images = [Image.open(item["image"]).convert("RGB") for item in image_items]
    images = list(source_images)  # running list aligned with all <image> markers in messages

    if high_resolution_images is not None:
        # high_resolution_images = [Image.open(item["image"]).convert("RGB") for item in high_resolution_images]
        pass

    # Load tool schemas for prompting (optional)
    tools = load_openai_tools_from_yaml(tool_config_path) if tool_config_path else None

    final_assistant_text = ""

    for _ in range(max_turns):
        # Render chat with tool schemas
        rendered_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            tools=tools,
            add_vision_id=False,
        )

        # Tokenize with all images placed in the order they appear in `messages`
        inputs = processor(text=rendered_text, images=images, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_out = model.generate(
            **inputs,
            max_new_tokens=1024,  # set to 16384 if you truly want to mirror training response_length
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,              # disable top-k in HF
            repetition_penalty=1.0,
            eos_token_id=eos_token_ids[0],
            pad_token_id=pad_token_id,
            stopping_criteria=stoppers,
            return_dict_in_generate=True,
        )

        full_seq = gen_out.sequences
        gen_only_ids = full_seq[:, prompt_len:][0].tolist()

        # Parse tool calls and assistant content
        assistant_content, tool_calls = _parse_tool_calls_from_ids(processor.tokenizer, gen_only_ids)
        final_assistant_text = processor.tokenizer.decode(gen_only_ids, skip_special_tokens=False)

        messages.append({"role": "assistant", "content": final_assistant_text})
        print("--------------------------------")
        print(final_assistant_text)

        # only execute the first tool call
        if len(tool_calls) > 1:
            print("Warning: more than one tool call found. Only executing the first one.")
            tool_calls = tool_calls[:1]

        if tool_calls:
            # Execute tools and append <tool_response> messages; also append new images
            for tc in tool_calls:
                name = tc.get("name")
                if name in {"temporal_zoom_tool"}:
                    # Use simple local tool executor that returns PIL images + tool message
                    from image_resize_tool import run_image_resize_tool_from_parsed_call
                    resized_images, tool_message = run_image_resize_tool_from_parsed_call(tc, high_resolution_images)
                    messages.append(tool_message)
                    images.extend(resized_images)
                else:
                    messages.append({"role": "tool", "content": f"Error: unknown tool '{name}'."})
            # Continue next turn after tool responses have been appended
            continue
        else:
            # No tool call: finalize this assistant response and stop
            # messages.append({"role": "assistant", "content": final_assistant_text})
            break

    return final_assistant_text, rendered_text, messages, images

def _mp_worker(rank: int, world_size: int, cfg: dict):
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    output_dir = os.path.join(cfg["OUTPUT_DIR"], f"rank{rank}")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model/processor with the provided template per process
    model, processor = init_qwen_vl_runtime(
        model_name=cfg["MODEL_NAME"],
        device=device,
        template_str=cfg.get("CUSTOM_TEMPLATE"),
    )

    parquet_data = list(load_iterable_parquet(cfg["PARQUET_PATH"]))
    print(f"[rank {rank}] total items: {len(parquet_data)}; processing shard idx % {world_size} == {rank}; using first {cfg['MAX_ITEMS']} items")

    results = []
    for idx, item in enumerate(parquet_data):
        if idx > cfg["MAX_ITEMS"]:
            break
        if (idx % world_size) != rank:
            continue

        prompt_json = [
            {"role": "user", "content": item["prompt"][0]["content"].replace("image_resize_tool", "temporal_zoom_tool")},
        ]

        final_assistant_text, rendered_text, messages, images = run_qwen_vl(
            prompt_json,
            item["images"],
            model=model,
            processor=processor,
            high_resolution_images=item["non_key_frame_image_paths"],
            tool_config_path=cfg["TOOL_CONFIG_PATH"],
            max_turns=cfg["MAX_TURNS"],
        )

        results.append({
            "question_id": item['extra_info']['question_id'],
            "question": item['extra_info']['question'],
            "gt_answers": json.loads(item['extra_info']['answer']),
            "pred_response": final_assistant_text,
        })

        payload = {
            "final_assistant_text": final_assistant_text,
            "rendered_text": rendered_text,
            "messages": messages,        # can contain complex stuff
            "images": images,            # list of PIL.Image or paths
        }
        with open(f"{output_dir}/{idx}.json", "w", encoding="utf-8") as f:
            json.dump(to_jsonable(payload, embed_images=cfg["EMBED_IMAGES"]), f, ensure_ascii=False, indent=2)

    with open(f"{output_dir}/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    TOOL_CONFIG_PATH = '/workspace/git/LLaMA-Factory/data/scanqa/temporal_zoom_tool_config.yaml'
    MAX_TURNS = 3
    EMBED_IMAGES = False  # set True if you really need the pixels

    PARQUET_PATH = '/workspace/data/verl/scanqa_images_16_keyframes_120_non_keyframes_504x504/val.parquet'
    MAX_ITEMS = 50

    MODEL_NAME = '../../pretrained/scanqa-llama-factory-sft-6000'
    OUTPUT_DIR = 'results/debug/'

    # Custom chat template (kept here; passed into each worker)
    CUSTOM_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{%- if tools %}{{- '<|im_start|>system\\n' }}{%- if messages[0]['role'] == 'system' %}{%- if messages[0]['content'] is string %}{{- messages[0]['content'] }}{%- else %}{{- messages[0]['content'][0]['text'] }}{%- endif %}{%- else %}{{- 'You are a helpful assistant.' }}{%- endif %}{{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}{%- for tool in tools %}{{- \"\\n\" }}{{- tool | tojson }}{%- endfor %}{{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}{% for message in messages %}{% if message['role'] != 'system' or loop.first == false %}{%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{%- elif message.role == \"assistant\" %}{{- '<|im_start|>' + message.role }}{%- if message.content %}{{- '\\n' + message.content }}{%- endif %}{%- for tool_call in message.tool_calls %}{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}{{- '\\n<tool_call>\\n{\"name\": \"' }}{{- tool_call.name }}{{- '\", \"arguments\": ' }}{{- tool_call.arguments | tojson }}{{- '}\\n</tool_call>' }}{%- endfor %}{{- '<|im_end|>\\n' }}{%- elif message.role == \"tool\" %}{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}{{- '<|im_start|>user' }}{%- endif %}{{- '\\n<tool_response>\\n' }}{% if message['content'] is string %}{{ message.content }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif content['type'] == 'text' or 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{{- '\\n</tool_response>' }}{%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}{{- '<|im_end|>\\n' }}{%- endif %}{%- endif %}{% endif %}{% endfor %}{%- else %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}{%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{%- elif message.role == \"assistant\" %}{{- '<|im_start|>' + message.role }}{%- if message.content %}{{- '\\n' + message.content }}{%- endif %}{%- for tool_call in message.tool_calls %}{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}{{- '\\n<tool_call>\\n{\"name\": \"' }}{{- tool_call.name }}{{- '\", \"arguments\": ' }}{{- tool_call.arguments | tojson }}{{- '}\\n</tool_call>' }}{%- endfor %}{{- '<|im_end|>\\n' }}{%- elif message.role == \"tool\" %}{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}{{- '<|im_start|>user' }}{%- endif %}{{- '\\n<tool_response>\\n' }}{% if message['content'] is string %}{{ message.content }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif content['type'] == 'text' or 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{{- '\\n</tool_response>' }}{%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}{{- '<|im_end|>\\n' }}{%- endif %}{%- endif %}{% endfor %}{%- endif %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # World size: default to number of visible GPUs, override via NUM_PROCS env
    try:
        default_world = torch.cuda.device_count() if torch.cuda.is_available() else 1
    except Exception:
        default_world = 1
    WORLD_SIZE = int(os.getenv("NUM_PROCS", default_world))
    assert WORLD_SIZE >= 1, "WORLD_SIZE must be >= 1"

    cfg = {
        "TOOL_CONFIG_PATH": TOOL_CONFIG_PATH,
        "MAX_TURNS": MAX_TURNS,
        "EMBED_IMAGES": EMBED_IMAGES,
        "PARQUET_PATH": PARQUET_PATH,
        "MAX_ITEMS": MAX_ITEMS,
        "MODEL_NAME": MODEL_NAME,
        "OUTPUT_DIR": OUTPUT_DIR,
        "CUSTOM_TEMPLATE": CUSTOM_TEMPLATE,
    }

    # Safer for CUDA than fork
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print(f"Launching {WORLD_SIZE} process(es) across available devices.")
    mp.spawn(_mp_worker, args=(WORLD_SIZE, cfg), nprocs=WORLD_SIZE, join=True)

    # Merge per-rank results
    merged = []
    for r in range(WORLD_SIZE):
        rp = os.path.join(OUTPUT_DIR, f"rank{r}", "results.json")
        if os.path.exists(rp):
            with open(rp, "r", encoding="utf-8") as f:
                merged.extend(json.load(f))
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Merged {len(merged)} results -> {os.path.join(OUTPUT_DIR, 'results.json')}")
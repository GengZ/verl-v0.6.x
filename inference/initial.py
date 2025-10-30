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


CUSTOM_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{%- if tools %}{{- '<|im_start|>system\\n' }}{%- if messages[0]['role'] == 'system' %}{%- if messages[0]['content'] is string %}{{- messages[0]['content'] }}{%- else %}{{- messages[0]['content'][0]['text'] }}{%- endif %}{%- else %}{{- 'You are a helpful assistant.' }}{%- endif %}{{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}{%- for tool in tools %}{{- \"\\n\" }}{{- tool | tojson }}{%- endfor %}{{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}{% for message in messages %}{% if message['role'] != 'system' or loop.first == false %}{%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{%- elif message.role == \"assistant\" %}{{- '<|im_start|>' + message.role }}{%- if message.content %}{{- '\\n' + message.content }}{%- endif %}{%- for tool_call in message.tool_calls %}{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}{{- '\\n<tool_call>\\n{\"name\": \"' }}{{- tool_call.name }}{{- '\", \"arguments\": ' }}{{- tool_call.arguments | tojson }}{{- '}\\n</tool_call>' }}{%- endfor %}{{- '<|im_end|>\\n' }}{%- elif message.role == \"tool\" %}{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}{{- '<|im_start|>user' }}{%- endif %}{{- '\\n<tool_response>\\n' }}{% if message['content'] is string %}{{ message.content }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif content['type'] == 'text' or 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{{- '\\n</tool_response>' }}{%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}{{- '<|im_end|>\\n' }}{%- endif %}{%- endif %}{% endif %}{% endfor %}{%- else %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}{%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{%- elif message.role == \"assistant\" %}{{- '<|im_start|>' + message.role }}{%- if message.content %}{{- '\\n' + message.content }}{%- endif %}{%- for tool_call in message.tool_calls %}{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}{{- '\\n<tool_call>\\n{\"name\": \"' }}{{- tool_call.name }}{{- '\", \"arguments\": ' }}{{- tool_call.arguments | tojson }}{{- '}\\n</tool_call>' }}{%- endfor %}{{- '<|im_end|>\\n' }}{%- elif message.role == \"tool\" %}{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}{{- '<|im_start|>user' }}{%- endif %}{{- '\\n<tool_response>\\n' }}{% if message['content'] is string %}{{ message.content }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif content['type'] == 'text' or 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{{- '\\n</tool_response>' }}{%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}{{- '<|im_end|>\\n' }}{%- endif %}{%- endif %}{% endfor %}{%- endif %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

# CUSTOM_TEMPLATE = (
#     "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}"
#     "{%- if tools %}{{- '<|im_start|>system\\n' }}"
#     "{%- if messages[0]['role'] == 'system' %}{%- if messages[0]['content'] is string %}{{- messages[0]['content'] }}"
#     "{%- else %}{{- messages[0]['content'][0]['text'] }}{%- endif %}{%- else %}{{- 'You are a helpful assistant.' }}{%- endif %}"
#     "{{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}"
#     "{%- for tool in tools %}{{- \"\\n\" }}{{- tool | tojson }}{%- endfor %}{{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}"
#     "{% for message in messages %}{% if message['role'] != 'system' or loop.first == false %}"
#     "{%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}"
#     "<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n"
#     "{% else %}{% for content in message['content'] %}"
#     "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}"
#     "{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>"
#     "{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}"
#     "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>"
#     "{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}"
#     "{%- elif message.role == \"assistant\" %}{{- '<|im_start|>' + message.role }}"
#     "{%- if message.content %}{{- '\\n' + message.content }}{%- endif %}"
#     "{%- for tool_call in message.tool_calls %}{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}"
#     "{{- '\\n<tool_call>\\n{\"name\": \"' }}{{- tool_call.name }}{{- '\", \"arguments\": ' }}{{- tool_call.arguments | tojson }}{{- '}\\n</tool_call>' }}"
#     "{%- endfor %}{{- '<|im_end|>\\n' }}"
#     "{%- elif message.role == \"tool\" %}{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}{{- '<|im_start|>user' }}{%- endif %}"
#     "{{- '\\n<tool_response>\\n' }}{% if message['content'] is string %}{{ message.content }}"
#     "{% else %}{% for content in message['content'] %}"
#     "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}"
#     "{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>"
#     "{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}"
#     "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>"
#     "{% elif content['type'] == 'text' or 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}"
#     "{{- '\\n</tool_response>' }}{%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}{{- '<|im_end|>\\n' }}{%- endif %}{%- endif %}{% endif %}{% endfor %}"
#     "{%- else %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}"
#     "{%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}"
#     "<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n"
#     "{% else %}{% for content in message['content'] %}"
#     "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}"
#     "{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>"
#     "{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}"
#     "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>"
#     "{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}"
#     "{%- elif message.role == \"assistant\" %}{{- '<|im_start|>' + message.role }}"
#     "{%- if message.content %}{{- '\\n' + message.content }}{%- endif %}"
#     "{%- for tool_call in message.tool_calls %}{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}"
#     "{{- '\\n<tool_call>\\n{\"name\": \"' }}{{- tool_call.name }}{{- '\", \"arguments\": ' }}{{- tool_call.arguments | tojson }}{{- '}\\n</tool_call>' }}"
#     "{%- endfor %}{{- '<|im_end|>\\n' }}"
#     "{%- elif message.role == \"tool\" %}{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}{{- '<|im_start|>user' }}{%- endif %}"
#     "{{- '\\n<tool_response>\\n' }}{% if message['content'] is string %}{{ message.content }}"
#     "{% else %}{% for content in message['content'] %}"
#     "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}"
#     "{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>"
#     "{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}"
#     "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>"
#     "{% elif content['type'] == 'text' or 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}"
#     "{{- '\\n</tool_response>' }}{%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}{{- '<|im_end|>\\n' }}{%- endif %}{%- endif %}{% endfor %}{%- endif %}"
#     "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
# )

# Replace the removed block with this
# TEMPLATE_PATH = (
#     Path(__file__).resolve().parents[1]
#     / "tests" / "experimental" / "agent_loop" / "qwen_vl_tool_chat_template.jinja2"
# )
# CUSTOM_TEMPLATE = TEMPLATE_PATH.read_text(encoding="utf-8")

def run_qwen_vl(prompt_json, image_items, model_name="Qwen/Qwen2.5-VL-3B-Instruct", tool_config_path: str | None = None):
    """
    Run Qwen2.5-VL-Instruct on your prompt (string with many '<image>' markers) and image list.
    """
    # Load model & processor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto").to(device)
    processor = AutoProcessor.from_pretrained(model_name)


    # after model, processor are created
    eos_ids, pad_id = _qwen_ids(processor)
    model.generation_config.eos_token_id = eos_ids if len(eos_ids) > 1 else eos_ids[0]
    model.generation_config.pad_token_id = pad_id

    # update prompt_json
    prompt_json = update_prompt_json_like_build_messages(prompt_json)

    # ⚠️ Install the custom chat template
    # Some versions read from tokenizer; others look at processor.chat_template as well.
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "chat_template"):
        processor.tokenizer.chat_template = CUSTOM_TEMPLATE
    if hasattr(processor, "chat_template"):
        processor.chat_template = CUSTOM_TEMPLATE

    # Load images (list of dicts with "image" path)
    images = [Image.open(item["image"]).convert("RGB") for item in image_items]

    # Render chat from your JSON (we pass your raw string content, so no per-item image objects here)
    if tool_config_path is not None:
        tools = load_openai_tools_from_yaml(tool_config_path)
    else:
        tools = None
    rendered_text = processor.apply_chat_template(
        prompt_json,
        add_generation_prompt=True,
        tokenize=False,
        tools=tools,          # the template has a tools branch; we're not providing tools
        add_vision_id=False  # set True if you want "Picture 1:", etc.
    )

    # Pack text + images; Qwen aligns <image> markers in text with this list
    inputs = processor(
        text=rendered_text,
        images=images,
        return_tensors="pt"
    ).to(model.device)

    # Keep a copy of the prompt token ids length for slicing later
    prompt_input_ids = inputs["input_ids"]  # shape: (1, prompt_len)
    prompt_len = prompt_input_ids.shape[1]

    # Generate
    gen_out = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        eos_token_id=151645,
        pad_token_id=151643,
        return_dict_in_generate=True
    )

    # The full sequences = [prompt_ids ... generated_ids]
    full_seq = gen_out.sequences  # shape: (1, prompt_len + generated_len)

    # Split
    gen_only_ids = full_seq[:, prompt_len:]           # just the generated continuation
    prompt_only_ids = full_seq[:, :prompt_len]        # just the original prompt (for reference)

    eos_id = 151645

    def _truncate_at_id(ids, stop_id: int):
        # ids: torch.LongTensor of shape (1, seq_len)
        hits = (ids[0] == stop_id).nonzero(as_tuple=False)
        return ids[:, :hits[0].item()+1] if hits.numel() > 0 else ids

    # gen_only_ids = _truncate_at_id(gen_only_ids, eos_id)
    # import epdb; epdb.st()

    import numpy as np
    np.savetxt('./gen_ids.txt', gen_only_ids.detach().cpu().numpy(), fmt='%d')
    np.savetxt('./prompt_only_ids.txt', prompt_only_ids.detach().cpu().numpy(), fmt='%d')

    # Decode separately
    prompt_text = processor.tokenizer.decode(
        prompt_only_ids[0], skip_special_tokens=True
    )
    generated_text = processor.tokenizer.decode(
        gen_only_ids[0], skip_special_tokens=False
    )

    # print("=== PROMPT (input only) ===")
    # print(prompt_text)
    print("\n=== GENERATED (model output only) ===")
    print(generated_text)


def single_run():
    # Example usage
    prompt_json = [
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
            "content": "Frame-0: <image>\nFrame-1: <image>\nFrame-2: <image>\nFrame-3: <image>\nFrame-4: <image>\nFrame-5: <image>\nFrame-6: <image>\nFrame-7: <image>\nFrame-8: <image>\nFrame-9: <image>\nFrame-10: <image>\nFrame-11: <image>\nFrame-12: <image>\nFrame-13: <image>\nFrame-14: <image>\nFrame-15: <image>\nFrame-16: <image>\nFrame-17: <image>\nFrame-18: <image>\nFrame-19: <image>\nFrame-20: <image>\nFrame-21: <image>\nFrame-22: <image>\nFrame-23: <image>\nFrame-24: <image>\nFrame-25: <image>\nFrame-26: <image>\nFrame-27: <image>\nFrame-28: <image>\nFrame-29: <image>\nFrame-30: <image>\nFrame-31: <image>\nFrame-32: <image>\nFrame-33: <image>\nFrame-34: <image>\nFrame-35: <image>\nFrame-36: <image>\nFrame-37: <image>\nFrame-38: <image>\nFrame-39: <image>\nFrame-40: <image>\nFrame-41: <image>\nFrame-42: <image>\nFrame-43: <image>\nFrame-44: <image>\nFrame-45: <image>\nFrame-46: <image>\nFrame-47: <image>\nFrame-48: <image>\nFrame-49: <image>\nFrame-50: <image>\nFrame-51: <image>\nFrame-52: <image>\nFrame-53: <image>\nFrame-54: <image>\nFrame-55: <image>\nFrame-56: <image>\nFrame-57: <image>\nFrame-58: <image>\nFrame-59: <image>\nFrame-60: <image>\nFrame-61: <image>\nFrame-62: <image>\nFrame-63: <image>\nAnswer the question: What is above the radiator?, Think first, call **image_resize_tool** if needed, then answer. Format strictly as <think>...</think><tool_call>...</tool_call>(if tools needed)<answer>...</answer>.The answer within <answer>...</answer> should be a short phrase, e.g., 'brown sofa'.",
            "role": "user"
        }
    ]

    # Only showing a few for brevity; include all paths in your actual run
    image_items = \
        [
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/0.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/60.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/120.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/181.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/241.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/301.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/361.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/414.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/474.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/534.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/594.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/655.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/715.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/775.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/835.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/895.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/956.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/1016.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/1076.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/1129.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/1189.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/1249.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/1309.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/1370.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/1430.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/1490.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/1550.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/1610.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/1671.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/1731.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/1791.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/1851.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/1904.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/1964.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/2024.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/2084.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/2145.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/2205.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/2265.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/2325.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/2385.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/2446.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/2506.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/2566.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/2626.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/2679.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/2739.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/2799.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/2860.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/2920.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/2980.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/3040.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/3100.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/3161.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/3221.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/3281.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/3341.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/3394.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/3454.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/3514.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/3574.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/3635.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/3695.jpg"
            },
            {
                "image": "/workspace/data/scannet_sampled/336x224/scene0331_00/images/3755.jpg"
            }
        ]

    run_qwen_vl(
        prompt_json,
         image_items,
         model_name='../pretrained/scannet_multiframe_qwen_chkpt_0_rerun/scanqa_qwen_cot_verl_2')

if __name__ == "__main__":
    for idx, item in enumerate(load_iterable_parquet()):
        if idx > 10:
            break
        prompt_json = [
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
                "content": item["prompt"][0]["content"],
                "role": "user"
            }
        ]
        print('Example ID: ', idx)
        print(item['extra_info']['answer'])
        run_qwen_vl(
            prompt_json,
            item["images"],
            # model_name='Qwen/Qwen2.5-VL-3B-Instruct',
            # model_name='../pretrained/scannet_multiframe_qwen_chkpt_0_rerun/scanqa_qwen_cot_verl_2',
            # model_name='../checkpoints/vqa_scanqa_images_multiturn_multi_choice/local_reward_0/global_step_100/actor/huggingface',
            model_name='../pretrained/scannet_multiframe_qwen_chkpt_0_rererun/scanqa_qwen_cot_verl_3',
            tool_config_path='../recipe/scanqa_multiturn_multi_choice/configs/image_resize_tool_config.yaml')
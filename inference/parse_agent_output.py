import json
import re
from typing import Any, Dict, List, Optional, Tuple

def parse_agent_output(text: str) -> Dict[str, Any]:
    """
    Parse a Qwen-style agent output that may contain:
      - <think> ... </think>
      - <tool_call> {"name": "...", "arguments": {...}} </tool_call>  (possibly multiple)
      - <answer> ... </answer>

    Returns:
      {
        "think": Optional[str],
        "tool_calls": List[{
            "name": str,
            "arguments": Dict[str, Any],
            "raw_json": str,         # original inner JSON string
            "raw_block": str         # full <tool_call>...</tool_call> block
        }],
        "answer": Optional[str]
      }
    """
    # Non-greedy, dotall parsing for tags; robust to whitespace/newlines
    tag = lambda t: rf"<{t}>\s*([\s\S]*?)\s*</{t}>"

    def _first_or_none(pattern: str) -> Optional[str]:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        return m.group(1).strip() if m else None

    think = _first_or_none(tag("think"))
    answer = _first_or_none(tag("answer"))

    # Extract all tool_call blocks; each inner is expected to be a JSON object
    tool_calls: List[Dict[str, Any]] = []
    for m in re.finditer(tag("tool_call"), text, flags=re.IGNORECASE):
        inner = m.group(1).strip()
        raw_block = m.group(0)

        # Some models may wrap JSON in code fences; strip if present
        inner_clean = inner
        if inner_clean.startswith("```") and inner_clean.endswith("```"):
            inner_clean = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", inner_clean)
            inner_clean = inner_clean[:-3].strip()

        # Try to locate a top-level JSON object inside the block if extra text exists
        obj_match = re.search(r"\{[\s\S]*\}", inner_clean)
        json_str = obj_match.group(0) if obj_match else inner_clean

        # Parse JSON and normalize arguments
        name: Optional[str] = None
        args: Dict[str, Any] = {}
        try:
            payload = json.loads(json_str)
            name = payload.get("name")
            arguments = payload.get("arguments", {})
            if isinstance(arguments, str):
                # Arguments may be serialized twice; try to decode again
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    pass
            if not isinstance(arguments, dict):
                # If still not a dict, coerce into a dict for safety
                arguments = {"_": arguments}
            args = arguments
        except Exception:
            # Return unparsed inner for debugging; caller can decide how to handle
            args = {"_unparsed": inner_clean}

        tool_calls.append(
            {
                "name": name,
                "arguments": args,
                "raw_json": json_str,
                "raw_block": raw_block,
            }
        )

    return {"think": think, "tool_calls": tool_calls, "answer": answer}

if __name__ == "__main__":
    text = '''<tool_call>{"name": "image_resize_tool", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}}</tool_call>{"name": "answer", "arguments": {"frames": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,'''
    # print(parse_agent_output(text))

    ans = parse_agent_output(text)
    print(ans['tool_calls'][0])
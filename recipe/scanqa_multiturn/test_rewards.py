from deepeyes import compute_score

def run_case(name: str, solution_str: str, ground_truth: str, extra_info: dict) -> float:
    score = compute_score("common_reasoning", solution_str, ground_truth, extra_info)
    print(f"{name}: {score:.3f}")
    return score

if __name__ == "__main__":
    # Case 1: No tool, well-formatted, correct
    sol_no_tool_good = "<think>Quick reasoning...</think><answer>left</answer>"
    gt_left = '["left", "to the left"]'
    info_left = {
        "question": "Is the woman to the left or to the right of the man who is holding the camera?",
    }
    s1 = run_case("No tool, well-formatted, correct", sol_no_tool_good, gt_left, info_left)

    # Case 2: Tool used, missing <answer> tags (format error), but correct textual content
    sol_tool_format_err = """<tool_call>
{"name": "image_resize_tool", "arguments": {"timestamp": -1}}
</tool_call>user
<tool_response>
Selected image at timestamp -1.
</tool_response>
assistant
Yes, the white van is indeed situated in the bottom part of the picture."""
    gt_van = '["Yes, the white van is indeed situated in the bottom part of the picture."]'
    info_van = {
        "question": "Is the white van in the bottom part of the picture?",
    }
    s2 = run_case("Tool used, format error, correct content", sol_tool_format_err, gt_van, info_van)

    # Case 3: Tool used, well-formatted, correct
    sol_tool_good = """<think>
I need to review a different frame to answer confidently.
</think>
<tool_call>
{"name": "image_resize_tool", "arguments": {"timestamp": -1}}
</tool_call>
<tool_response>
Selected image at timestamp -1.
</tool_response>
<answer>Yes, the white van is indeed situated in the bottom part of the picture.</answer>"""
    s3 = run_case("Tool used, well-formatted, correct", sol_tool_good, gt_van, info_van)

    # Case 4: Very long answer triggers length penalty
    long_text = "a" * 1200
    sol_long = f"<think>...</think><answer>{long_text}</answer>"
    s4 = run_case("Very long answer (penalty)", sol_long, '["a"]', {"question": "Dummy?"})

    # Case 5: Tool used, well-formatted, wrong answer (tests tool gating if acc_reward reflects EM/F1)
    sol_tool_wrong = """<think>...</think>
<tool_call>{"name": "image_resize_tool", "arguments": {"timestamp": -1}}</tool_call>
<tool_response>Selected image at timestamp -1.</tool_response>
<answer>right</answer>"""
    s5 = run_case("Tool used, well-formatted, wrong answer", sol_tool_wrong, gt_left, info_left)

    # Basic relational checks (robust across current and EM/F1-restored variants)
    try:
        assert s3 > s2, "Well-formatted with tool should score higher than format-error with tool."
        assert s3 > s1, "Tool usage (with proper format) should improve score over no tool."
        assert s4 < s1, "Very long answers should be penalized below normal cases."
    except AssertionError as e:
        print(f"Assertion failed: {e}")

    print("\nScores summary:")
    print(f"- No tool, well-formatted, correct: {s1:.3f}")
    print(f"- Tool used, format error, correct content: {s2:.3f}")
    print(f"- Tool used, well-formatted, correct: {s3:.3f}")
    print(f"- Very long answer (penalty): {s4:.3f}")
    print(f"- Tool used, well-formatted, wrong answer: {s5:.3f}")
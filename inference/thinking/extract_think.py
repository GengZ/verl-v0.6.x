import re
from typing import Optional

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)

def extract_last_think(text: str) -> Optional[str]:
    """
    Extract the content of the last <think>...</think> block, excluding the tags.
    Returns None if no such block exists.
    """
    last_match = None
    for m in THINK_PATTERN.finditer(text):
        last_match = m
    return None if last_match is None else last_match.group(1).strip()

SUMMARY_PATTERN = re.compile(r"<evidence_summary>(.*?)</evidence_summary>", flags=re.DOTALL | re.IGNORECASE)

def extract_last_summary(text: str) -> Optional[str]:
    """
    Extract the content of the last <evidence_summary>...</evidence_summary> block, excluding the tags.
    Returns None if no such block exists.
    """
    last_match = None
    for m in SUMMARY_PATTERN.finditer(text):
        last_match = m
    return None if last_match is None else last_match.group(1).strip()

if __name__ == "__main__":
    sample = """system
    You are an expert multimodal reasoning assistant...
    assistant
    <think>
    In frame 7, we see a beige wooden bookshelf filled with books and DVDs...
    I need to look closer at between frame 7 and frame 8 to take a deeper look.
    </think>
    <answer>
    intervals=[7]; summary=The transition highlights...
    </answer>"""

    print(extract_last_think(sample))
    # -> returns the inner text between <think> and </think>, without the tags

